"""HTTP adapter and interchangeable retrieval strategies for the running app."""

import math
import time
from dataclasses import dataclass
from typing import Protocol

import httpx
from loguru import logger

from app.core.logging import log_error, safe_error_text
from app.core.redaction import redact_credentials


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class AppClient(Protocol):
    def request(self, method: str, path: str, body: dict | None = None,
                *, retry: bool = False) -> dict: ...


class HttpAppClient:
    def __init__(self, client: httpx.Client, attempts: int = 3, sleep=time.sleep):
        self.client, self.attempts, self.sleep = client, attempts, sleep

    def request(self, method: str, path: str, body: dict | None = None,
                *, retry: bool = False) -> dict:
        for attempt in range(self.attempts if retry else 1):
            delay = min(2 ** attempt, 10)
            try:
                response = self.client.request(method, path, json=body)
                if response.is_success:
                    try:
                        payload = response.json()
                        if not isinstance(payload, dict):
                            raise ValueError()
                        return payload
                    except ValueError as exc:
                        raise ApiError('App returned invalid JSON') from exc
                error = ApiError(f'App HTTP {response.status_code} during {method} {path}',
                                 response.status_code)
                try:
                    detail = response.json().get('detail', 'No error detail')
                except (ValueError, AttributeError):
                    detail = 'Non-JSON error response'
                detail = safe_error_text(redact_credentials(detail))[:4000]
                request_id = response.headers.get('X-Request-ID')
                logger.bind(method=method, path=path, status=response.status_code,
                            request_id=request_id, attempt=attempt + 1).error(
                    'App request failed: {}', detail)
                error = ApiError(f'{error}; request_id={request_id}; detail={detail}',
                                 response.status_code)
                transient = response.status_code in (408, 429, 500, 502, 503, 504)
                try:
                    delay = min(30, max(delay, float(response.headers.get('Retry-After', 0))))
                except ValueError:
                    pass
            except httpx.TransportError as exc:
                log_error(exc, service='app_http', method=method, path=path,
                          attempt=attempt + 1)
                error, transient = ApiError(f'App transport failure during {method} {path}'), True
            if not retry or not transient or attempt == self.attempts - 1:
                raise error
            self.sleep(delay)
        raise AssertionError('Unreachable retry state')

    def embeddings(self, texts: list[str]) -> tuple[str, list[list[float]]]:
        if not 1 <= len(texts) <= 32:
            raise ValueError('Embedding batches must contain 1..32 texts')
        payload = self.request('POST', '/embeddings', {'input': texts}, retry=True)['azure']
        rows = sorted(payload['data'], key=lambda row: row['index'])
        if [row['index'] for row in rows] != list(range(len(texts))):
            raise ApiError('Embedding response has missing or duplicate indices')
        vectors = [row['embedding'] for row in rows]
        if not vectors[0] or any(len(v) != len(vectors[0]) or
            any(isinstance(n, bool) or not isinstance(n, (float, int)) or not math.isfinite(n)
                for n in v) for v in vectors):
            raise ApiError('Embedding response contains invalid vectors')
        return payload['model'], [[float(n) for n in v] for v in vectors]


@dataclass
class RetrievalResult:
    keys: list[str]
    server_ms: float | None
    embedding_ms: float | None
    activity: list[dict]


class RetrievalStrategy(Protocol):
    name: str
    def retrieve(self, question: str, exclusion: str) -> RetrievalResult: ...


def normalize(payload: dict, known_keys: set[str], *, agentic: bool = False) -> RetrievalResult:
    azure = payload['azure']
    rows = azure.get('references') if agentic else azure.get('value')
    if not isinstance(rows, list):
        raise ApiError('Missing retrieval references/results')
    keys = []
    for row in rows:
        key = row.get('docKey' if agentic else 'chunk_id')
        if not isinstance(key, str) or key not in known_keys:
            raise ApiError('Retrieval returned an unknown or missing document key')
        if key not in keys:
            keys.append(key)
    activity = azure.get('activity') or []
    if any(item.get('error') or item.get('type') == 'error' for item in activity):
        raise ApiError('Agentic activity contains a partial retrieval failure')
    return RetrievalResult(keys, payload.get('elapsed_ms'), payload.get('embedding_ms'), activity)


class HybridStrategy:
    def __init__(self, app: AppClient, index: str, known_keys: set[str], *, semantic=False):
        self.app, self.index, self.known_keys = app, index, known_keys
        self.name = 'hybrid_semantic' if semantic else 'hybrid'

    def retrieve(self, question: str, exclusion: str) -> RetrievalResult:
        payload = self.app.request('POST', f'/indexes/{self.index}/search/text', {
            'query': question, 'mode': self.name, 'vector_field': 'content_vector', 'k': 50,
            'semantic_configuration': 'semantic-config',
            'parameters': {'top': 50, 'filter': exclusion, 'select': ['chunk_id'],
                           'search_fields': ['title', 'content']}})
        return normalize(payload, self.known_keys)


class AgenticStrategy:
    name = 'agentic'

    def __init__(self, app: AppClient, base: str, source: str, known_keys: set[str]):
        self.app, self.base, self.source, self.known_keys = app, base, source, known_keys

    def retrieve(self, question: str, exclusion: str) -> RetrievalResult:
        payload = self.app.request('POST', f'/knowledge-bases/{self.base}/retrieve', {
            'messages': [{'role': 'user', 'content': question}], 'max_output_documents': 50,
            'knowledge_source_params': [{'knowledge_source_name': self.source,
                                          'filter_add_on': exclusion}]})
        return normalize(payload, self.known_keys, agentic=True)


def index_definition(name: str, dimensions: int, endpoint: str, deployment: str,
                     model: str) -> dict:
    fields = [
        {'name': field, 'type': 'Edm.String', 'key': field == 'chunk_id',
         'searchable': field in ('title', 'content'), 'retrievable': True,
         'filterable': field not in ('title', 'content')}
        for field in ('chunk_id', 'document_id', 'webqsp_id', 'question_hash',
                      'composition_type', 'title', 'content')]
    fields += [{'name': 'answer_ids', 'type': 'Collection(Edm.String)', 'retrievable': True},
               {'name': 'content_vector', 'type': 'Collection(Edm.Single)',
                'searchable': True, 'retrievable': False, 'dimensions': dimensions,
                'vectorSearchProfile': 'vector-profile'}]
    return {'name': name, 'fields': fields, 'vectorSearch': {
        'algorithms': [{'name': 'hnsw', 'kind': 'hnsw', 'hnswParameters': {'metric': 'cosine'}}],
        'profiles': [{'name': 'vector-profile', 'algorithm': 'hnsw',
                      'vectorizer': 'cwq-vectorizer'}],
        'vectorizers': [{'name': 'cwq-vectorizer', 'kind': 'azureOpenAI',
                         'azureOpenAIParameters': {'resourceUri': endpoint,
                                                  'deploymentId': deployment,
                                                  'modelName': model}}]},
        'semantic': {'defaultConfiguration': 'semantic-config', 'configurations': [
            {'name': 'semantic-config', 'prioritizedFields': {
                'titleField': {'fieldName': 'title'},
                'prioritizedContentFields': [{'fieldName': 'content'}]}}]}}
