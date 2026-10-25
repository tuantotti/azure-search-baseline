"""Durable indexing lifecycle and deterministic benchmark orchestration."""

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .api import AgenticStrategy, ApiError, HybridStrategy, index_definition
from .dataset import Dataset, question_hash
from .evaluate import score, write_reports


class ExperimentClient(Protocol):
    def request(self, method: str, path: str, body: dict | None = None,
                *, retry: bool = False) -> dict: ...
    def embeddings(self, texts: list[str]) -> tuple[str, list[list[float]]]: ...


@dataclass(frozen=True)
class Config:
    base_url: str
    vectorizer_endpoint: str
    vectorizer_deployment: str
    vectorizer_model: str | None = None
    question_limit: int = 100
    repeats: int = 3
    seed: int = 42
    train_limit: int = 0
    batch_size: int = 32
    readiness_timeout: float = 300
    http_timeout: float = 180


def resource_fingerprint(resource: dict) -> str:
    def stable(value):
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items() if k not in ('@odata.etag', 'etag')}
        if isinstance(value, list):
            return [stable(v) for v in value]
        return value
    return hashlib.sha256(json.dumps(stable(resource), sort_keys=True).encode()).hexdigest()


def checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class Experiment:
    def __init__(self, app: ExperimentClient, dataset: Dataset, config: Config,
                 directory: Path, *, resume: bool = False, sleep=time.sleep):
        self.app, self.dataset, self.config, self.directory = app, dataset, config, directory
        self.sleep = sleep
        self.queries = dataset.sample(config.question_limit, config.seed)
        self.judgments = dataset.judgments(self.queries)
        self.corpus = {r.key: r for r in dataset.training}
        self.rows = []
        identity = {'version': 1, 'config': asdict(config), 'dataset_hashes': dataset.hashes,
                    'sampled_ids': [q.id for q in self.queries],
                    'training_ids': [r.id for r in dataset.training]}
        manifest_path = directory / 'manifest.json'
        if resume:
            self.manifest = json.loads(manifest_path.read_text())
            if any(self.manifest.get(k) != v for k, v in identity.items()):
                raise ValueError('Resume configuration, dataset or sample differs from manifest')
            self.progress = json.loads((directory / 'progress.json').read_text())
            results = directory / 'results.jsonl'
            if results.exists():
                raw = results.read_bytes()
                # A crash can interrupt only the final append. Preserve every complete line.
                if raw and not raw.endswith(b'\n'):
                    raw = raw[:raw.rfind(b'\n') + 1]
                    results.write_bytes(raw)
                self.rows = [json.loads(line) for line in raw.splitlines()]
            if not set(self.progress['uploaded']) <= set(self.corpus):
                raise ValueError('Checkpoint contains unknown uploaded keys')
            seen = set()
            for row in self.rows:
                key = (row['query_id'], row['repeat'], row['method'])
                if (key in seen or row['query_id'] not in identity['sampled_ids']
                        or row['method'] not in ('hybrid', 'hybrid_semantic', 'agentic')
                        or not 0 <= row['repeat'] < config.repeats):
                    raise ValueError('Invalid or duplicate result checkpoint')
                seen.add(key)
        else:
            directory.mkdir(parents=True, exist_ok=False)
            prefix = directory.name
            self.manifest = {**identity, 'resources': {
                'index': f'{prefix}-index', 'source': f'{prefix}-source', 'base': f'{prefix}-base'}}
            self.progress = {'created': [], 'uploaded': [], 'warmups': [], 'setup_seconds': 0}
            checkpoint(manifest_path, self.manifest)
            checkpoint(directory / 'progress.json', self.progress)
        self.resources = self.manifest['resources']

    def save_progress(self):
        checkpoint(self.directory / 'progress.json', self.progress)

    def create(self, kind: str, route: str, body: dict) -> None:
        path = f'/{route}/{self.resources[kind]}'
        if kind in self.progress['created']:
            resource = self.app.request('GET', path, retry=True)['azure']
            expected = self.progress.get('resource_fingerprints', {}).get(kind)
            if expected is None or resource_fingerprint(resource) != expected:
                raise ValueError(f'Resource configuration changed since checkpoint: {kind}')
            return
        # Never adopt an ambiguous write after a crash: require operator inspection/new run.
        if self.progress.get('pending_create'):
            raise RuntimeError('Ambiguous resource creation checkpoint; inspect resources and '
                               'start a fresh run. No resource was overwritten or deleted.')
        try:
            self.app.request('GET', path, retry=True)
        except ApiError as exc:
            if exc.status != 404:
                raise
        else:
            raise RuntimeError(f'Resource already exists: {path}')
        self.progress['pending_create'] = kind
        self.save_progress()
        resource = self.app.request('POST', path, body, retry=True)['azure']
        self.progress.setdefault('resource_fingerprints', {})[kind] = resource_fingerprint(resource)
        self.progress['created'].append(kind)
        self.progress.pop('pending_create')
        self.save_progress()

    def wait_ready(self, keys: list[str], *, expected_count: int | None = None) -> None:
        deadline = time.monotonic() + self.config.readiness_timeout
        while True:
            visible = True
            for start in range(0, len(keys), 100):
                batch = keys[start:start + 100]
                result = self.app.request('POST', f"/indexes/{self.resources['index']}/search", {
                    'search': '*', 'filter': f"search.in(chunk_id, '{','.join(batch)}', ',')",
                    'select': 'chunk_id', 'top': len(batch)}, retry=True)['azure']
                if {row['chunk_id'] for row in result['value']} != set(batch):
                    visible = False
                    break
            if visible and expected_count is not None:
                result = self.app.request('POST', f"/indexes/{self.resources['index']}/search", {
                    'search': '*', 'count': True, 'top': 1}, retry=True)['azure']
                visible = result.get('@odata.count') == expected_count
            if visible:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('Index count/key visibility readiness timed out')
            self.sleep(2)

    def ingest(self, records) -> None:
        uploaded = set(self.progress['uploaded'])
        pending = [r for r in records if r.key not in uploaded]
        for offset in range(0, len(pending), self.config.batch_size):
            batch = pending[offset:offset + self.config.batch_size]
            docs = [r.document() for r in batch]
            model, vectors = self.app.embeddings([d['content'] for d in docs])
            if model != self.manifest['embedding']['response_model'] or any(
                len(v) != self.manifest['embedding']['dimensions'] for v in vectors
            ):
                raise ApiError('Embedding model/dimensions changed during ingestion')
            outstanding = {d['chunk_id']: {**d, 'content_vector': v, '@search.action': 'upload'}
                           for d, v in zip(docs, vectors, strict=True)}
            for attempt in range(3):
                result = self.app.request('POST',
                    f"/indexes/{self.resources['index']}/documents",
                    {'value': list(outstanding.values())}, retry=True)['azure']['value']
                if len(result) != len(outstanding) or {r['key'] for r in result} != set(outstanding):
                    raise ApiError('Upload response keys differ from request')
                permanent = False
                for status in result:
                    if status.get('status') is True:
                        uploaded.add(status['key'])
                        outstanding.pop(status['key'])
                    elif status.get('statusCode') not in (408, 429, 500, 502, 503, 504):
                        permanent = True
                self.progress['uploaded'] = sorted(uploaded)
                self.save_progress()
                if not outstanding:
                    break
                if permanent or attempt == 2:
                    raise ApiError('Document upload failed; successful keys checkpointed')
                self.sleep(2 ** attempt)

    def setup(self):
        started = time.perf_counter()
        response_model, vectors = self.app.embeddings(['Embedding compatibility probe'])
        model = self.config.vectorizer_model or response_model
        recognized_models = ('text-embedding-ada-002', 'text-embedding-3-small',
                             'text-embedding-3-large')
        if response_model in recognized_models and model != response_model:
            raise ValueError('Vectorizer model contradicts the recognized embedding response model')
        if model not in recognized_models:
            raise ValueError('Cannot infer Azure vectorizer model; set --vectorizer-model')
        embedding = {'response_model': response_model, 'vectorizer_model': model,
                     'dimensions': len(vectors[0])}
        if self.manifest.get('embedding', embedding) != embedding:
            raise ValueError('Embedding configuration changed since checkpoint')
        self.manifest['embedding'] = embedding
        checkpoint(self.directory / 'manifest.json', self.manifest)
        self.create('index', 'indexes', index_definition(self.resources['index'], len(vectors[0]),
                    self.config.vectorizer_endpoint, self.config.vectorizer_deployment, model))
        if not self.progress.get('vectorizer_validated'):
            self.ingest(self.dataset.training[:1])
            self.wait_ready([self.dataset.training[0].key])
            self.app.request('POST', f"/indexes/{self.resources['index']}/search", {
                'vectorQueries': [{'kind': 'text', 'text': 'Embedding compatibility probe',
                                   'fields': 'content_vector', 'k': 1}],
                'top': 1, 'select': 'chunk_id'}, retry=True)
            self.progress['vectorizer_validated'] = True
            self.save_progress()
        self.ingest(self.dataset.training)
        self.wait_ready(list(self.corpus), expected_count=len(self.corpus))
        self.create('source', 'knowledge-sources', {
            'index_name': self.resources['index'], 'description': self.directory.name,
            'semantic_configuration_name': 'semantic-config',
            'search_fields': ['title', 'content'],
            'source_data_fields': ['chunk_id', 'document_id', 'title', 'content']})
        self.create('base', 'knowledge-bases', {
            'knowledge_source_names': [self.resources['source']], 'description': self.directory.name})
        self.progress['setup_seconds'] += time.perf_counter() - started
        self.save_progress()

    def run(self) -> bool:
        expected = len(self.queries) * self.config.repeats * 3
        phase = 'setup'
        started_setup = time.perf_counter()
        try:
            self.setup()
            self.progress.pop('failure', None)
            phase = 'retrieval'
            strategies = [HybridStrategy(self.app, self.resources['index'], set(self.corpus)),
                          HybridStrategy(self.app, self.resources['index'], set(self.corpus),
                                         semantic=True),
                          AgenticStrategy(self.app, self.resources['base'], self.resources['source'],
                                          set(self.corpus))]
            for strategy in strategies:
                if strategy.name not in self.progress['warmups']:
                    # Record even unsuccessful warm-ups separately; never include in metrics.
                    try:
                        strategy.retrieve(self.queries[0].question, self.queries[0].exclusion)
                    except (ApiError, ValueError, KeyError) as exc:
                        self.progress.setdefault('warmup_errors', {})[strategy.name] = type(exc).__name__
                    self.progress['warmups'].append(strategy.name)
                    self.save_progress()
            completed = {(r['query_id'], r['repeat'], r['method']) for r in self.rows}
            rng = random.Random(self.config.seed)
            with (self.directory / 'results.jsonl').open('a') as handle:
                for repeat in range(self.config.repeats):
                    for query in self.queries:
                        ordered = rng.sample(strategies, len(strategies))
                        for strategy in ordered:
                            if (query.id, repeat, strategy.name) in completed:
                                continue
                            row = {'query_id': query.id, 'kind': query.kind,
                                   'repeat': repeat, 'method': strategy.name, 'keys': [],
                                   'metrics': {}, 'success': False}
                            started = time.perf_counter()
                            try:
                                result = strategy.retrieve(query.question, query.exclusion)
                                row['latency_ms'] = (time.perf_counter() - started) * 1000
                                if any(self.corpus[key].base_id == query.base_id or
                                       question_hash(self.corpus[key].question) ==
                                       question_hash(query.question)
                                       for key in result.keys):
                                    raise ApiError('Retrieval returned an excluded record')
                                row.update(asdict(result))
                                row['metrics'] = score(query, result.keys,
                                                       self.judgments[query.id], self.corpus)
                                row['success'] = True
                            except (ApiError, ValueError, KeyError, TypeError) as exc:
                                row['error'] = str(exc) if isinstance(exc, ApiError) else type(exc).__name__
                            row.setdefault('latency_ms', (time.perf_counter() - started) * 1000)
                            handle.write(json.dumps(row) + '\n')
                            handle.flush()
                            self.rows.append(row)
        except Exception as exc:
            self.progress['failure'] = {
                'phase': phase, 'type': type(exc).__name__,
                'message': str(exc) if isinstance(exc, (ApiError, ValueError, RuntimeError))
                else 'Benchmark interrupted by an unexpected error',
            }
            if phase == 'setup':
                self.progress['setup_seconds'] += time.perf_counter() - started_setup
            self.save_progress()
            raise
        finally:
            summary = write_reports(self.directory, self.dataset, self.queries, self.judgments,
                                    self.rows, expected, self.progress['setup_seconds'],
                                    failure=self.progress.get('failure'),
                                    smoke=self.config.train_limit > 0)
        return summary['complete']
