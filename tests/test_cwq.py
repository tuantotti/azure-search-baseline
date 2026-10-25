"""Offline behavioral tests against a stateful fake app HTTP API."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import httpx

from app.schemas.agentic import (
    AgenticRetrievalRequest,
    KnowledgeBaseRequest,
    KnowledgeSourceRequest,
)
from app.schemas.search import RawSearchRequest, TextSearchRequest
from experiments.api import ApiError, HttpAppClient, normalize
from experiments.dataset import Dataset, Record, question_hash
from experiments.evaluate import score
from experiments.experiment import Config, Experiment


class FakeApp:
    def __init__(self):
        self.resources, self.docs, self.calls = {}, {}, []
        self.fail_retrieval = False
        self.failed_key = None
        self.delay_visibility = 0
        self.embedding_model = 'text-embedding-3-small'

    def __call__(self, request):
        path, method = request.url.path, request.method
        body = json.loads(request.content) if request.content else None
        self.calls.append((method, path, body))
        if method == 'POST':
            if path.endswith('/search'):
                RawSearchRequest.model_validate(body)
            elif path.endswith('/search/text'):
                TextSearchRequest.model_validate(body)
            elif path.endswith('/retrieve'):
                AgenticRetrievalRequest.model_validate(body)
            elif path.startswith('/knowledge-sources/'):
                KnowledgeSourceRequest.model_validate(body)
            elif path.startswith('/knowledge-bases/'):
                KnowledgeBaseRequest.model_validate(body)
        if path == '/embeddings':
            return httpx.Response(200, json={'azure': {'model': self.embedding_model,
                'data': [{'index': i, 'embedding': [1., 0., 0.]} for i in
                         reversed(range(len(body['input'])))]}})
        if path.endswith('/documents'):
            statuses = []
            for doc in body['value']:
                key = doc['chunk_id']
                failed = key == self.failed_key
                if not failed:
                    self.docs[key] = doc
                statuses.append({'key': key, 'status': not failed,
                                 'statusCode': 400 if failed else 200})
            return httpx.Response(200, json={'azure': {'value': statuses}})
        if path.endswith('/search'):
            docs = list(self.docs.values())
            if body.get('filter'):
                docs = [d for d in docs if d['chunk_id'] in body['filter']]
            if self.delay_visibility:
                self.delay_visibility -= 1
                docs = []
            return httpx.Response(200, json={'azure': {'value': docs,
                                                      '@odata.count': len(self.docs)}})
        if path.endswith(('/search/text', '/retrieve')):
            if self.fail_retrieval:
                return httpx.Response(503)
            keys = list(self.docs)
            if path.endswith('/retrieve'):
                azure = {'references': [{'id': str(i), 'docKey': k} for i, k in enumerate(keys)],
                         'activity': [{'type': 'modelQueryPlanning', 'inputTokens': 10,
                                       'outputTokens': 2}]}
            else:
                azure = {'value': [{'chunk_id': k} for k in keys]}
            return httpx.Response(200, json={'azure': azure, 'elapsed_ms': 1, 'embedding_ms': 2})
        if method == 'GET':
            return httpx.Response(200, json={'azure': self.resources[path]}) \
                if path in self.resources else httpx.Response(404)
        if path in self.resources:
            return httpx.Response(409)
        self.resources[path] = body
        return httpx.Response(201, json={'azure': body})


def fixture(count=35):
    training = [Record(f'train-{i}', f'base-{i}', f'Question {i}', 'composition',
                       frozenset({'a'}), ('Answer',)) for i in range(count)]
    dev = [Record('dev-1', 'dev-base', 'Dev question?', 'composition', frozenset({'a'}), ('Answer',))]
    return Dataset(training, dev, {'train': 'hash-a', 'dev': 'hash-b'})


class DatasetTests(unittest.TestCase):
    def test_normalization_sampling_exclusions_and_no_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = [{'ID': f'id-{i}', 'webqsp_ID': f"base'{i}", 'question': f'Question {i}',
                     'compositionality_type': 'composition' if i < 8 else 'conjunction',
                     'machine_question': 'NEVER', 'sparql': 'NEVER',
                     'answers': [{'answer_id': 'a', 'answer': None, 'aliases': ['', 'Alias']}]} 
                    for i in range(10)]
            for split in ('train', 'dev'):
                (directory / f'ComplexWebQuestions_{split}.json').write_text(json.dumps(rows))
            data = Dataset.load(directory)
            self.assertEqual(data.training[0].labels, ('Alias',))
            self.assertNotIn('NEVER', json.dumps(data.training[0].document()))
            self.assertIn("base''0", data.training[0].exclusion)
            self.assertEqual(question_hash(' HELLO\u00a0WORLD '), question_hash('hello world'))
            sample = data.sample(5, 42)
            self.assertEqual(sample, data.sample(5, 42))
            self.assertEqual(sum(r.kind == 'conjunction' for r in sample), 1)
            self.assertNotIn(data.training[0].key, data.judgments(data.development)[rows[0]['ID']])

    def test_metrics_uncovered_and_duplicates(self):
        data = fixture(2)
        q = data.development[0]
        corpus = {r.key: r for r in data.training}
        keys = list(corpus)
        result = score(q, [keys[0], keys[0]], set(keys), corpus)
        self.assertEqual(result['document_recall@5'], .5)
        self.assertEqual(result['answer_hit@1'], 1)
        result = score(q, [], set(), corpus)
        self.assertIsNone(result['ndcg@10'])
        self.assertEqual(result['answer_hit@10'], 0)

    def test_reference_keys_are_not_citation_ids(self):
        result = normalize({'azure': {'references': [{'id': '0', 'docKey': 'actual'},
                                                      {'id': '1', 'docKey': 'actual'}]}},
                           {'actual'}, agentic=True)
        self.assertEqual(result.keys, ['actual'])
        with self.assertRaises(ApiError):
            normalize({'azure': {'references': [{'id': 'actual'}]}}, {'actual'}, agentic=True)


class WorkflowTests(unittest.TestCase):
    def test_full_workflow_batches_readiness_reports_and_resume(self):
        fake = FakeApp()
        fake.delay_visibility = 1
        with tempfile.TemporaryDirectory() as tmp, httpx.Client(
            base_url='http://app', transport=httpx.MockTransport(fake)
        ) as client:
            app = HttpAppClient(client, sleep=lambda _: None)
            config = Config('http://app', 'https://example.openai.azure.com', 'embedding', repeats=2)
            directory = Path(tmp) / 'cwq-test'
            run = Experiment(app, fixture(), config, directory, sleep=lambda _: None)
            self.assertTrue(run.run())
            self.assertEqual(len(run.rows), 6)
            self.assertEqual(len(fake.docs), 35)
            embedding_batches = [len(b['input']) for _, p, b in fake.calls if p == '/embeddings']
            self.assertEqual(embedding_batches, [1, 1, 32, 2])
            vector_position = next(i for i, (_, p, b) in enumerate(fake.calls)
                                   if p.endswith('/search') and b.get('vectorQueries'))
            bulk_position = next(i for i, (_, p, b) in enumerate(fake.calls)
                                 if p.endswith('/documents') and len(b['value']) > 1)
            self.assertLess(vector_position, bulk_position)
            retrievals = [(p, b) for _, p, b in fake.calls if p.endswith(('/retrieve', '/search/text'))]
            self.assertEqual(len(retrievals), 9)
            for path, body in retrievals:
                if path.endswith('/retrieve'):
                    self.assertEqual(body['max_output_documents'], 50)
                    self.assertEqual(body['messages'][0]['content'], 'Dev question?')
                else:
                    self.assertEqual(body['k'], 50)
            for file in ('manifest.json', 'progress.json', 'results.jsonl', 'summary.json',
                         'per-query.csv', 'report.md'):
                self.assertTrue((directory / file).exists())
            resumed = Experiment(app, fixture(), config, directory, resume=True, sleep=lambda _: None)
            self.assertTrue(resumed.run())
            self.assertEqual(len([p for _, p, _ in fake.calls
                                  if p.endswith(('/retrieve', '/search/text'))]), 9)
            bad = fixture()
            bad.hashes['train'] = 'changed'
            with self.assertRaises(ValueError):
                Experiment(app, bad, config, directory, resume=True)

    def test_failures_are_not_retried_and_partial_reports_saved(self):
        fake = FakeApp()
        fake.fail_retrieval = True
        with tempfile.TemporaryDirectory() as tmp, httpx.Client(
            base_url='http://app', transport=httpx.MockTransport(fake)
        ) as client:
            run = Experiment(HttpAppClient(client), fixture(2),
                             Config('http://app', 'https://example.openai.azure.com', 'embed', repeats=1),
                             Path(tmp) / 'cwq-failure')
            self.assertFalse(run.run())
            self.assertEqual(len(run.rows), 3)
            self.assertTrue(all(not r['success'] for r in run.rows))
            self.assertEqual(len([p for _, p, _ in fake.calls
                                  if p.endswith(('/retrieve', '/search/text'))]), 6)

    def test_partial_ingestion_and_conflicts(self):
        fake = FakeApp()
        data = fixture(3)
        fake.failed_key = data.training[2].key
        with tempfile.TemporaryDirectory() as tmp, httpx.Client(
            base_url='http://app', transport=httpx.MockTransport(fake)
        ) as client:
            run = Experiment(HttpAppClient(client), data,
                             Config('http://app', 'https://example.openai.azure.com', 'embed'),
                             Path(tmp) / 'cwq-partial')
            with self.assertRaises(ApiError):
                run.run()
            self.assertEqual(len(run.progress['uploaded']), 2)
            self.assertTrue((run.directory / 'report.md').exists())
            summary = json.loads((run.directory / 'summary.json').read_text())
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['failure']['phase'], 'setup')
            self.assertIn('upload failed', summary['failure']['message'])
            self.assertIn('upload failed', (run.directory / 'report.md').read_text())

    def test_setup_retries_but_retrieval_does_not(self):
        calls = []
        def transport(request):
            calls.append(request)
            return httpx.Response(503 if len(calls) < 3 else 200, json={})
        with httpx.Client(base_url='http://app', transport=httpx.MockTransport(transport)) as client:
            app = HttpAppClient(client, sleep=lambda _: None)
            app.request('GET', '/health', retry=True)
            self.assertEqual(len(calls), 3)

    def test_resource_conflict_and_changed_schema_fail_without_overwrite(self):
        fake = FakeApp()
        with tempfile.TemporaryDirectory() as tmp, httpx.Client(
            base_url='http://app', transport=httpx.MockTransport(fake)
        ) as client:
            config = Config('http://app', 'https://example.openai.azure.com', 'embed', repeats=1)
            directory = Path(tmp) / 'cwq-resume'
            app = HttpAppClient(client)
            run = Experiment(app, fixture(2), config, directory)
            self.assertTrue(run.run())
            fake.resources['/indexes/cwq-resume-index']['fields'][0]['type'] = 'Edm.Int32'
            resumed = Experiment(app, fixture(2), config, directory, resume=True)
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                resumed.run()
            conflict = Experiment(app, fixture(2), config, Path(tmp) / 'cwq-conflict')
            fake.resources['/indexes/cwq-conflict-index'] = {'name': 'unrelated'}
            with self.assertRaisesRegex(RuntimeError, 'already exists'):
                conflict.run()
            self.assertEqual(fake.resources['/indexes/cwq-conflict-index'], {'name': 'unrelated'})

    def test_readiness_timeout_checks_key_visibility(self):
        fake = FakeApp()
        with tempfile.TemporaryDirectory() as tmp, httpx.Client(
            base_url='http://app', transport=httpx.MockTransport(fake)
        ) as client:
            run = Experiment(HttpAppClient(client), fixture(1),
                Config('http://app', 'https://example.openai.azure.com', 'embed',
                       readiness_timeout=.00001), Path(tmp) / 'cwq-timeout', sleep=lambda _: None)
            with self.assertRaises(TimeoutError):
                run.wait_ready([fixture(1).training[0].key])

    def test_embedding_order_and_invalid_vectors(self):
        fake = FakeApp()
        with httpx.Client(base_url='http://app', transport=httpx.MockTransport(fake)) as client:
            app = HttpAppClient(client)
            model, vectors = app.embeddings(['a', 'b'])
            self.assertEqual(model, 'text-embedding-3-small')
            self.assertEqual(vectors, [[1., 0., 0.], [1., 0., 0.]])
            with self.assertRaises(ValueError):
                app.embeddings(['too many'] * 33)
        def invalid(request):
            return httpx.Response(200, json={'azure': {'model': 'model',
                'data': [{'index': 0, 'embedding': [True]}]}})
        with httpx.Client(base_url='http://app', transport=httpx.MockTransport(invalid)) as client:
            with self.assertRaises(ApiError):
                HttpAppClient(client).embeddings(['a'])


    def test_known_model_contradiction_rejected_before_creation(self):
        fake = FakeApp()
        with tempfile.TemporaryDirectory() as tmp, httpx.Client(
            base_url='http://app', transport=httpx.MockTransport(fake)
        ) as client:
            config = Config('http://app', 'https://example.openai.azure.com', 'embed',
                            vectorizer_model='text-embedding-ada-002')
            run = Experiment(HttpAppClient(client), fixture(1), config, Path(tmp) / 'cwq-model')
            with self.assertRaisesRegex(ValueError, 'contradicts'):
                run.run()
            self.assertEqual(fake.resources, {})
            self.assertEqual([p for _, p, _ in fake.calls], ['/embeddings'])
            fake.embedding_model = 'deployment-alias'
            alias_run = Experiment(HttpAppClient(client), fixture(1), config,
                                   Path(tmp) / 'cwq-alias')
            self.assertTrue(alias_run.run())

    def test_smoke_labels_and_http_timeout_resume_validation(self):
        fake = FakeApp()
        with tempfile.TemporaryDirectory() as tmp, httpx.Client(
            base_url='http://app', transport=httpx.MockTransport(fake)
        ) as client:
            config = Config('http://app', 'https://example.openai.azure.com', 'embed',
                            train_limit=2, repeats=1, http_timeout=25)
            directory = Path(tmp) / 'cwq-smoke'
            app = HttpAppClient(client)
            run = Experiment(app, fixture(2), config, directory)
            self.assertTrue(run.run())
            summary = json.loads((directory / 'summary.json').read_text())
            self.assertEqual(summary['run_mode'], 'smoke')
            self.assertFalse(summary['representative_corpus'])
            self.assertIn('not representative', summary['corpus_warning'])
            self.assertIn('SMOKE RUN', (directory / 'report.md').read_text())
            manifest = json.loads((directory / 'manifest.json').read_text())
            self.assertEqual(manifest['config']['http_timeout'], 25)
            with self.assertRaisesRegex(ValueError, 'Resume configuration'):
                Experiment(app, fixture(2), replace(config, http_timeout=26), directory, resume=True)
