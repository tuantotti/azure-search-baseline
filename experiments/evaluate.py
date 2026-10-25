"""Proxy Q&A retrieval metrics and transparent operational comparisons."""

import csv
import json
import math
import statistics
from itertools import combinations
from pathlib import Path

from .dataset import Dataset, Record


def score(query: Record, keys: list[str], relevant: set[str], corpus: dict[str, Record]) -> dict:
    keys = list(dict.fromkeys(keys))
    metrics = {}
    for k in (1, 5, 10):
        selected = keys[:k]
        found = set().union(*(corpus[key].answers for key in selected)) & query.answers
        hits = [int(key in relevant) for key in selected]
        metrics[f'answer_hit@{k}'] = float(bool(found))
        metrics[f'answer_coverage@{k}'] = len(found) / len(query.answers) if query.answers else None
        metrics[f'all_answers_hit@{k}'] = float(found == query.answers) if query.answers else None
        metrics[f'document_recall@{k}'] = sum(hits) / len(relevant) if relevant else None
        metrics[f'mrr@{k}'] = next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0) \
            if relevant else None
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        metrics[f'ndcg@{k}'] = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits)) \
            / ideal if ideal else None
    return metrics


def distribution(values: list[float]) -> dict:
    if not values:
        return {'mean': None, 'median': None, 'p90': None, 'p95': None}
    ordered = sorted(values)
    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        low = int(position)
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) \
            * (position - low)
    return {'mean': statistics.mean(values), 'median': statistics.median(values),
            'p90': percentile(.9), 'p95': percentile(.95)}


def aggregate(rows: list[dict]) -> dict:
    successes = [row for row in rows if row['success']]
    names = sorted({name for row in successes for name in row['metrics']})
    quality = {name: statistics.mean(values) if (values := [r['metrics'][name]
               for r in successes if r['metrics'][name] is not None]) else None for name in names}
    return {'attempts': len(rows), 'successes': len(successes),
            'failure_rate': 1 - len(successes) / len(rows) if rows else None,
            'latency_ms': distribution([r['latency_ms'] for r in rows]),
            'successful_latency_ms': distribution([r['latency_ms'] for r in successes]),
            'returned_documents': distribution([len(r['keys']) for r in successes]),
            'server_ms': distribution([r['server_ms'] for r in successes
                                       if r.get('server_ms') is not None]),
            'embedding_ms': distribution([r['embedding_ms'] for r in successes
                                          if r.get('embedding_ms') is not None]),
            'quality_success_only': quality,
            'failure_adjusted_answer_hit@10': sum(r['metrics']['answer_hit@10']
                for r in successes) / len(rows) if rows else None,
            'agentic_input_tokens': sum(a.get('inputTokens', 0) or 0 for r in successes
                                        for a in r.get('activity', [])),
            'agentic_output_tokens': sum(a.get('outputTokens', 0) or 0 for r in successes
                                         for a in r.get('activity', []))}


def write_reports(directory: Path, dataset: Dataset, queries: list[Record],
                  judgments: dict[str, set[str]], rows: list[dict], expected: int,
                  setup_seconds: float, *, failure: dict | None = None, smoke: bool = False) -> dict:
    methods = ('hybrid', 'hybrid_semantic', 'agentic')
    corpus = {r.key: r for r in dataset.training}
    coverage = []
    for query in queries:
        available = set().union(*(corpus[k].answers for k in judgments[query.id])) & query.answers
        coverage.append({'query_id': query.id, 'relevant_documents': len(judgments[query.id]),
                         'answer_coverage': len(available) / len(query.answers)
                         if query.answers else None})
    summary = {'run_mode': 'smoke' if smoke else 'benchmark',
               'representative_corpus': not smoke,
               'corpus_warning': ('Restricted training corpus: smoke verification only; quality '
                                  'is not representative of the full benchmark') if smoke else None,
               'evaluation': 'Q&A retrieval with shared answer IDs as proxy relevance; '
               'not evidence relevance or generated-answer accuracy',
               'expected_attempts': expected, 'completed_attempts': len(rows),
               'complete': len(rows) == expected and all(r['success'] for r in rows),
               'setup_seconds': setup_seconds, 'failure': failure, 'training_records': len(dataset.training),
               'missing_readable_labels': sum(r.missing_labels for r in dataset.training),
               'corpus_coverage': coverage,
               'methods': {m: aggregate([r for r in rows if r['method'] == m]) for m in methods},
               'by_compositionality': {}, 'paired_success': {}}
    for kind in sorted({q.kind for q in queries}):
        summary['by_compositionality'][kind] = {
            m: aggregate([r for r in rows if r['kind'] == kind and r['method'] == m])
            for m in methods}
    for a, b in combinations(methods, 2):
        left = {(r['query_id'], r['repeat']): r for r in rows
                if r['method'] == a and r['success']}
        pairs = [(left[(r['query_id'], r['repeat'])], r) for r in rows
                 if r['method'] == b and r['success'] and (r['query_id'], r['repeat']) in left]
        summary['paired_success'][f'{b} minus {a}'] = {
            'pairs': len(pairs), 'quality_difference': {
                metric: statistics.mean(values) if (values := [right['metrics'][metric]
                    - lft['metrics'][metric] for lft, right in pairs
                    if lft['metrics'][metric] is not None and right['metrics'][metric] is not None])
                else None for metric in (pairs[0][0]['metrics'] if pairs else [])}}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    metric_names = sorted({m for row in rows for m in row.get('metrics', {})})
    with (directory / 'per-query.csv').open('w', newline='') as handle:
        fields = ['query_id', 'kind', 'method', 'repeat', 'success', 'latency_ms', 'error',
                  'returned_count', *metric_names]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row.get(k) for k in fields[:7]},
                             'returned_count': len(row.get('keys', [])), **row.get('metrics', {})})
    lines = ['# ComplexWebQuestions Q&A retrieval benchmark', '', summary['evaluation'], '',
             f"Training records: {len(dataset.training)}; measured attempts: {len(rows)}/{expected}.",
             f"Queries with a relevant eligible training record: "
             f"{sum(bool(judgments[q.id]) for q in queries)}/{len(queries)}.", '',
             '| Method | Successes | Failure rate | Mean latency ms | Answer Hit@10 (success only) |',
             '|---|---:|---:|---:|---:|']
    for method, result in summary['methods'].items():
        lines.append(f"| {method} | {result['successes']} | {result['failure_rate']} | "
                     f"{result['latency_ms']['mean']} | "
                     f"{result['quality_success_only'].get('answer_hit@10')} |")
    lines += ['', 'Uncovered queries remain in the sample. Document ranking metrics are undefined '
              'when no eligible relevant records exist. Warm-ups are excluded. Repetitions are '
              'repeated observations, not independent questions. JSON includes paired-success '
              'quality differences, operational failures, type breakdowns and corpus coverage.', '',
              'Equal output limits do not equalize agentic internal retrieval work. '
              'HTTP latency includes query embedding; server timing is reported separately.']
    if smoke:
        lines[2:2] = ['**SMOKE RUN — restricted training corpus. Quality results are not '
                      'representative of the full benchmark.**', '']
    if failure:
        lines += ['', f"Failure during {failure['phase']}: {failure['type']}: "
                  f"{failure['message']}"]
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    return summary
