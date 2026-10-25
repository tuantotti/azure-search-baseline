"""Run explicitly: python -m experiments.main --run-live [options]."""

import argparse
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from dotenv import dotenv_values

from .api import ApiError, HttpAppClient
from .dataset import Dataset
from .experiment import Config, Experiment


def main(argv: list[str] | None = None) -> int:
    defaults = {**dotenv_values('.env'), **os.environ}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-live', action='store_true', help='Explicitly allow API resource creation')
    parser.add_argument('--base-url', default='http://localhost:8000')
    parser.add_argument('--dataset-dir', type=Path, default=Path('data/ComplexWebQuestions'))
    parser.add_argument('--artifact-root', type=Path, default=Path('artifacts/cwq'))
    parser.add_argument('--run-id')
    parser.add_argument('--resume', action='store_true', help='Resume --run-id with identical settings')
    parser.add_argument('--question-limit', type=int, default=100, help='0 selects all dev questions')
    parser.add_argument('--train-limit', type=int, default=0, help='Smoke runs only; 0 indexes all training')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--readiness-timeout', type=float, default=300)
    parser.add_argument('--http-timeout', type=float, default=180)
    parser.add_argument('--vectorizer-endpoint', default=defaults.get('AZURE_OPENAI_ENDPOINT'))
    parser.add_argument('--vectorizer-deployment',
                        default=defaults.get('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'))
    parser.add_argument('--vectorizer-model', default=defaults.get('CWQ_VECTORIZER_MODEL'))
    args = parser.parse_args(argv)
    if not args.run_live:
        parser.error('--run-live is required; importing or displaying help performs no requests')
    if args.resume and not args.run_id:
        parser.error('--resume requires --run-id')
    if args.question_limit < 0 or args.train_limit < 0 or args.repeats < 1 or not 1 <= args.batch_size <= 32:
        parser.error('limits must be nonnegative; repeats positive; batch size 1..32')
    if args.readiness_timeout <= 0 or args.http_timeout <= 0:
        parser.error('timeouts must be positive')
    for name, value in [('base URL', args.base_url), ('vectorizer endpoint', args.vectorizer_endpoint)]:
        parsed = urlsplit(value or '')
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username \
                or parsed.password or parsed.query or parsed.fragment:
            parser.error(f'{name} must be an HTTP URL without credentials/query/fragment')
    if not args.vectorizer_deployment:
        parser.error('--vectorizer-deployment or AZURE_OPENAI_EMBEDDING_DEPLOYMENT is required')
    run_id = args.run_id or f"cwq-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,80}[a-z0-9]', run_id):
        parser.error('--run-id must contain 3..82 lowercase letters, digits or internal hyphens')
    config = Config(args.base_url.rstrip('/'), args.vectorizer_endpoint.rstrip('/'),
                    args.vectorizer_deployment, args.vectorizer_model, args.question_limit,
                    args.repeats, args.seed, args.train_limit, args.batch_size, args.readiness_timeout,
                    args.http_timeout)
    directory = args.artifact_root / run_id
    try:
        dataset = Dataset.load(args.dataset_dir, args.train_limit)
        with httpx.Client(base_url=config.base_url, timeout=args.http_timeout) as client:
            app = HttpAppClient(client)
            schema = app.request('GET', '/openapi.json', retry=True)
            properties = schema.get('components', {}).get('schemas', {}).get(
                'AgenticRetrievalRequest', {}).get('properties', {})
            if 'max_output_documents' not in properties:
                raise ValueError('Running app lacks max_output_documents; restart with updated code')
            experiment = Experiment(app, dataset, config, directory, resume=args.resume)
            complete = experiment.run()
        print(f"{'Complete' if complete else 'Incomplete'}: {directory / 'report.md'}")
        return 0 if complete else 1
    except (ApiError, OSError, ValueError, RuntimeError, KeyError, TimeoutError) as exc:
        print(f'Benchmark stopped: {exc}. Artifacts: {directory}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
