import io
import json
import unittest
from contextlib import redirect_stdout

import httpx
from loguru import logger

from app.core.logging import configure_logging, log_error
from experiments.api import ApiError, HttpAppClient


class LoggingTests(unittest.TestCase):
    def test_error_source_traceback_and_redaction(self):
        output = io.StringIO()
        with redirect_stdout(output):
            configure_logging('INFO')
            with logger.contextualize(request_id='test-request'):
                try:
                    raise RuntimeError('Unauthorized api_key=secret-value Bearer token-value')
                except RuntimeError as error:
                    log_error(error, service='azure_search')
        record = json.loads(output.getvalue())['record']
        extra = record['extra']
        self.assertEqual(extra['request_id'], 'test-request')
        self.assertEqual(extra['error_module'], __name__)
        self.assertTrue(extra['error_file'].endswith('test_logging.py'))
        self.assertGreater(extra['error_line'], 0)
        self.assertIn('test_error_source', extra['error_function'])
        self.assertIn('test_logging.py', extra['traceback'])
        self.assertIn('Unauthorized', record['message'])
        self.assertNotIn('secret-value', output.getvalue())
        self.assertNotIn('token-value', output.getvalue())
        configure_logging('INFO')

    def test_api_error_keeps_upstream_detail_and_request_id(self):
        def respond(request):
            return httpx.Response(502, headers={'X-Request-ID': 'request-123'}, json={
                'detail': {'message': 'Vectorization endpoint returned 401 Unauthorized',
                           'apiKey': 'do-not-log'}})
        output = io.StringIO()
        with redirect_stdout(output):
            configure_logging('INFO')
            with httpx.Client(base_url='http://app', transport=httpx.MockTransport(respond)) as client:
                with self.assertRaises(ApiError) as caught:
                    HttpAppClient(client).request('POST', '/search')
        self.assertEqual(caught.exception.status, 502)
        self.assertIn('401 Unauthorized', str(caught.exception))
        self.assertIn('request-123', str(caught.exception))
        self.assertNotIn('do-not-log', output.getvalue() + str(caught.exception))
        self.assertIn('experiments.api', output.getvalue())
        configure_logging('INFO')
