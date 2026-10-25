import unittest

from app.domain.metrics import metrics


class MetricTests(unittest.TestCase):
    def test_duplicate_chunks_do_not_inflate_recall(self):
        hits = [{"document_id": "a"}, {"document_id": "a"}]
        self.assertEqual(metrics(hits, {"a": 3, "b": 3}, 10)["recall"], 0.5)

    def test_empty_qrels_not_scored_as_failure(self):
        self.assertIsNone(metrics([], {}, 10)["recall"])
