import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import vector_store


class VectorStoreIndexConfigTests(unittest.TestCase):
    def test_prefers_grpc_by_default(self):
        self.assertTrue(vector_store.QDRANT_PREFER_GRPC)

    @staticmethod
    def _collection(indexing_threshold: int, full_scan_threshold: int):
        return SimpleNamespace(
            config=SimpleNamespace(
                optimizer_config=SimpleNamespace(
                    indexing_threshold=indexing_threshold,
                ),
                hnsw_config=SimpleNamespace(
                    full_scan_threshold=full_scan_threshold,
                ),
            )
        )

    def test_updates_outdated_hnsw_thresholds(self):
        collection = self._collection(20000, 10000)
        with patch.object(vector_store, "_client") as client:
            vector_store._ensure_index_config(collection)

        client.update_collection.assert_called_once()
        call = client.update_collection.call_args.kwargs
        self.assertEqual(call["collection_name"], vector_store.QDRANT_COLLECTION)
        self.assertEqual(
            call["optimizers_config"].indexing_threshold,
            vector_store.QDRANT_INDEXING_THRESHOLD_KB,
        )
        self.assertEqual(
            call["hnsw_config"].full_scan_threshold,
            vector_store.QDRANT_FULL_SCAN_THRESHOLD_KB,
        )

    def test_keeps_matching_hnsw_thresholds_unchanged(self):
        collection = self._collection(
            vector_store.QDRANT_INDEXING_THRESHOLD_KB,
            vector_store.QDRANT_FULL_SCAN_THRESHOLD_KB,
        )
        with patch.object(vector_store, "_client") as client:
            vector_store._ensure_index_config(collection)

        client.update_collection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
