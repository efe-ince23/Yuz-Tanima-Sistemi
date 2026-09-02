import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from qdrant_client import models

from app import vector_store


def point(point_id: int) -> models.PointStruct:
    return models.PointStruct(id=point_id, vector=[float(point_id)])


class VectorStoreSyncTests(unittest.TestCase):
    def test_upsert_sends_points_in_bounded_batches(self) -> None:
        consumed = []

        def generated_points():
            for point_id in range(1, 6):
                consumed.append(point_id)
                yield point(point_id)

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(vector_store, "QDRANT_UPSERT_BATCH_SIZE", 2)
            )
            client = stack.enter_context(patch.object(vector_store, "_client"))
            indexed = vector_store._upsert(generated_points())

        self.assertEqual(indexed, 5)
        self.assertEqual(consumed, [1, 2, 3, 4, 5])
        self.assertEqual(client.upsert.call_count, 3)
        self.assertEqual(
            [len(call.kwargs["points"]) for call in client.upsert.call_args_list],
            [2, 2, 1],
        )

    def test_synchronize_all_tracks_ids_and_removes_only_stale_points(self) -> None:
        session = MagicMock()
        database_points = (point(point_id) for point_id in (1, 2, 3))

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(vector_store, "QDRANT_UPSERT_BATCH_SIZE", 2)
            )
            stack.enter_context(patch.object(vector_store, "ensure_collection"))
            stack.enter_context(
                patch.object(
                    vector_store,
                    "_database_points",
                    return_value=database_points,
                )
            )
            stack.enter_context(
                patch.object(
                    vector_store,
                    "_stored_point_ids",
                    return_value={"1", "2", "3", "4"},
                )
            )
            client = stack.enter_context(patch.object(vector_store, "_client"))
            result = vector_store.synchronize_all(session)

        self.assertEqual(result.indexed_points, 3)
        self.assertEqual(result.removed_points, 1)
        client.delete.assert_called_once()
        selector = client.delete.call_args.kwargs["points_selector"]
        self.assertEqual(selector.points, ["4"])


if __name__ == "__main__":
    unittest.main()
