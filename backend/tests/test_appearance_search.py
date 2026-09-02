import unittest
from datetime import datetime, timezone
from uuid import uuid4

from app.api_errors import ApiHTTPException
from app.models import User
from app.routers.search import _matches_filters, _resolve_owner_scope, _validate_ranges
from app.schemas import AppearanceSearchItemResponse


class AppearanceSearchFilterTests(unittest.TestCase):
    def item(self, **changes):
        values = {
            "source_type": "video",
            "process_id": uuid4(),
            "face_id": uuid4(),
            "status": "known",
            "person_id": 7,
            "first_name": "Okan",
            "last_name": "Buruk",
            "metadata": {"description": "Teknik direktor"},
            "owner_user_id": uuid4(),
            "owner_username": "efe",
            "owner_full_name": "Tayyar Efe Ince",
            "occurred_at": datetime.now(timezone.utc),
            "confidence": 0.78,
            "original_filename": "mac-roportaji.mp4",
            "preview_url": "/media/frame.jpg",
            "content_url": "/api/videos/example/content",
            "observation_count": 5,
            "first_seen_ms": 1000,
            "last_seen_ms": 5000,
            "intervals": [],
        }
        values.update(changes)
        return AppearanceSearchItemResponse(**values)

    def test_query_matches_only_identity_name_or_face_id(self):
        item = self.item()
        for query in ("okan buruk", "okan", str(item.face_id)):
            self.assertTrue(_matches_filters(item, query, None, None, None, None))
        for query in ("roportaji", "tayyar efe"):
            self.assertFalse(_matches_filters(item, query, None, None, None, None))

    def test_video_filename_does_not_match_unrelated_face(self):
        item = self.item(
            first_name="Beyazit",
            last_name="Ozturk",
            original_filename="Beyaz Kivanc Tatlitug'u Programa Nasil Getirdi.mp4",
        )
        self.assertFalse(_matches_filters(item, "Kivanc", None, None, None, None))

    def test_identity_and_confidence_filters_are_combined(self):
        item = self.item()
        self.assertTrue(_matches_filters(item, None, None, "known", 0.7, 0.8))
        self.assertFalse(_matches_filters(item, None, None, "anonymous", None, None))
        self.assertFalse(_matches_filters(item, None, None, None, 0.8, None))

    def test_anonymous_filter_includes_new_anonymous(self):
        item = self.item(status="new_anonymous", person_id=None, first_name=None, last_name=None)
        self.assertTrue(_matches_filters(item, None, None, "anonymous", None, None))

    def test_invalid_ranges_are_rejected(self):
        later = datetime(2026, 8, 31, tzinfo=timezone.utc)
        earlier = datetime(2026, 8, 30, tzinfo=timezone.utc)
        with self.assertRaises(ApiHTTPException):
            _validate_ranges(later, earlier, None, None)
        with self.assertRaises(ApiHTTPException):
            _validate_ranges(None, None, 0.9, 0.5)

    def test_search_scope_is_always_the_current_user(self):
        for role in ("admin", "user"):
            user = User(
                id=uuid4(),
                username=f"{role}-account",
                email=f"{role}@example.local",
                full_name=role.title(),
                password_hash="test",
                role=role,
            )
            self.assertEqual(_resolve_owner_scope(user, None), user.id)
            self.assertEqual(_resolve_owner_scope(user, user.id), user.id)
            with self.assertRaises(ApiHTTPException):
                _resolve_owner_scope(user, uuid4())


if __name__ == "__main__":
    unittest.main()
