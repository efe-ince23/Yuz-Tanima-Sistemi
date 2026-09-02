import unittest

from app.video_streaming import InvalidByteRange, parse_byte_range


class VideoStreamingTests(unittest.TestCase):
    def test_missing_range_requests_full_content(self):
        self.assertIsNone(parse_byte_range(None, 100))

    def test_supported_byte_ranges(self):
        cases = [
            ("bytes=0-9", (0, 9)),
            ("bytes=10-", (10, 99)),
            ("bytes=-10", (90, 99)),
            ("bytes=90-150", (90, 99)),
        ]
        for header, expected in cases:
            with self.subTest(header=header):
                parsed = parse_byte_range(header, 100)
                self.assertIsNotNone(parsed)
                self.assertEqual((parsed.start, parsed.end), expected)
                self.assertEqual(parsed.length, expected[1] - expected[0] + 1)

    def test_invalid_or_unsatisfiable_ranges_are_rejected(self):
        headers = [
            "items=0-9",
            "bytes=",
            "bytes=10-9",
            "bytes=100-",
            "bytes=-0",
            "bytes=0-1,4-5",
        ]
        for header in headers:
            with self.subTest(header=header):
                with self.assertRaises(InvalidByteRange):
                    parse_byte_range(header, 100)


if __name__ == "__main__":
    unittest.main()
