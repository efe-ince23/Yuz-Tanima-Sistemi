from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


class InvalidByteRange(ValueError):
    pass


def parse_byte_range(header: Optional[str], size: int) -> Optional[ByteRange]:
    if not header:
        return None
    if size <= 0 or not header.startswith("bytes="):
        raise InvalidByteRange

    value = header[6:].strip()
    if not value or "," in value or "-" not in value:
        raise InvalidByteRange
    start_text, end_text = value.split("-", 1)
    try:
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                raise InvalidByteRange
            start = max(0, size - suffix_length)
            return ByteRange(start=start, end=size - 1)

        start = int(start_text)
        if start < 0 or start >= size:
            raise InvalidByteRange
        end = size - 1 if not end_text else int(end_text)
        if end < start:
            raise InvalidByteRange
        return ByteRange(start=start, end=min(end, size - 1))
    except (TypeError, ValueError) as error:
        raise InvalidByteRange from error
