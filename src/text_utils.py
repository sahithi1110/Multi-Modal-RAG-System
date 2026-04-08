import re
from typing import List

space_pattern = re.compile(r"\s+")


def clean_text(raw_text: str) -> str:
    text = raw_text.replace("\x00", " ")
    text = space_pattern.sub(" ", text)
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 220, overlap: int = 40) -> List[str]:
    words = text.split()
    if not words:
        return []

    parts: List[str] = []
    start_index = 0
    while start_index < len(words):
        end_index = min(start_index + chunk_size, len(words))
        chunk = " ".join(words[start_index:end_index]).strip()
        if chunk:
            parts.append(chunk)
        if end_index >= len(words):
            break
        start_index = max(0, end_index - overlap)
    return parts
