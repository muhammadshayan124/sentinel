import pytest

from sentinel.ingestion.chunking import chunk_text


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("", source="a.txt") == []


def test_chunk_text_short_text_single_chunk():
    chunks = chunk_text("hello world foo bar", source="a.txt", chunk_size=10, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world foo bar"
    assert chunks[0].source == "a.txt"
    assert chunks[0].index == 0


def test_chunk_text_produces_overlap():
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)
    chunks = chunk_text(text, source="a.txt", chunk_size=10, overlap=4)

    assert len(chunks) > 1
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    overlap_words = set(first_words) & set(second_words)
    assert len(overlap_words) == 4


def test_chunk_text_rejects_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("a b c", source="a.txt", chunk_size=5, overlap=5)


def test_chunk_text_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("a b c", source="a.txt", chunk_size=0, overlap=0)
