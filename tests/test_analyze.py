from __future__ import annotations


import pytest

from bd_pipeline import analyze


def test_analyze_book_parses_response(fake_client):
    fake_client.analyze_response = {
        "summary": "Tintin part en voyage.",
        "tags": ["Aventure", "Jeunesse"],
        "characters": ["Tintin", "Milou"],
        "locations": ["Bruxelles"],
        "notable_people": [],
    }
    data = analyze.analyze_book(
        "Tintin", ["bonjour"], client=fake_client, model="test-llm"
    )
    assert data["summary"] == "Tintin part en voyage."
    # tags normalised to lowercase
    assert data["tags"] == ["aventure", "jeunesse"]
    assert data["characters"] == ["Tintin", "Milou"]
    assert data["locations"] == ["Bruxelles"]
    assert data["notable_people"] == []


def test_analyze_book_uses_json_format(fake_client):
    analyze.analyze_book("X", ["hello"], client=fake_client, model="m")
    call = fake_client.calls[-1]
    assert call["format"] == "json"
    assert call["options"]["temperature"] == 0


def test_analyze_book_malformed_json_raises(fake_client):
    class BadClient:
        calls: list = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return {"message": {"content": "not json at all"}}

    with pytest.raises(analyze.AnalysisError):
        analyze.analyze_book("X", ["hello"], client=BadClient(), model="m")


def test_analyze_book_empty_pages_short_circuits(fake_client):
    data = analyze.analyze_book("X", ["", "  "], client=fake_client, model="m")
    assert data == {
        "summary": "",
        "tags": [],
        "characters": [],
        "locations": [],
        "notable_people": [],
    }
    assert fake_client.calls == []


def test_analyze_book_map_reduce_merges_names(fake_client):
    # Pages of 200 chars each with a chunk budget of 500 -> 2 map chunks
    # (first chunk holds 2 pages, second chunk holds 1). Plus one reduce = 3 calls.
    fake_client.analyze_response_queue = [
        {
            "summary": "Partie 1.",
            "tags": ["aventure"],
            "characters": ["Tintin", "Milou"],
            "locations": ["Bruxelles"],
            "notable_people": [],
        },
        {
            "summary": "Partie 2.",
            "tags": ["mystère"],
            "characters": ["Haddock"],
            "locations": ["Moulinsart"],
            "notable_people": [],
        },
        {
            "summary": "Résumé global.",
            "tags": ["aventure", "mystère"],
            "characters": ["Tintin", "Milou"],  # reducer drops Haddock
            "locations": ["Bruxelles", "Moulinsart"],
            "notable_people": [],
        },
    ]
    data = analyze.analyze_book(
        "Long", ["a" * 200, "b" * 200, "c" * 200],
        client=fake_client, model="m", map_chunk_chars=500,
    )
    assert len(fake_client.calls) == 3  # 2 map + 1 reduce
    assert "Tintin" in data["characters"]
    assert "Milou" in data["characters"]
    assert "Haddock" in data["characters"]  # defensive merge restores it
    assert set(data["tags"]) == {"aventure", "mystère"}
    assert "Moulinsart" in data["locations"]


def test_analyze_book_coerces_wrong_types(fake_client):
    fake_client.analyze_response = {
        "summary": "ok",
        "tags": "pas une liste",  # invalid type
        "characters": ["x", None, "  "],  # blanks/None must be filtered
        "locations": [],
        "notable_people": [],
    }
    data = analyze.analyze_book("X", ["p"], client=fake_client, model="m")
    assert data["tags"] == []  # invalid type coerced to []
    assert data["characters"] == ["x"]
