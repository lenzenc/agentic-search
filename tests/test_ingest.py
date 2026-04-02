"""Unit tests for ingest utilities."""
import pytest
from ingest.build_embeddings import build_text_blob, card_to_document


SAMPLE_CARD = {
    "id": "xy1-11",
    "name": "Charizard",
    "supertype": "Pokémon",
    "subtypes": ["Stage 2"],
    "types": ["Fire"],
    "hp": "250",
    "rarity": "Rare Holo",
    "set": {"name": "XY", "id": "xy1"},
    "artist": "5ban Graphics",
    "flavorText": "It spits fire that is hot enough to melt boulders.",
    "attacks": [
        {
            "name": "Fire Spin",
            "cost": ["Fire", "Fire", "Fire", "Colorless"],
            "damage": "200",
            "text": "Discard 2 Energy attached to this Pokemon.",
        }
    ],
    "abilities": [
        {
            "name": "Blaze",
            "type": "Ability",
            "text": "Once during your turn, you may search your deck for a Fire Energy card and attach it.",
        }
    ],
    "weaknesses": [{"type": "Water", "value": "×2"}],
    "resistances": [],
    "nationalPokedexNumbers": [6],
}


def test_build_text_blob_contains_card_name():
    blob = build_text_blob(SAMPLE_CARD)
    assert "Charizard" in blob


def test_build_text_blob_contains_type():
    blob = build_text_blob(SAMPLE_CARD)
    assert "Fire" in blob


def test_build_text_blob_contains_attack_name():
    blob = build_text_blob(SAMPLE_CARD)
    assert "Fire Spin" in blob


def test_build_text_blob_contains_attack_text():
    blob = build_text_blob(SAMPLE_CARD)
    assert "Discard 2 Energy" in blob


def test_build_text_blob_contains_ability():
    blob = build_text_blob(SAMPLE_CARD)
    assert "Blaze" in blob
    assert "Fire Energy" in blob


def test_build_text_blob_contains_weakness():
    blob = build_text_blob(SAMPLE_CARD)
    assert "Water" in blob


def test_card_to_document_hp_parsed_as_int():
    doc = card_to_document(SAMPLE_CARD)
    assert doc["hp"] == 250
    assert isinstance(doc["hp"], int)


def test_card_to_document_stage_detected():
    doc = card_to_document(SAMPLE_CARD)
    assert doc["stage"] == "Stage 2"


def test_card_to_document_attacks_text_built():
    doc = card_to_document(SAMPLE_CARD)
    assert "Fire Spin" in doc["attacks_text"]
    assert "200" in doc["attacks_text"]


def test_card_to_document_abilities_text_built():
    doc = card_to_document(SAMPLE_CARD)
    assert "Blaze" in doc["abilities_text"]


def test_card_to_document_full_text_non_empty():
    doc = card_to_document(SAMPLE_CARD)
    assert len(doc["full_text"]) > 50


def test_card_to_document_missing_hp_is_none():
    card = {**SAMPLE_CARD, "hp": None}
    doc = card_to_document(card)
    assert doc["hp"] is None


def test_card_to_document_no_attacks():
    card = {**SAMPLE_CARD, "attacks": []}
    doc = card_to_document(card)
    assert doc["attacks_text"] == ""


def test_card_to_document_set_name():
    doc = card_to_document(SAMPLE_CARD)
    assert doc["set_name"] == "XY"
    assert doc["set_id"] == "xy1"
