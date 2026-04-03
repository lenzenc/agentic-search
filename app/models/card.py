from pydantic import BaseModel, Field


class CardDocument(BaseModel):
    """Document stored in Elasticsearch."""
    card_id: str
    name: str
    supertype: str = ""
    subtypes: list[str] = Field(default_factory=list)
    types: list[str] = Field(default_factory=list)
    hp: int | None = None
    stage: str | None = None
    rarity: str | None = None
    set_name: str = ""
    set_id: str = ""
    collector_number: str = ""
    set_printed_total: int | None = None
    artist: str | None = None
    flavor_text: str | None = None
    attacks_text: str = ""
    abilities_text: str = ""
    full_text: str = ""
    image_small: str | None = None
    image_large: str | None = None
    weaknesses: list[str] = Field(default_factory=list)
    resistances: list[str] = Field(default_factory=list)
    national_pokedex_numbers: list[int] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)


class CardResult(BaseModel):
    """Card data returned to the frontend — no embedding vector."""
    card_id: str
    name: str
    types: list[str] = Field(default_factory=list)
    hp: int | None = None
    stage: str | None = None
    rarity: str | None = None
    set_name: str = ""
    collector_number: str = ""
    set_printed_total: int | None = None
    image_small: str | None = None
    image_large: str | None = None
    attacks_text: str = ""
    abilities_text: str = ""

    @classmethod
    def from_es_hit(cls, hit: dict) -> "CardResult":
        src = hit.get("_source", hit)
        return cls(
            card_id=src.get("card_id", ""),
            name=src.get("name", ""),
            types=src.get("types", []),
            hp=src.get("hp"),
            stage=src.get("stage"),
            rarity=src.get("rarity"),
            set_name=src.get("set_name", ""),
            collector_number=src.get("collector_number", ""),
            set_printed_total=src.get("set_printed_total"),
            image_small=src.get("image_small"),
            image_large=src.get("image_large"),
            attacks_text=src.get("attacks_text", ""),
            abilities_text=src.get("abilities_text", ""),
        )
