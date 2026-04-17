from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class BookAnalysis(BaseModel):
    title: str
    path: str
    page_count: int
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    notable_people: list[str] = Field(default_factory=list)
    vlm_model: str = ""
    llm_model: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def sidecar_path(self) -> Path:
        return Path(self.path).with_suffix(".json")
