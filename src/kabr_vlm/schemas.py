from typing import Literal

from pydantic import BaseModel, Field


SpeciesLabel = Literal["giraffe", "plains zebra", "grevys zebra"]


class KABRAnnotation(BaseModel):
    video_id: str
    species: SpeciesLabel
    behavior: str
    clip_path: str
    split: str = "train"
    question_id: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    notes: str | None = None


class VideoQAExample(BaseModel):
    id: str
    video_id: str
    clip_path: str | None = None
    image_paths: list[str] = Field(default_factory=list)
    split: str = "train"
    task: Literal["animal_recognition", "action_recognition", "joint_animal_action_recognition"]
    prompt: str
    answer: str
    metadata: dict[str, str | float | int | None] = Field(default_factory=dict)
