from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = 6
    include_images: bool = True


class EvidenceItem(BaseModel):
    item_id: str
    item_type: Literal['text', 'image']
    source_name: str
    page_number: Optional[int] = None
    text: str
    score: float
    path: Optional[str] = None


class AskResponse(BaseModel):
    rewritten_question: str
    answer: str
    evidence: List[EvidenceItem]
    blocked: bool = False
    block_reason: Optional[str] = None


class IngestResponse(BaseModel):
    success: bool
    message: str
    text_items: int
    image_items: int
