from functools import lru_cache
from app.config import settings
from src.pipeline import RagPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    return RagPipeline(settings)
