from app.config import settings
from src.pipeline import RagPipeline


if __name__ == '__main__':
    pipeline = RagPipeline(settings)
    result = pipeline.ingest_all()
    print(result)
