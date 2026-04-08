from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    text_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    vector_score_weight: float = 0.65
    bm25_score_weight: float = 0.35
    min_evidence_score: float = 0.22
    max_context_items: int = 6
    project_root: Path = Path(__file__).resolve().parents[1]

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / 'data' / 'raw'

    @property
    def processed_data_dir(self) -> Path:
        return self.project_root / 'data' / 'processed'

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / 'artifacts'


settings = Settings()
