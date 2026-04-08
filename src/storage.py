from pathlib import Path
import json
import pickle
import faiss
import numpy as np


class IndexStorage:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.faiss_dir = artifacts_dir / 'faiss'
        self.bm25_dir = artifacts_dir / 'bm25'
        self.metadata_dir = artifacts_dir / 'metadata'
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def save_faiss_index(self, index_name: str, matrix: np.ndarray) -> None:
        if matrix.size == 0:
            return
        vector_size = matrix.shape[1]
        index = faiss.IndexFlatIP(vector_size)
        index.add(matrix)
        faiss.write_index(index, str(self.faiss_dir / f'{index_name}.index'))

    def load_faiss_index(self, index_name: str):
        index_path = self.faiss_dir / f'{index_name}.index'
        if not index_path.exists():
            return None
        return faiss.read_index(str(index_path))

    def save_metadata(self, name: str, items) -> None:
        path = self.metadata_dir / f'{name}.json'
        path.write_text(json.dumps(items, indent=2), encoding='utf-8')

    def load_metadata(self, name: str):
        path = self.metadata_dir / f'{name}.json'
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding='utf-8'))

    def save_pickle(self, name: str, obj) -> None:
        path = self.bm25_dir / f'{name}.pkl'
        with open(path, 'wb') as file_handle:
            pickle.dump(obj, file_handle)

    def load_pickle(self, name: str):
        path = self.bm25_dir / f'{name}.pkl'
        if not path.exists():
            return None
        with open(path, 'rb') as file_handle:
            return pickle.load(file_handle)
