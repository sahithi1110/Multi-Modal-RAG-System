from typing import List, Dict
import numpy as np


class HybridRetriever:
    def __init__(self, storage, embedder, vector_score_weight: float, bm25_score_weight: float):
        self.storage = storage
        self.embedder = embedder
        self.vector_score_weight = vector_score_weight
        self.bm25_score_weight = bm25_score_weight

    def retrieve(self, question: str, top_k: int = 6, include_images: bool = True) -> List[Dict]:
        text_results = self._retrieve_text(question, top_k)
        image_results = self._retrieve_images(question, top_k) if include_images else []
        all_results = text_results + image_results
        all_results.sort(key=lambda item: item['score'], reverse=True)
        return all_results[:top_k * 2]

    def _retrieve_text(self, question: str, top_k: int) -> List[Dict]:
        text_items = self.storage.load_metadata('text_items')
        text_index = self.storage.load_faiss_index('text_vectors')
        bm25_model = self.storage.load_pickle('text_bm25')
        if not text_items or text_index is None:
            return []

        query_vector = self.embedder.encode_text([question])
        vector_scores, vector_positions = text_index.search(query_vector, min(top_k * 3, len(text_items)))
        vector_score_map = {}
        for score, position in zip(vector_scores[0], vector_positions[0]):
            if position != -1:
                vector_score_map[int(position)] = float(score)

        bm25_score_map = {}
        if bm25_model is not None:
            raw_bm25_scores = bm25_model.get_scores(question.lower().split())
            if len(raw_bm25_scores) > 0:
                max_bm25 = max(raw_bm25_scores)
                if max_bm25 > 0:
                    for item_position, score in enumerate(raw_bm25_scores):
                        bm25_score_map[item_position] = float(score / max_bm25)

        candidate_positions = set(vector_score_map.keys()) | set(bm25_score_map.keys())
        merged_results = []
        for position in candidate_positions:
            item = text_items[position]
            vector_part = vector_score_map.get(position, 0.0)
            bm25_part = bm25_score_map.get(position, 0.0)
            final_score = (self.vector_score_weight * vector_part) + (self.bm25_score_weight * bm25_part)
            merged_results.append({**item, 'score': round(final_score, 4)})

        merged_results.sort(key=lambda item: item['score'], reverse=True)
        return merged_results[: top_k * 2]

    def _retrieve_images(self, question: str, top_k: int) -> List[Dict]:
        image_items = self.storage.load_metadata('image_items')
        image_index = self.storage.load_faiss_index('image_vectors')
        if not image_items or image_index is None:
            return []

        query_vector = self.embedder.encode_query_for_images(question)
        image_scores, image_positions = image_index.search(query_vector, min(top_k, len(image_items)))
        results = []
        for score, position in zip(image_scores[0], image_positions[0]):
            if position == -1:
                continue
            item = image_items[int(position)]
            results.append({**item, 'score': round(float(score), 4)})
        return results
