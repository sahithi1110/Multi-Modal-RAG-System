from typing import List, Dict
from sentence_transformers import CrossEncoder


class ResultReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, items: List[Dict], top_k: int) -> List[Dict]:
        if not items:
            return []

        pair_inputs = []
        for item in items:
            pair_inputs.append([question, item['text']])

        scores = self.model.predict(pair_inputs)
        enriched_items = []
        for item, score in zip(items, scores):
            updated_item = dict(item)
            updated_item['score'] = round((item['score'] * 0.5) + (float(score) * 0.5), 4)
            enriched_items.append(updated_item)

        enriched_items.sort(key=lambda item: item['score'], reverse=True)
        return enriched_items[:top_k]
