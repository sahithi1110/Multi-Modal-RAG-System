from typing import Dict, List

from openai import OpenAI


class AnswerBuilder:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key) if api_key else None

    def build_answer(self, question: str, evidence_items: List[Dict]) -> str:
        if not evidence_items:
            return "I could not find enough grounded evidence to answer this question safely."

        if not self.client:
            return self._build_fallback_answer(question, evidence_items)

        evidence_lines: List[str] = []
        for item in evidence_items:
            page_number = item.get("page_number")
            location = f"page {page_number}" if page_number else "no page number"
            evidence_lines.append(
                f"Source: {item['source_name']} | Type: {item['item_type']} | Location: {location}\n"
                f"Evidence: {item['text']}"
            )

        prompt = (
            "You are answering only from grounded retrieval evidence. "
            "If the evidence is incomplete, say that clearly. Do not invent details.\n\n"
            f"Question:\n{question}\n\n"
            f"Evidence:\n{chr(10).join(evidence_lines)}\n\n"
            "Write a concise answer in plain English."
        )

        response = self.client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.1,
        )
        return response.output_text.strip()

    def _build_fallback_answer(self, question: str, evidence_items: List[Dict]) -> str:
        top_items = evidence_items[:3]
        answer_parts = [
            f"Based on the retrieved evidence, here is the best grounded answer for: {question}"
        ]
        for item in top_items:
            answer_parts.append(f"{item['source_name']}: {item['text']}")
        answer_parts.append(
            "This answer was built without a live language model key, so it stays extractive and conservative."
        )
        return " ".join(answer_parts)
