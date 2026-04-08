from typing import List, Dict, Optional


class GuardrailChecker:
    def __init__(self, min_evidence_score: float):
        self.min_evidence_score = min_evidence_score
        self.blocked_terms = {
            'build a bomb',
            'make malware',
            'credit card fraud',
            'steal password',
        }

    def inspect_question(self, question: str) -> Optional[str]:
        lowered_question = question.lower()
        for blocked_phrase in self.blocked_terms:
            if blocked_phrase in lowered_question:
                return 'The question matches a blocked safety pattern.'
        return None

    def inspect_evidence(self, items: List[Dict]) -> Optional[str]:
        if not items:
            return 'No evidence was retrieved.'
        best_score = items[0]['score']
        if best_score < self.min_evidence_score:
            return 'The retrieved evidence is too weak to answer confidently.'
        return None
