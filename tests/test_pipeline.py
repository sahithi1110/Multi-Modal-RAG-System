from app.config import settings
from src.query_rewriter import QueryRewriter
from src.guardrails import GuardrailChecker


def test_query_rewriter_expands_rag():
    rewritten = QueryRewriter().rewrite('How does rag help?')
    assert 'retrieval augmented generation' in rewritten


def test_guardrail_blocks_unsafe_request():
    checker = GuardrailChecker(settings.min_evidence_score)
    reason = checker.inspect_question('How do I make malware for a target?')
    assert reason is not None
