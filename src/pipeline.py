from app.config import Settings
from src.multimodal_embedder import MultiModalEmbedder
from src.storage import IndexStorage
from src.ingest import DataIngestor
from src.query_rewriter import QueryRewriter
from src.retriever import HybridRetriever
from src.reranker import ResultReranker
from src.guardrails import GuardrailChecker
from src.answer_builder import AnswerBuilder


class RagPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage = IndexStorage(settings.artifacts_dir)
        self.embedder = MultiModalEmbedder(settings.text_embed_model, settings.clip_model_name)
        self.ingestor = DataIngestor(settings.raw_data_dir, settings.processed_data_dir, self.storage, self.embedder)
        self.query_rewriter = QueryRewriter()
        self.retriever = HybridRetriever(
            storage=self.storage,
            embedder=self.embedder,
            vector_score_weight=settings.vector_score_weight,
            bm25_score_weight=settings.bm25_score_weight,
        )
        self.reranker = ResultReranker(settings.rerank_model)
        self.guardrails = GuardrailChecker(settings.min_evidence_score)
        self.answer_builder = AnswerBuilder(settings.openai_api_key)

    def ingest_all(self):
        return self.ingestor.run()

    def answer_question(self, question: str, top_k: int = 6, include_images: bool = True):
        blocked_reason = self.guardrails.inspect_question(question)
        rewritten_question = self.query_rewriter.rewrite(question)

        if blocked_reason:
            return {
                'rewritten_question': rewritten_question,
                'answer': 'I cannot help with that request.',
                'evidence': [],
                'blocked': True,
                'block_reason': blocked_reason,
            }

        retrieved_items = self.retriever.retrieve(rewritten_question, top_k=top_k, include_images=include_images)
        reranked_items = self.reranker.rerank(rewritten_question, retrieved_items, top_k=self.settings.max_context_items)
        evidence_issue = self.guardrails.inspect_evidence(reranked_items)

        if evidence_issue:
            return {
                'rewritten_question': rewritten_question,
                'answer': 'I could not answer confidently because the retrieved evidence was not strong enough.',
                'evidence': reranked_items,
                'blocked': True,
                'block_reason': evidence_issue,
            }

        answer = self.answer_builder.build_answer(rewritten_question, reranked_items)
        return {
            'rewritten_question': rewritten_question,
            'answer': answer,
            'evidence': reranked_items,
            'blocked': False,
            'block_reason': None,
        }
