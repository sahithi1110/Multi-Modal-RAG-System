import json

from app.config import settings
from src.pipeline import RagPipeline


if __name__ == "__main__":
    pipeline = RagPipeline(settings)
    question_file = settings.project_root / "data" / "sample_questions.json"
    if not question_file.exists():
        raise FileNotFoundError("sample_questions.json was not found")

    questions = json.loads(question_file.read_text(encoding="utf-8"))
    for row in questions:
        result = pipeline.answer_question(row["question"])
        print("\n" + "=" * 80)
        print("Question:", row["question"])
        print("Expected topic:", row.get("expected_topic", "n/a"))
        print("Rewritten:", result["rewritten_question"])
        print("Answer:", result["answer"])
        print("Blocked:", result["blocked"])
        print("Top evidence count:", len(result["evidence"]))
