import re


class QueryRewriter:
    def __init__(self):
        self.short_forms = {
            "rag": "retrieval augmented generation",
            "llm": "large language model",
            "api": "application programming interface",
            "ui": "user interface",
        }

    def rewrite(self, question: str) -> str:
        cleaned_question = question.strip()
        rewritten = cleaned_question.lower()

        for short_name, full_name in self.short_forms.items():
            rewritten = re.sub(rf"\b{re.escape(short_name)}\b", full_name, rewritten)

        rewritten = rewritten.replace("pic", "image")
        rewritten = rewritten.replace("photo", "image")
        rewritten = rewritten.replace("docs", "documents")

        if rewritten and not rewritten.endswith("?"):
            rewritten += "?"

        return rewritten
