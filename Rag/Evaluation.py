import json
import re
from collections import Counter
from difflib import SequenceMatcher
from statistics import mean
from typing import Any, Iterable, Sequence


class RAGEvaluator:
    """Evaluate your actual RAG pipeline from Rag/Retrival.py."""

    def __init__(self, rag_pipeline: Any, top_k: int = 5) -> None:
        self.rag = rag_pipeline
        self.top_k = top_k

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return RAGEvaluator._normalize(text).split()

    @staticmethod
    def token_f1(prediction: str, ground_truth: str) -> float:
        pred_tokens = RAGEvaluator._tokenize(prediction)
        truth_tokens = RAGEvaluator._tokenize(ground_truth)
        if not pred_tokens and not truth_tokens:
            return 1.0
        if not pred_tokens or not truth_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def similarity_score(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0

        token_f1 = RAGEvaluator.token_f1(a, b)
        sequence = SequenceMatcher(None, RAGEvaluator._normalize(a), RAGEvaluator._normalize(b)).ratio()
        return 0.7 * token_f1 + 0.3 * sequence

    @staticmethod
    def _doc_text(doc: Any) -> str:
        if hasattr(doc, "page_content"):
            return str(doc.page_content)
        if isinstance(doc, dict):
            return str(doc.get("page_content") or doc.get("content") or doc)
        return str(doc)

    def _retrieved_docs(self, question: str) -> list[str]:
        retriever = getattr(self.rag, "retriever", None)
        if retriever is None:
            raise AttributeError("Your RAG object does not have a 'retriever' attribute.")

        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(question)[: self.top_k]
        elif hasattr(retriever, "invoke"):
            docs = retriever.invoke(question)[: self.top_k]
        else:
            raise TypeError("Retriever must support get_relevant_documents(...) or invoke(...)")

        return [self._doc_text(doc) for doc in docs]

    def retrieval_recall(self, question: str, expected_contexts: Sequence[str]) -> float:
        """Recall@k: how much of the expected evidence is present in retrieved chunks."""
        if not expected_contexts:
            return 1.0

        retrieved = self._retrieved_docs(question)
        matches = 0
        for context in expected_contexts:
            if any(self.similarity_score(context, chunk) > 0.35 for chunk in retrieved):
                matches += 1
        return matches / len(expected_contexts)

    def answer_score(self, question: str, expected_answer: str) -> float:
        """Return a 0..1 score for the generated final answer."""
        generated = self.rag.retrieve(question)
        if not generated or not expected_answer:
            return 0.0
        return self.similarity_score(generated, expected_answer)

    def evaluate_dataset(self, dataset: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate QA items like:
            {
                'question': '... ',
                'expected_answer': '...',
                'expected_contexts': ['...']
            }
        """
        item_results = []
        retrieval_scores = []
        answer_scores = []

        for item in dataset:
            question = str(item["question"])
            expected_answer = str(item.get("expected_answer", ""))
            expected_contexts = item.get("expected_contexts", [])

            retrieval_score = self.retrieval_recall(question, expected_contexts)
            answer_score = self.answer_score(question, expected_answer)

            retrieval_scores.append(retrieval_score)
            answer_scores.append(answer_score)

            item_results.append(
                {
                    "question": question,
                    "retrieval_recall": retrieval_score,
                    "answer_score": answer_score,
                    "generated_answer": self.rag.retrieve(question),
                }
            )

        return {
            "average_retrieval_recall": mean(retrieval_scores) if retrieval_scores else 0.0,
            "average_answer_score": mean(answer_scores) if answer_scores else 0.0,
            "results": item_results,
        }


def evaluate_rag(rag_pipeline: Any, dataset: Iterable[dict[str, Any]]) -> dict[str, Any]:
    evaluator = RAGEvaluator(rag_pipeline)
    return evaluator.evaluate_dataset(dataset)


if __name__ == "__main__":
    import os
    from Rag.Retrival import Retrival

    rag = Retrival(
        path="/Users/apple/Advance_RAG_SYSTEM/Documents/your_file.pdf",
        api_key=os.getenv("OPENAI_API_KEY") or "None",
    )

    dataset = [
        {
            "question": "What does this document say about the system?",
            "expected_answer": "It describes a retrieval augmented generation system for document search and question answering.",
            "expected_contexts": [
                "This project is a RAG system for document retrieval and question answering.",
                "It retrieves relevant document chunks and generates answers from them.",
            ],
        }
    ]

    print(json.dumps(evaluate_rag(rag, dataset), indent=2))
