"""
Step 4: Offline Ragas Evaluation Script.

Runs the local RAG pipeline against a labelled test set and computes:
  - Context Precision
  - Faithfulness
  - Answer Relevancy

The Ragas judge LLM and embeddings are wired to the SAME local llama.cpp +
HuggingFace stack used by the RAG pipeline, so evaluation is fully offline.

Outputs metrics to JSON and CSV under ./eval_results/.
"""
from __future__ import annotations

import os
# Avoid the OpenMP duplicate-library segfault on Windows when llama-cpp-python
# and PyTorch coexist in the same process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    faithfulness,
)
from ragas.run_config import RunConfig

from llm_chain import RAGChain, load_local_llm, LLMConfig, QWEN3_NO_THINK
from vector_store import build_default_store, DEFAULT_EMBED_MODEL

logger = logging.getLogger("evaluate")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

OUTPUT_DIR = Path("./eval_results")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
@dataclass
class TestCase:
    question: str
    ground_truth: str


# A small built-in test set. Replace or extend with your own questions that
# match the PDFs you have indexed in ChromaDB.
DEFAULT_TEST_SET: List[TestCase] = [
    TestCase(
        question="What is the main topic of the document?",
        ground_truth="The document discusses its primary subject matter as introduced in its abstract or introduction.",
    ),
    TestCase(
        question="Summarize the key conclusions.",
        ground_truth="The conclusions summarize the principal findings and recommendations of the document.",
    ),
    TestCase(
        question="List any tables or numerical data referenced.",
        ground_truth="The document includes tables or numerical data described in the relevant sections.",
    ),
]


def load_test_set(path: Optional[Path]) -> List[TestCase]:
    if path is None:
        logger.info("Using built-in default test set (%d items).", len(DEFAULT_TEST_SET))
        return DEFAULT_TEST_SET

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test set not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [TestCase(question=row["question"], ground_truth=row["ground_truth"]) for row in raw]
    logger.info("Loaded %d test cases from %s", len(cases), path)
    return cases


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def run_rag_over_cases(chain: RAGChain, cases: List[TestCase]) -> Dataset:
    questions: List[str] = []
    answers: List[str] = []
    contexts: List[List[str]] = []
    ground_truths: List[str] = []

    for i, case in enumerate(cases, start=1):
        logger.info("[%d/%d] Asking: %s", i, len(cases), case.question)
        result = chain.ask_no_memory(case.question)

        questions.append(case.question)
        answers.append(result["answer"])
        contexts.append([d.page_content for d in result["source_documents"]])
        ground_truths.append(case.ground_truth)

    # Ragas expects these exact column names.
    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------
def run_evaluation(
    test_set_path: Optional[Path] = None,
    output_dir: Path = OUTPUT_DIR,
    include_context_precision: bool = False,
    include_faithfulness: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build the RAG chain (shares one local LLM instance with Ragas judge).
    logger.info("Initializing local LLM and vector store...")
    llm = load_local_llm(LLMConfig())
    store = build_default_store()
    chain = RAGChain(vector_store=store, llm=llm)

    # 2. Generate predictions.
    cases = load_test_set(test_set_path)
    dataset = run_rag_over_cases(chain, cases)

    # 3. Wire Ragas to the LOCAL judge LLM + embeddings (no API calls).
    # `llm` is now a ChatOpenAI pointed at our local llama-server, so judge
    # calls are just HTTP requests against the GPU-accelerated server. No
    # need for the monkey-patching workaround we used with ChatLlamaCpp.
    judge_llm = LangchainLLMWrapper(llm)
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=DEFAULT_EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    # All three metrics enabled by default now that the LLM is GPU-served
    # via llama-server (sub-5s per call). Flags kept for backwards compat.
    metrics = [context_precision, faithfulness, answer_relevancy]

    # CRITICAL: llama-cpp-python's underlying llama.cpp model instance is NOT
    # thread-safe. Concurrent decode calls corrupt the KV cache and produce
    # `llama_decode returned -1`. Force ragas to run jobs serially.
    # timeout=1200s (20 min) tolerates the long judge generations a small
    # CPU-only model produces; max_retries=1 to avoid wasting compute on
    # systematic failures.
    run_config = RunConfig(
        max_workers=1,
        timeout=1200,
        max_retries=1,
    )

    logger.info("Running Ragas evaluation with metrics: %s", [m.name for m in metrics])
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=run_config,
            raise_exceptions=False,
            show_progress=True,
        )
    except Exception as exc:
        logger.exception("Ragas evaluation failed.")
        raise RuntimeError("Ragas evaluation failed; see logs.") from exc

    # 4. Persist results.
    df: pd.DataFrame = result.to_pandas()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"ragas_results_{timestamp}.csv"
    json_path = output_dir / f"ragas_results_{timestamp}.json"

    df.to_csv(csv_path, index=False)
    metric_cols = {"context_precision", "faithfulness", "answer_relevancy"}
    summary = {
        "timestamp": timestamp,
        "metrics": {
            col: (float(df[col].mean()) if df[col].notna().any() else None)
            for col in df.columns if col in metric_cols
        },
        "num_samples": len(df),
        "csv_path": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Evaluation complete. Summary: %s", summary["metrics"])
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Offline RAG evaluation with Ragas.")
    parser.add_argument(
        "--test-set",
        type=Path,
        default=None,
        help="Path to JSON list of {question, ground_truth} objects. "
             "If omitted, the built-in default set is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write metrics CSV/JSON.",
    )
    parser.add_argument(
        "--include-context-precision",
        action="store_true",
        help="Also compute 'context_precision' (moderate cost).",
    )
    parser.add_argument(
        "--include-faithfulness",
        action="store_true",
        help="Also compute the heavy 'faithfulness' metric (slow; may "
             "hit LLMDidNotFinishException on small CPU models).",
    )
    args = parser.parse_args(argv)

    run_evaluation(
        test_set_path=args.test_set,
        output_dir=args.output_dir,
        include_context_precision=args.include_context_precision,
        include_faithfulness=args.include_faithfulness,
    )


if __name__ == "__main__":
    main()
