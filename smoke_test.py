"""End-to-end smoke test: load chain, ask 3 questions, print answers."""
import os
# Avoid the OpenMP duplicate-library segfault when llama-cpp-python and
# PyTorch (via sentence-transformers) coexist in the same process on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
from llm_chain import RAGChain

print("=" * 70)
print("Initializing RAG chain (loads embedding model + GGUF LLM)...")
print("=" * 70)
t0 = time.time()
chain = RAGChain(k=6)
print(f"[init] ready in {time.time() - t0:.1f}s\n")

questions = [
    "What is the Transformer architecture?",
    "How does multi-head attention work?",
    "What dataset was used in the experiments?",
]

for i, q in enumerate(questions, 1):
    print("=" * 70)
    print(f"Q{i}: {q}")
    print("=" * 70)
    t0 = time.time()
    result = chain.ask(q, session_id="smoke")
    dt = time.time() - t0
    print(f"\n[standalone] {result['standalone_question']}")
    print(f"\n[answer] ({dt:.1f}s)\n{result['answer']}")
    print(f"\n[sources] {len(result['source_documents'])} chunks:")
    for d in result["source_documents"]:
        meta = d.metadata
        print(f"  - {meta.get('filename')} p.{meta.get('page', '?')}"
              f" — {meta.get('headings', '')[:60]}")
    print()

print("=" * 70)
print("Smoke test complete.")
print("=" * 70)
