

from __future__ import annotations

import logging

from src.agents.graph import build_graph
from src.agents.state import MedAssistState, empty_state
from src.generation.llm import get_llm
from src.generation.prompts import fallback_rag_prompt
from src.retrieval.faiss_store import load_faiss_retriever
from src.utils.confidence import apply_confidence_gate, format_citations
from src.utils.constants import MEDICAL_DISCLAIMER

logger = logging.getLogger(__name__)


class MedAssistPipeline:
    
    def __init__(self, faiss_index_path: str | None = None) -> None:
        self._graph     = build_graph(faiss_index_path)
        self._llm       = get_llm()
        self._retriever = load_faiss_retriever(faiss_index_path)

    
    def run(self, question: str, verbose: bool = True) -> MedAssistState:
        sep = "=" * 65
        if verbose:
            print(f"\n{sep}")
            print(f"❓  {question}")
            print(sep)

        state = empty_state(question)

        try:
            final_state = self._graph.invoke(state)
            answer = final_state.get("final_answer") or ""

            if not answer or len(answer.strip()) < 20:
                logger.warning("Graph returned an empty answer — running fallback RAG.")
                if verbose:
                    print("⚠️  Empty answer from graph — running fallback RAG.")
                answer = self._direct_rag_fallback(question)
                final_state["final_answer"] = answer

        except Exception as exc:
            logger.error("Graph raised an exception: %s", exc)
            if verbose:
                print(f"⚠️  Graph exception: {exc} — running fallback RAG.")
            answer = self._direct_rag_fallback(question)
            final_state = {**state, "final_answer": answer, "error": str(exc)}

        if verbose:
            self._print_result(final_state)

        return final_state

   
    def _direct_rag_fallback(self, question: str) -> str:
        
        logger.info("[FALLBACK] Running direct RAG …")
        try:
            docs    = self._retriever.get_relevant_documents(question)
            context = "\n\n".join(d.page_content[:500] for d in docs[:3])
            prompt  = fallback_rag_prompt(question, context)
            answer  = self._llm.invoke(prompt).strip()
            gated, score = apply_confidence_gate(answer, docs)
            citations    = format_citations(docs)
            return (
                gated
                + citations
                + f"\n\n[Fallback RAG | Confidence: {score:.2f}]"
            )
        except Exception as exc:
            logger.error("[FALLBACK] Also failed: %s", exc)
            return f"[Fallback also failed: {exc}]\n{MEDICAL_DISCLAIMER}"

    @staticmethod
    def _print_result(state: MedAssistState) -> None:
        """Pretty-print the final state to stdout."""
        print(f"\n{'─' * 65}")
        print("🤖  FINAL ANSWER:")
        print(state.get("final_answer", ""))
        if state.get("key_findings"):
            print("\n📌  KEY FINDINGS:")
            for finding in state["key_findings"]:
                print(f"    • {finding}")
        conf = state.get("confidence")
        if conf is not None:
            print(
                f"\n📊  Confidence: {conf:.3f} | "
                f"Retries: {state.get('retry_count', 0)} | "
                f"Tool: {state.get('tool')}"
            )
