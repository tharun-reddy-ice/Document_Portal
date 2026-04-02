"""
RAG Fusion retriever — generates N query variants, retrieves docs for each,
then re-ranks everything with Reciprocal Rank Fusion (RRF).

Drop-in replacement for any LangChain BaseRetriever:

    fusion = RAGFusionRetriever(
        base_retriever=vs.as_retriever(...),
        llm=llm,
        n_queries=3,
    )
    # use exactly like a normal retriever
    docs = fusion.invoke("What is insulin resistance?")
"""
from __future__ import annotations

import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException


# ── Prompt for multi-query generation ────────────────────────────────────────

_MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """You are an AI assistant helping to improve document retrieval.
Your task is to generate {n_queries} different rephrasings of the user question below.
Each rephrasing must preserve the original intent but use different wording or perspective.
Output ONLY the questions, one per line, with no numbering or preamble.

Original question: {question}"""
)


class RAGFusionRetriever(BaseRetriever):
    """
    Wraps any BaseRetriever and applies RAG Fusion on top:

    1.  Generate `n_queries` alternative phrasings of the user question.
    2.  Run each query (plus the original) through `base_retriever`.
    3.  Merge all result lists with Reciprocal Rank Fusion (RRF).
    4.  Return the top `top_k` unique documents.

    Parameters
    ----------
    base_retriever : BaseRetriever
        The underlying FAISS / other retriever.
    llm : BaseLanguageModel
        Used only for query generation (cheap, single call).
    n_queries : int
        Extra query variants to generate (original always included → n_queries+1 total).
    top_k : int
        Final number of docs to return after fusion.
    rrf_k : int
        RRF constant (higher = smoother rank differences). Default 60 is standard.
    """

    base_retriever: BaseRetriever
    llm: Any  # BaseLanguageModel — Any to avoid Pydantic v1 issues
    n_queries: int = Field(default=3)
    top_k: int = Field(default=5)
    rrf_k: int = Field(default=60)

    class Config:
        arbitrary_types_allowed = True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_query_variants(self, question: str) -> List[str]:
        """Return n_queries alternative phrasings (LLM call)."""
        try:
            chain = _MULTI_QUERY_PROMPT | self.llm | StrOutputParser()
            raw: str = chain.invoke({"question": question, "n_queries": self.n_queries})
            variants = [q.strip() for q in raw.splitlines() if q.strip()]
            log.info(
                "RAG Fusion: generated query variants",
                original=question,
                variants=variants,
            )
            return variants
        except Exception as e:
            log.warning("RAG Fusion: query generation failed, falling back to original only", error=str(e))
            return []  # graceful degradation — fusion still runs on original

    @staticmethod
    def _deduplicate(docs: List[Document]) -> List[Document]:
        """Remove exact-content duplicates, preserving order."""
        seen: set[str] = set()
        out: List[Document] = []
        for d in docs:
            key = d.page_content.strip()
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out

    def _reciprocal_rank_fusion(
        self, ranked_lists: List[List[Document]]
    ) -> List[Document]:
        """
        RRF score for doc d = Σ  1 / (rrf_k + rank_i(d))
        where rank_i is 1-based position in list i.
        Higher score = better.
        """
        scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        for result_list in ranked_lists:
            for rank, doc in enumerate(result_list, start=1):
                key = doc.page_content.strip()
                scores[key] += 1.0 / (self.rrf_k + rank)
                doc_map[key] = doc  # keep latest metadata (same content)

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        fused = [doc_map[k] for k in sorted_keys]

        log.info(
            "RAG Fusion: RRF complete",
            total_unique_docs=len(fused),
            returning=self.top_k,
        )
        return fused[: self.top_k]

    # ── BaseRetriever contract ────────────────────────────────────────────────

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        try:
            variants = self._generate_query_variants(query)
            all_queries = [query] + variants  # original always first

            ranked_lists: List[List[Document]] = []
            for q in all_queries:
                try:
                    docs = self.base_retriever.invoke(q)
                    ranked_lists.append(docs)
                except Exception as e:
                    log.warning("RAG Fusion: retrieval failed for variant", query=q, error=str(e))

            if not ranked_lists:
                raise DocumentPortalException("RAG Fusion: all retrievals failed", sys)

            fused = self._reciprocal_rank_fusion(ranked_lists)
            return self._deduplicate(fused)

        except DocumentPortalException:
            raise
        except Exception as e:
            log.error("RAG Fusion retriever error", error=str(e))
            raise DocumentPortalException("RAGFusionRetriever._get_relevant_documents failed", sys)
