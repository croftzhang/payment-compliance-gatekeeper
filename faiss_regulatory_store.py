"""FAISS vector store for Regulatory Rules and Compliance Runbooks (with NumPy fallback).

The index is FAISS in both modes. What changes is where the vectors come from:

  * no embedder (default) - a keyword-count vector written by hand. It is not a
    semantic embedding; it exists so the code and tests run offline.
  * embedder supplied     - real embeddings (see `gemini_client.GeminiEmbedder`),
    which is what makes the free-text retrieval in the agent meaningful.

The rule and runbook texts are illustrative policy written for this demo. They
paraphrase the general shape of public rules and are not statements of current law.
"""

from typing import List, Optional, Protocol, Tuple
import numpy as np
from models import ComplianceRunbook, RegulatoryRule

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class Embedder(Protocol):
    """Anything that turns text into vectors of a fixed dimension."""

    dim: int

    def embed_documents(self, texts: List[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class _NumpyFlatIPIndex:
    """Exact inner-product index fallback when faiss-cpu is not installed."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.empty((0, dim), dtype="float32")

    def add(self, x: np.ndarray) -> None:
        self.vectors = np.vstack([self.vectors, x.astype("float32")])

    def search(self, q: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.dot(q.astype("float32"), self.vectors.T)
        idx = np.argsort(-scores, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(scores, idx, axis=1)
        return top_scores, idx


def _normalize_l2(x: np.ndarray) -> None:
    if FAISS_AVAILABLE:
        faiss.normalize_L2(x)
    else:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        x /= norms


class FAISSRegulatoryRetriever:
    """Dual vector retriever for Regulatory Rules and Operational Runbooks."""

    RULE_CORPUS = [
        {
            "rule_id": "REG-OFAC-50PCT",
            "title": "OFAC Revised 50 Percent Rule (Aggregate Ownership)",
            "content": (
                "Any entity owned in the aggregate, directly or indirectly, 50 percent "
                "or more by one or more blocked persons is itself considered a blocked "
                "person. Transactions involving such entities must be rejected or blocked "
                "immediately regardless of how many intermediate shell layers exist."
            ),
        },
        {
            "rule_id": "REG-FINCEN-CDD-25",
            "title": "FinCEN Customer Due Diligence (CDD) Beneficial Ownership Threshold",
            "content": (
                "Financial institutions must identify and verify any individual who, "
                "directly or indirectly, owns 25 percent or more of the equity interests "
                "of a legal entity customer, and perform enhanced due diligence (EDD) on "
                "high-risk offshore jurisdictions."
            ),
        },
        {
            "rule_id": "REG-FATF-R16",
            "title": "FATF Recommendation 16 - Cross-Border Wire Transfers",
            "content": (
                "Cross-border wire transfers exceeding USD 10,000 require verified "
                "originator and beneficiary metadata. Transfers involving high-risk "
                "corridors with opaque multi-hop ownership between 25% and 49.9% "
                "require manual compliance escalation."
            ),
        },
        {
            "rule_id": "REG-DOMESTIC-LOW",
            "title": "Standard Domestic Low-Risk Clearing Policy",
            "content": (
                "Domestic USD clearing transactions below USD 25,000 with verified US "
                "beneficiaries and zero sanctioned ownership may be cleared via fast-path."
            ),
        },
        {
            "rule_id": "REG-SANCT-JURIS",
            "title": "Comprehensively Sanctioned Jurisdictions - Payment Nexus Policy",
            "content": (
                "A payment connected, directly or indirectly, to a comprehensively "
                "sanctioned territory under bank policy (Iran, North Korea, Cuba, the "
                "Crimea region) - goods, vessels, "
                "ports, consignees or end parties located there, or a payment made for a "
                "party there - is not to be released. Hold it and refer it to the "
                "sanctions desk for review. A request to omit, alter or remove a "
                "reference to such a territory from the payment message is a red flag "
                "and is referred in the same way."
            ),
        },
        {
            "rule_id": "REG-SDN-LIST-MATCH",
            "title": "Sanctions List Name Match - Adjudication Policy",
            "content": (
                "A payment to or from a party on the SDN list is prohibited. A screening "
                "hit on a listed name or alias is settled with a secondary identifier "
                "such as country or registration number: a hit confirmed as the same "
                "party is rejected and reported; a hit shown to be a different party "
                "with a similar name is documented and released; a hit the available "
                "information cannot settle is held for review."
            ),
        },
    ]

    RUNBOOK_CORPUS = [
        {
            "runbook_id": "RB-OFAC-FREEZE-01",
            "title": "Sanctioned UBO Aggregate Ownership Block & OFAC Report Runbook",
            "trigger_condition": "50 percent aggregate blocked shell entities OFAC velocity restructuring",
            "action_steps": [
                "1. Immediately place payment funds into suspense holding account (GL-9902).",
                "2. Lock the beneficiary profile and every shell entity on the sanctioned ownership paths.",
                "3. File initial Blocking Report with OFAC Compliance within 10 business days.",
            ],
        },
        {
            "runbook_id": "RB-EDD-ESCALATE-02",
            "title": "Cross-Border Correspondent Wire Enhanced Due Diligence Runbook",
            "trigger_condition": "cross-border wire 10,000 25 percent offshore manual escalation",
            "action_steps": [
                "1. Pause STP clearing and route wire packet to L2 AML Investigator queue.",
                "2. Request certified UBO registry extract and MT103 field 50F provenance.",
            ],
        },
        {
            "runbook_id": "RB-DOMESTIC-STP-03",
            "title": "Domestic Low-Risk Straight-Through Processing (STP) Runbook",
            "trigger_condition": "domestic fast-path low-risk US normal velocity",
            "action_steps": [
                "1. Release payment to Fedwire STP clearing queue.",
                "2. Archive telemetry check in standard audit log.",
            ],
        },
        {
            "runbook_id": "RB-SANCT-NEXUS-04",
            "title": "Sanctioned Territory Nexus Hold & Referral Runbook",
            "trigger_condition": (
                "payment details or shipping documents point to a comprehensively "
                "sanctioned territory, port, vessel or party"
            ),
            "action_steps": [
                "1. Hold the payment; neither release nor return the funds while it is under review.",
                "2. Request the commercial documents behind it (invoice, bill of lading, end-user statement).",
                "3. Refer the case to the sanctions desk with the payment text and the documents.",
            ],
        },
        {
            "runbook_id": "RB-LIST-HIT-05",
            "title": "Sanctions List Hit Adjudication Runbook",
            "trigger_condition": "sdn list name hit alias identifier adjudication",
            "action_steps": [
                "1. Compare the party's country and registration number with the list entry.",
                "2. Same party: reject the payment and file the report. Different party: record why and release.",
                "3. Not settled: hold the payment and request identifying documents from the originator.",
            ],
        },
    ]

    def __init__(self, embedding_dim: int = 64, embedder: Optional[Embedder] = None):
        self.embedder = embedder
        self.dim = embedder.dim if embedder is not None else embedding_dim
        self.vector_source = "embedder" if embedder is not None else "keyword-hash"
        IndexCls = faiss.IndexFlatIP if FAISS_AVAILABLE else _NumpyFlatIPIndex

        self.rule_index = IndexCls(self.dim)
        self.rule_vectors = self._embed_documents(
            [self._rule_text(d) for d in self.RULE_CORPUS]
        )
        self.rule_index.add(self.rule_vectors)

        self.rb_index = IndexCls(self.dim)
        self.rb_vectors = self._embed_documents(
            [f"{d['title']} {d['trigger_condition']}" for d in self.RUNBOOK_CORPUS]
        )
        self.rb_index.add(self.rb_vectors)

    def _rule_text(self, doc: dict) -> str:
        # The keyword vectors were built from the content alone; a real embedding
        # benefits from the title as well.
        if self.embedder is None:
            return doc["content"]
        return f"{doc['title']}. {doc['content']}"

    def _as_matrix(self, vectors: np.ndarray, rows: int) -> np.ndarray:
        matrix = np.array(vectors, dtype="float32").reshape(rows, -1)
        if matrix.shape[1] != self.dim:
            raise ValueError(
                f"Embedder returned {matrix.shape[1]}-dimensional vectors, expected {self.dim}."
            )
        _normalize_l2(matrix)
        return matrix

    def _embed_documents(self, texts: List[str]) -> np.ndarray:
        if self.embedder is None:
            return self._encode_texts(texts)
        return self._as_matrix(self.embedder.embed_documents(texts), len(texts))

    def _embed_query(self, text: str) -> np.ndarray:
        if self.embedder is None:
            return self._encode_texts([text])
        return self._as_matrix(self.embedder.embed_query(text), 1)

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Offline stand-in: keyword counts plus a little character noise. Not semantic."""
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        keywords = [
            "50", "blocked", "aggregate", "indirectly", "shell",
            "25", "cdd", "offshore", "edd",
            "cross-border", "10,000", "manual", "escalation",
            "domestic", "fast-path", "low-risk", "us", "velocity", "restructuring",
            "sdn", "alias", "identifier",
        ]
        for i, text in enumerate(texts):
            lower = text.lower()
            for idx, kw in enumerate(keywords):
                if kw in lower:
                    vectors[i, idx % self.dim] += 1.0
            for ch_idx, ch in enumerate(lower[: self.dim]):
                vectors[i, ch_idx] += (ord(ch) % 7) * 0.02
        _normalize_l2(vectors)
        return vectors

    def search_rules(self, query: str, top_k: int = 2) -> List[RegulatoryRule]:
        q_vec = self._embed_query(query)
        scores, indices = self.rule_index.search(q_vec, top_k)
        results: List[RegulatoryRule] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self.RULE_CORPUS[int(idx)]
            results.append(
                RegulatoryRule(
                    rule_id=doc["rule_id"],
                    title=doc["title"],
                    content=doc["content"],
                    similarity_score=round(float(score), 4),
                )
            )
        return results

    def search_runbooks(self, query: str, top_k: int = 1) -> List[ComplianceRunbook]:
        q_vec = self._embed_query(query)
        _, indices = self.rb_index.search(q_vec, top_k)
        results: List[ComplianceRunbook] = []
        for idx in indices[0]:
            if idx < 0:
                continue
            rb = self.RUNBOOK_CORPUS[int(idx)]
            results.append(ComplianceRunbook(**rb))
        return results
