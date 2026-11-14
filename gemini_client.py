"""Gemini adapters: decision drafting with structured output, and text embeddings.

Both go through the google-genai SDK against the Gemini platform (Vertex AI)
endpoint of your Google Cloud project, authenticated with Application Default
Credentials (`gcloud auth application-default login`).

Nothing in this module is imported by the offline path.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from config import CloudConfig, load_config
from models import ComplianceDecision

_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


def make_genai_client(project: str, location: str):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed. Run: pip install -r requirements-cloud.txt"
        ) from exc
    return genai.Client(vertexai=True, project=project, location=location)


def _call_with_retry(fn: Callable[[], Any], attempts: int = 4, base_delay: float = 2.0) -> Any:
    """Retry rate-limit and server errors with a growing pause (2s, 6s, 18s)."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - the SDK error carries an HTTP code
            code = getattr(exc, "code", None)
            if code not in _RETRYABLE_CODES or attempt == attempts - 1:
                raise
            time.sleep(base_delay * (3 ** attempt))
    raise RuntimeError("unreachable")


class GeminiDecisionLLM:
    """Drafts a ComplianceDecision from the prompt. `invoke` never raises.

    A failed call or an unparseable reply returns None; the provenance gate then
    treats the draft as missing and the graph retries or falls back to escalation.
    Every call is recorded in `self.calls` (latency, token counts, any error) so the
    evaluation can report cost and distinguish API failures from bad drafts.
    """

    def __init__(self, cfg: Optional[CloudConfig] = None, client: Any = None):
        self.cfg = cfg or load_config()
        self.client = client or make_genai_client(
            self.cfg.require_project(), self.cfg.gemini_location
        )
        self.calls: List[Dict[str, Any]] = []

    def _generation_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": ComplianceDecision,
        }
        if self.cfg.temperature is not None:
            config["temperature"] = self.cfg.temperature
        if self.cfg.thinking_level:
            config["thinking_config"] = {"thinking_level": self.cfg.thinking_level}
        return config

    @staticmethod
    def _parse(response: Any) -> Optional[ComplianceDecision]:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, ComplianceDecision):
            return parsed
        try:
            if isinstance(parsed, dict):
                return ComplianceDecision.model_validate(parsed)
            text = getattr(response, "text", None)
            if text:
                return ComplianceDecision.model_validate_json(text)
        except Exception:  # noqa: BLE001 - schema or JSON error both mean "no usable draft"
            return None
        return None

    def invoke(self, prompt: str) -> Optional[ComplianceDecision]:
        record: Dict[str, Any] = {
            "model": self.cfg.gemini_model,
            "prompt_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0,
            "latency_s": 0.0, "error": None, "parsed": False,
        }
        started = time.perf_counter()
        try:
            response = _call_with_retry(
                lambda: self.client.models.generate_content(
                    model=self.cfg.gemini_model,
                    contents=prompt,
                    config=self._generation_config(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            record["latency_s"] = round(time.perf_counter() - started, 3)
            record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            self.calls.append(record)
            return None

        record["latency_s"] = round(time.perf_counter() - started, 3)
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            record["prompt_tokens"] = int(getattr(usage, "prompt_token_count", 0) or 0)
            record["output_tokens"] = int(getattr(usage, "candidates_token_count", 0) or 0)
            record["thoughts_tokens"] = int(getattr(usage, "thoughts_token_count", 0) or 0)
        decision = self._parse(response)
        record["parsed"] = decision is not None
        if decision is None:
            record["error"] = "reply did not parse as a ComplianceDecision"
        self.calls.append(record)
        return decision

    def usage_summary(self) -> Dict[str, Any]:
        """Totals over all calls so far, with a cost estimate at the configured list prices.

        Thinking tokens are billed as output tokens, so they are priced as output.
        """
        prompt = sum(c["prompt_tokens"] for c in self.calls)
        output = sum(c["output_tokens"] + c["thoughts_tokens"] for c in self.calls)
        cost = (prompt * self.cfg.price_in_per_m + output * self.cfg.price_out_per_m) / 1_000_000
        return {
            "calls": len(self.calls),
            "api_errors": sum(1 for c in self.calls if c["error"] and not c["parsed"]
                              and "did not parse" not in c["error"]),
            "unparsed_replies": sum(1 for c in self.calls if c["error"]
                                    and "did not parse" in c["error"]),
            "prompt_tokens": prompt,
            "output_tokens_incl_thinking": output,
            "est_cost_usd": round(cost, 4),
            "mean_latency_s": round(
                sum(c["latency_s"] for c in self.calls) / len(self.calls), 2
            ) if self.calls else 0.0,
        }


class GeminiEmbedder:
    """Text embeddings for the FAISS retriever, with a small on-disk cache.

    gemini-embedding-001 accepts one input per request on this endpoint, so texts are
    embedded one at a time. Documents and queries use different task types, which is
    how the model is meant to be used for retrieval. Vectors are cached by
    (task type, text) so the corpus is embedded once, not on every run.
    """

    def __init__(
        self,
        cfg: Optional[CloudConfig] = None,
        client: Any = None,
        cache_dir: Optional[Path] = None,
    ):
        self.cfg = cfg or load_config()
        self.dim = self.cfg.embed_dim
        self.client = client or make_genai_client(
            self.cfg.require_project(), self.cfg.embed_location
        )
        self.api_calls = 0
        chosen = cache_dir or self.cfg.cache_dir
        directory = Path(chosen) if chosen else Path(__file__).resolve().parent / ".pcg_cache"
        safe_model = self.cfg.embed_model.replace("/", "_")
        self._cache_path = directory / f"embeddings_{safe_model}_{self.dim}.json"
        self._cache: Dict[str, List[float]] = {}
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._cache = {}

    def _save_cache(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except OSError:
            pass  # the cache is an optimisation; a read-only folder must not stop a run

    def _embed_one(self, text: str, task_type: str) -> List[float]:
        key = hashlib.sha256(f"{task_type}\n{text}".encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]
        response = _call_with_retry(
            lambda: self.client.models.embed_content(
                model=self.cfg.embed_model,
                contents=text,
                config={"task_type": task_type, "output_dimensionality": self.dim},
            )
        )
        self.api_calls += 1
        values = [float(v) for v in response.embeddings[0].values]
        if len(values) != self.dim:
            raise ValueError(
                f"{self.cfg.embed_model} returned {len(values)} dimensions, expected {self.dim}. "
                "Set PCG_EMBED_DIM to match, or to a size the model supports."
            )
        self._cache[key] = values
        self._save_cache()
        return values

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return np.array(
            [self._embed_one(t, "RETRIEVAL_DOCUMENT") for t in texts], dtype="float32"
        )

    def embed_query(self, text: str) -> np.ndarray:
        return np.array([self._embed_one(text, "RETRIEVAL_QUERY")], dtype="float32")
