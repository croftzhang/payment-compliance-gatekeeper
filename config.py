"""Settings for the cloud backends, read from environment variables.

Nothing here is needed for the offline run (rule path, local graph, keyword vectors).
Which backend is used is never decided by an environment variable: it is chosen
explicitly in code or on the command line, so the tests stay offline whatever is set.
"""

import os
from dataclasses import dataclass
from typing import Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class CloudConfig:
    project: Optional[str]
    # Gemini (decision drafting)
    gemini_model: str
    gemini_location: str
    thinking_level: Optional[str]
    temperature: Optional[float]
    # Gemini embeddings (retrieval)
    embed_model: str
    embed_location: str
    embed_dim: int
    cache_dir: Optional[str]
    # Vertex AI Gen AI evaluation service / Experiments
    vertex_location: str
    # Spanner Graph
    spanner_instance: str
    spanner_database: str
    # List prices in USD per 1M tokens, used only to estimate cost in the report
    price_in_per_m: float
    price_out_per_m: float

    def require_project(self) -> str:
        if not self.project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is not set. In PowerShell: "
                '$env:GOOGLE_CLOUD_PROJECT = "your-project-id"'
            )
        return self.project


def load_config() -> CloudConfig:
    temperature = _env("PCG_TEMPERATURE")
    thinking = _env("PCG_THINKING_LEVEL", "LOW")
    return CloudConfig(
        project=_env("GOOGLE_CLOUD_PROJECT"),
        # gemini-3.8-flash is served from the `global` endpoint and the `us` / `eu`
        # multi-regions; it is not offered in us-central1.
        gemini_model=_env("PCG_GEMINI_MODEL", "gemini-3.8-flash"),
        gemini_location=_env("PCG_GEMINI_LOCATION", "global"),
        thinking_level=None if thinking.upper() in ("NONE", "DEFAULT") else thinking.upper(),
        temperature=float(temperature) if temperature is not None else None,
        embed_model=_env("PCG_EMBED_MODEL", "gemini-embedding-001"),
        embed_location=_env("PCG_EMBED_LOCATION", "us-central1"),
        embed_dim=int(_env("PCG_EMBED_DIM", "768")),
        cache_dir=_env("PCG_CACHE_DIR"),
        vertex_location=_env("PCG_VERTEX_LOCATION", "us-central1"),
        spanner_instance=_env("PCG_SPANNER_INSTANCE", "pcg-trial"),
        spanner_database=_env("PCG_SPANNER_DATABASE", "compliance"),
        price_in_per_m=float(_env("PCG_PRICE_IN_PER_M", "0.75")),
        price_out_per_m=float(_env("PCG_PRICE_OUT_PER_M", "3.75")),
    )
