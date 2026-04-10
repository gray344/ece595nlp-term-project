from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional in early setup
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PipelineConfig:
    generation_model: str = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
    act_selection_model: str = os.getenv("ACT_SELECTION_MODEL", "gpt-4o-mini")
    judge_model: str = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
    judge_reasoning_effort: str = os.getenv("JUDGE_REASONING_EFFORT", "")
    judge_reasoning_summary: str = os.getenv("JUDGE_REASONING_SUMMARY", "")
    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "250"))
    judge_max_output_tokens: int = int(
        os.getenv("JUDGE_MAX_OUTPUT_TOKENS", os.getenv("MAX_OUTPUT_TOKENS", "250"))
    )
    max_concurrency: int = int(os.getenv("MAX_CONCURRENCY", "4"))
    default_repetitions: int = int(os.getenv("DEFAULT_REPETITIONS", "1"))
    scenario_path: str = os.getenv("SCENARIO_PATH", "data/scenarios.jsonl")
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")
    debug_api: bool = _env_flag("DEBUG_API", False)
    include_student_state: bool = _env_flag("INCLUDE_STUDENT_STATE", True)


def load_config() -> PipelineConfig:
    return PipelineConfig()
