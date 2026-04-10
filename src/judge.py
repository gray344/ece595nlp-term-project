from __future__ import annotations

from dataclasses import asdict, dataclass

from src.config import PipelineConfig
from src.data import Scenario
from src.models.openai_client import OpenAITextClient
from src.prompts import JudgeResponseOutput, LeakageLevel, build_judge_prompt

LEAKAGE_CAPS: dict[LeakageLevel, dict[str, float | int | None]] = {
    "none": {
        "max_scaffolding": None,
        "max_self_correction_support": None,
        "max_pedagogy_mean": None,
    },
    "decisive_step": {
        "max_scaffolding": 4,
        "max_self_correction_support": 4,
        "max_pedagogy_mean": 4.2,
    },
    "full_answer": {
        "max_scaffolding": 2,
        "max_self_correction_support": 2,
        "max_pedagogy_mean": 3.0,
    },
}


def normalize_leakage(raw_value: object) -> LeakageLevel:
    text = str(raw_value).strip().lower()
    if text in LEAKAGE_CAPS:
        return text  # type: ignore[return-value]
    if text in {"0", "none", ""}:
        return "none"
    if text in {"1", "true", "yes"}:
        return "decisive_step"
    return "none"


@dataclass
class Judgment:
    leakage: LeakageLevel
    correctness: int
    scaffolding: int
    self_correction_support: int
    non_overload: int
    tone: int
    reasoning: str
    summary: str

    @property
    def pedagogy_raw_mean(self) -> float:
        scores = [
            self.correctness,
            self.scaffolding,
            self.self_correction_support,
            self.non_overload,
            self.tone,
        ]
        return sum(scores) / len(scores)

    @property
    def scaffolding_capped(self) -> int:
        max_value = LEAKAGE_CAPS[self.leakage]["max_scaffolding"]
        if max_value is None:
            return self.scaffolding
        return min(self.scaffolding, int(max_value))

    @property
    def self_correction_support_capped(self) -> int:
        max_value = LEAKAGE_CAPS[self.leakage]["max_self_correction_support"]
        if max_value is None:
            return self.self_correction_support
        return min(self.self_correction_support, int(max_value))

    @property
    def pedagogy_mean(self) -> float:
        capped_scores = [
            self.correctness,
            self.scaffolding_capped,
            self.self_correction_support_capped,
            self.non_overload,
            self.tone,
        ]
        capped_mean = sum(capped_scores) / len(capped_scores)
        max_value = LEAKAGE_CAPS[self.leakage]["max_pedagogy_mean"]
        if max_value is None:
            return capped_mean
        return min(capped_mean, float(max_value))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["scaffolding_capped"] = self.scaffolding_capped
        payload["self_correction_support_capped"] = self.self_correction_support_capped
        payload["pedagogy_raw_mean"] = self.pedagogy_raw_mean
        payload["pedagogy_mean"] = self.pedagogy_mean
        return payload


def _build_judgment(result: dict) -> Judgment:
    leakage = normalize_leakage(result.get("leakage", "none"))
    correctness = int(result.get("correctness", 3))
    scaffolding = int(result.get("scaffolding", 3))
    self_correction_support = int(result.get("self_correction_support", 3))
    non_overload = int(result.get("non_overload", 3))
    tone = int(result.get("tone", 3))

    return Judgment(
        leakage=leakage,
        correctness=correctness,
        scaffolding=scaffolding,
        self_correction_support=self_correction_support,
        non_overload=non_overload,
        tone=tone,
        reasoning=str(result.get("reasoning", result.get("summary", ""))).strip(),
        summary=str(result.get("summary", "")).strip(),
    )


def judge_response(
    client: OpenAITextClient,
    scenario: Scenario,
    response: str,
    config: PipelineConfig,
    selected_act: str | None = None,
) -> Judgment:
    _ = selected_act
    system_prompt, user_prompt = build_judge_prompt(
        scenario=scenario,
        response=response,
        include_student_state=config.include_student_state,
    )
    result = client.complete_json(
        model=config.judge_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_output_tokens=config.judge_max_output_tokens,
        label=f"judge:{scenario.scenario_id}",
        reasoning_effort=config.judge_reasoning_effort,
        reasoning_summary=config.judge_reasoning_summary,
        response_model=JudgeResponseOutput,
    )
    return _build_judgment(result)


async def ajudge_response(
    client: OpenAITextClient,
    scenario: Scenario,
    response: str,
    config: PipelineConfig,
    selected_act: str | None = None,
) -> Judgment:
    _ = selected_act
    system_prompt, user_prompt = build_judge_prompt(
        scenario=scenario,
        response=response,
        include_student_state=config.include_student_state,
    )
    result = await client.acomplete_json(
        model=config.judge_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
        max_output_tokens=config.judge_max_output_tokens,
        label=f"judge:{scenario.scenario_id}",
        reasoning_effort=config.judge_reasoning_effort,
        reasoning_summary=config.judge_reasoning_summary,
        response_model=JudgeResponseOutput,
    )
    return _build_judgment(result)
