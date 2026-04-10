from __future__ import annotations

from dataclasses import asdict, dataclass

from src.config import PipelineConfig
from src.data import Scenario
from src.models.openai_client import OpenAITextClient
from src.prompts import JudgeResponseOutput, build_judge_prompt


@dataclass
class Judgment:
    leakage: int
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
    def pedagogy_mean(self) -> float:
        raw_mean = self.pedagogy_raw_mean
        if self.leakage == 1:
            return min(raw_mean, 3.5)
        return raw_mean

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["pedagogy_raw_mean"] = self.pedagogy_raw_mean
        payload["pedagogy_mean"] = self.pedagogy_mean
        return payload


def _build_judgment(result: dict) -> Judgment:
    leakage = int(result.get("leakage", 0))
    correctness = int(result.get("correctness", 3))
    scaffolding = int(result.get("scaffolding", 3))
    self_correction_support = int(result.get("self_correction_support", 3))
    non_overload = int(result.get("non_overload", 3))
    tone = int(result.get("tone", 3))

    if leakage == 1:
        scaffolding = min(scaffolding, 3)
        self_correction_support = min(self_correction_support, 3)

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
