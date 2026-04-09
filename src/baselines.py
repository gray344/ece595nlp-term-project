from __future__ import annotations

from src.config import PipelineConfig
from src.data import Scenario
from src.models.openai_client import OpenAITextClient
from src.prompts import build_baseline_prompt


def generate_direct_baseline(
    client: OpenAITextClient,
    scenario: Scenario,
    config: PipelineConfig,
) -> str:
    system_prompt, user_prompt = build_baseline_prompt(scenario, pedagogical=False)
    return client.complete_text(
        model=config.generation_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        label=f"baseline_direct:{scenario.scenario_id}",
    )


async def agenerate_direct_baseline(
    client: OpenAITextClient,
    scenario: Scenario,
    config: PipelineConfig,
) -> str:
    system_prompt, user_prompt = build_baseline_prompt(scenario, pedagogical=False)
    return await client.acomplete_text(
        model=config.generation_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        label=f"baseline_direct:{scenario.scenario_id}",
    )


def generate_pedagogical_baseline(
    client: OpenAITextClient,
    scenario: Scenario,
    config: PipelineConfig,
) -> str:
    system_prompt, user_prompt = build_baseline_prompt(scenario, pedagogical=True)
    return client.complete_text(
        model=config.generation_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        label=f"baseline_pedagogical:{scenario.scenario_id}",
    )


async def agenerate_pedagogical_baseline(
    client: OpenAITextClient,
    scenario: Scenario,
    config: PipelineConfig,
) -> str:
    system_prompt, user_prompt = build_baseline_prompt(scenario, pedagogical=True)
    return await client.acomplete_text(
        model=config.generation_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        label=f"baseline_pedagogical:{scenario.scenario_id}",
    )
