from __future__ import annotations

from src.config import PipelineConfig
from src.data import Scenario
from src.models.openai_client import OpenAITextClient
from src.prompts import build_response_prompt


def generate_act_conditioned_response(
    client: OpenAITextClient,
    scenario: Scenario,
    selected_act: str,
    config: PipelineConfig,
) -> str:
    system_prompt, user_prompt = build_response_prompt(
        scenario,
        selected_act,
        include_student_state=config.include_student_state,
    )
    return client.complete_text(
        model=config.generation_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        label=f"act_response:{scenario.scenario_id}:{selected_act}",
    )


async def agenerate_act_conditioned_response(
    client: OpenAITextClient,
    scenario: Scenario,
    selected_act: str,
    config: PipelineConfig,
) -> str:
    system_prompt, user_prompt = build_response_prompt(
        scenario,
        selected_act,
        include_student_state=config.include_student_state,
    )
    return await client.acomplete_text(
        model=config.generation_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        label=f"act_response:{scenario.scenario_id}:{selected_act}",
    )
