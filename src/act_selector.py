from __future__ import annotations

from src.config import PipelineConfig
from src.data import Scenario
from src.models.openai_client import OpenAITextClient
from src.prompts import ActSelectionOutput, TUTOR_ACTS, build_act_selection_prompt


def select_act(client: OpenAITextClient, scenario: Scenario, config: PipelineConfig) -> dict[str, str]:
    system_prompt, user_prompt = build_act_selection_prompt(scenario)
    result = client.complete_json(
        model=config.act_selection_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        label=f"act_selection:{scenario.scenario_id}",
        response_model=ActSelectionOutput,
    )
    selected_act = str(result.get("selected_act", "")).strip()
    if selected_act not in TUTOR_ACTS:
        selected_act = "give_hint"
    rationale = str(result.get("rationale", "")).strip()
    return {"selected_act": selected_act, "rationale": rationale}


async def aselect_act(
    client: OpenAITextClient,
    scenario: Scenario,
    config: PipelineConfig,
) -> dict[str, str]:
    system_prompt, user_prompt = build_act_selection_prompt(scenario)
    result = await client.acomplete_json(
        model=config.act_selection_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        label=f"act_selection:{scenario.scenario_id}",
        response_model=ActSelectionOutput,
    )
    selected_act = str(result.get("selected_act", "")).strip()
    if selected_act not in TUTOR_ACTS:
        selected_act = "give_hint"
    rationale = str(result.get("rationale", "")).strip()
    return {"selected_act": selected_act, "rationale": rationale}
