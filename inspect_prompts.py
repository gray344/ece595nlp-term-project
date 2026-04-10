from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.data import Scenario, load_scenarios
from src.prompts import (
    TUTOR_ACTS,
    build_act_selection_prompt,
    build_baseline_prompt,
    build_judge_prompt,
    build_response_prompt,
)

DEFAULT_JUDGE_RESPONSE = "[Tutor response under evaluation]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the fully assembled prompts for each tutoring system to markdown."
    )
    parser.add_argument(
        "--scenario-id",
        default="",
        help="Scenario id to render. Defaults to the first scenario in the scenario file.",
    )
    parser.add_argument(
        "--scenario-path",
        default="",
        help="Optional override for the scenario file path. Defaults to SCENARIO_PATH from config.",
    )
    parser.add_argument(
        "--selected-act",
        default="",
        choices=sorted(TUTOR_ACTS),
        help="Assigned act to render for the act-conditioned response prompt. Defaults to the scenario's ideal_move.",
    )
    parser.add_argument(
        "--all-acts",
        action="store_true",
        help="Render the act-conditioned response prompt for every tutor act.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Markdown output path. Defaults to a state-aware file under outputs/.",
    )
    parser.add_argument(
        "--judge-response",
        default=DEFAULT_JUDGE_RESPONSE,
        help="Tutor response text to embed in the rendered judge prompt.",
    )
    student_state_group = parser.add_mutually_exclusive_group()
    student_state_group.add_argument(
        "--include-student-state",
        dest="include_student_state",
        action="store_true",
        help="Pass the scenario's student_state label to the rendered tutor prompts.",
    )
    student_state_group.add_argument(
        "--omit-student-state",
        dest="include_student_state",
        action="store_false",
        help="Render tutor prompts without the student_state label.",
    )
    parser.set_defaults(include_student_state=None)
    return parser.parse_args()


def select_scenario(scenarios: list[Scenario], scenario_id: str) -> Scenario:
    if scenario_id:
        for scenario in scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise ValueError(f"No scenario found for scenario_id={scenario_id}")
    if not scenarios:
        raise ValueError("No scenarios found in the scenario file.")
    return scenarios[0]


def render_prompt_section(title: str, system_prompt: str, user_prompt: str) -> str:
    return "\n".join(
        [
            f"## {title}",
            "",
            "### System Prompt",
            "```text",
            system_prompt,
            "```",
            "",
            "### User Prompt",
            "```text",
            user_prompt,
            "```",
            "",
        ]
    )


def render_act_response_sections(
    scenario: Scenario,
    *,
    include_student_state: bool,
    selected_act: str,
    all_acts: bool,
) -> list[str]:
    acts = sorted(TUTOR_ACTS) if all_acts else [selected_act]
    sections: list[str] = []
    for act_name in acts:
        system_prompt, user_prompt = build_response_prompt(
            scenario,
            act_name,
            include_student_state=include_student_state,
        )
        sections.append(
            render_prompt_section(
                f"Act-Conditioned Response Prompt ({act_name})",
                system_prompt,
                user_prompt,
            )
        )
    return sections


def build_markdown(
    scenario: Scenario,
    *,
    include_student_state: bool,
    selected_act: str,
    all_acts: bool,
    scenario_path: str,
    judge_response: str,
) -> str:
    direct_system_prompt, direct_user_prompt = build_baseline_prompt(
        scenario,
        pedagogical=False,
        include_student_state=include_student_state,
    )
    pedagogical_system_prompt, pedagogical_user_prompt = build_baseline_prompt(
        scenario,
        pedagogical=True,
        include_student_state=include_student_state,
    )
    selector_system_prompt, selector_user_prompt = build_act_selection_prompt(
        scenario,
        include_student_state=include_student_state,
    )
    judge_system_prompt, judge_user_prompt = build_judge_prompt(
        scenario,
        response=judge_response,
        include_student_state=include_student_state,
    )

    sections = [
        "# Prompt Snapshot",
        "",
        f"- Scenario path: `{scenario_path}`",
        f"- Scenario id: `{scenario.scenario_id}`",
        f"- Difficulty: `{scenario.difficulty}`",
        f"- Topic: `{scenario.topic}`",
        f"- Student state visible to tutors: `{'yes' if include_student_state else 'no'}`",
        f"- Act-conditioned response prompt rendered for: `{'all acts' if all_acts else selected_act}`",
        f"- Judge response placeholder: `{judge_response}`",
        "",
        render_prompt_section("Direct Baseline", direct_system_prompt, direct_user_prompt),
        render_prompt_section(
            "Pedagogical Baseline",
            pedagogical_system_prompt,
            pedagogical_user_prompt,
        ),
        render_prompt_section(
            "Act Selector",
            selector_system_prompt,
            selector_user_prompt,
        ),
        render_prompt_section(
            "Judge Prompt",
            judge_system_prompt,
            judge_user_prompt,
        ),
    ]
    sections.extend(
        render_act_response_sections(
            scenario,
            include_student_state=include_student_state,
            selected_act=selected_act,
            all_acts=all_acts,
        )
    )
    return "\n".join(sections).strip() + "\n"


def resolve_output_path(explicit_output: str, include_student_state: bool) -> Path:
    if explicit_output:
        return Path(explicit_output)
    suffix = "" if include_student_state else "_no_state"
    return Path(f"outputs/_prompt_snapshot{suffix}.md")


def main() -> None:
    args = parse_args()
    config = load_config()
    scenario_path = args.scenario_path or config.scenario_path
    include_student_state = (
        config.include_student_state
        if args.include_student_state is None
        else args.include_student_state
    )

    scenarios = load_scenarios(scenario_path)
    scenario = select_scenario(scenarios, args.scenario_id)
    selected_act = args.selected_act or scenario.ideal_move

    markdown = build_markdown(
        scenario,
        include_student_state=include_student_state,
        selected_act=selected_act,
        all_acts=args.all_acts,
        scenario_path=scenario_path,
        judge_response=args.judge_response,
    )

    output_path = resolve_output_path(args.output, include_student_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote prompt snapshot to {output_path}")


if __name__ == "__main__":
    main()
