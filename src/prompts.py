from __future__ import annotations

from textwrap import dedent
from typing import Literal

from pydantic import BaseModel

from src.data import Scenario


TUTOR_ACTS: dict[str, str] = {
    "ask_subproblem": "Ask one smaller leading question about a prerequisite relationship, quantity, or micro-step.",
    "give_hint": "Provide one concrete conceptual cue or partial setup without instantiating the decisive next step.",
    "locate_error": "Point out the specific mistaken step or assumption in the student's current work without replacing it with the corrected step.",
    "prompt_self_correction": "Ask the student to recheck, revise, or fix their own reasoning.",
    "explain_concept": "Briefly explain the needed concept, rule, or formula, then let the student apply it.",
    "check_understanding": "Ask the student to justify, verify, or restate reasoning they already have.",
    "give_direct_answer": "Provide the solution or a decisive next step directly; use only when stronger scaffolding is no longer appropriate.",
}

ScoreValue = Literal[1, 2, 3, 4, 5]

BASE_TUTOR_ROLE = dedent(
    """
    You are a math tutor responding to a student.
    Be accurate, concise, and easy to understand.
    """
).strip()

PEDAGOGICAL_POLICY = dedent(
    """
    Guide the student instead of simply solving the problem.
    Prefer the smallest useful intervention that helps the student make the next move.
    Prefer hints, leading questions, error diagnosis, and support for self-correction.
    Keep the help targeted to the student's current mistake or uncertainty and limited to one next-step idea.
    """
).strip()

PEDAGOGICAL_RESPONSE_SHAPE = dedent(
    """
    Default response shape:
    - 1 to 3 short sentences.
    - Ask at most 2 questions.
    - Do not write a full worked solution for this problem.
    """
).strip()

SHARED_LEAKAGE_POLICY_RULES = (
    "Do not reveal the final answer.",
    "Do not write a fully instantiated equation or a full calculation using this problem's numbers.",
    "If the student's unresolved obstacle is choosing an operation or next step, do not name that operation directly for ask_subproblem, give_hint, locate_error, prompt_self_correction, or check_understanding. Ask about the relationship, goal, missing quantity, or quantity change instead.",
    "For explain_concept, you may state a general rule or formula in abstract terms or with placeholders, but do not immediately plug in this problem's numbers or carry out the decisive next calculation.",
)


def render_leakage_policy() -> str:
    return "\n".join(f"- {rule}" for rule in SHARED_LEAKAGE_POLICY_RULES)


def render_state_guidance(include_student_state: bool) -> str:
    if include_student_state:
        return (
            "A student_state label will be provided. Use it as a strong cue, "
            "but stay grounded in the actual student attempt and dialogue context."
        )
    return (
        "No student_state label will be provided. Infer the student's current barrier "
        "from the student attempt and dialogue context."
    )


def build_act_selection_guidance(include_student_state: bool) -> str:
    state_guidance = (
        "- A student_state label is provided; use it as a strong cue, but confirm it against the attempt and dialogue context."
        if include_student_state
        else "- No student_state label is provided. Infer whether the student is mainly stuck, wrong_step, or partially_correct from the attempt and dialogue context before choosing an act."
    )
    return dedent(
        f"""
        Selection guidance:
        {state_guidance}
        - Start with the least revealing move that can resolve the student's current barrier.
        - If the student is or seems wrong_step, prefer locate_error or prompt_self_correction; use explain_concept only when the error comes from a missing rule or formula.
        - If the student is or seems stuck, prefer ask_subproblem or give_hint; use explain_concept only when the student explicitly lacks the underlying concept.
        - If the student is or seems partially_correct, prefer prompt_self_correction or check_understanding.
        - Choose give_direct_answer only when a non-revealing move would no longer be useful.

        Act distinctions:
        - ask_subproblem: ask one smaller question about a prerequisite relationship, quantity, or micro-step; if the unresolved obstacle is the operation choice, do not name the operation.
        - give_hint: give one concrete cue or partial setup that keeps meaningful work with the student; do not instantiate or name the decisive next step when that choice is the barrier.
        - locate_error: identify the exact mistaken step or assumption, then redirect the student to repair it; do not replace it with the corrected numeric step.
        - prompt_self_correction: explicitly ask the student to revise, retry, or check their own work without supplying the repaired step.
        - explain_concept: explain the needed rule or formula briefly in general terms, then ask the student to apply it.
        - check_understanding: ask the student to justify, verify, or restate reasoning they already have; do not introduce a new solution path.
        - give_direct_answer: provide the solution or decisive next step directly.
        """
    ).strip()


ACT_REALIZATION_RULES: dict[str, str] = {
    "ask_subproblem": dedent(
        """
        Ask one tightly focused smaller question about a prerequisite relationship, quantity, or micro-step.
        You may ask one brief follow-up only if it is part of the same micro-step.
        If the student's unresolved obstacle is choosing an operation, do not name that operation.
        Do not write the full equation or calculation.
        """
    ).strip(),
    "give_hint": dedent(
        """
        Give one nudge, cue, representation, or partial setup that keeps meaningful work with the student.
        If the student's unresolved obstacle is choosing an operation, do not name that operation directly.
        Do not write the decisive equation or calculation with the problem's numbers.
        """
    ).strip(),
    "locate_error": dedent(
        """
        Name the specific mistaken step, omission, or assumption in the student's current work.
        After naming the error, redirect the student to repair that step themselves.
        Do not replace the student's step with the corrected calculation or corrected numeric step.
        """
    ).strip(),
    "prompt_self_correction": dedent(
        """
        Explicitly ask the student to recheck or revise their own prior step or answer.
        Reference what they should inspect, but do not provide the corrected step.
        If the student's unresolved obstacle is choosing an operation, do not name that operation directly.
        """
    ).strip(),
    "explain_concept": dedent(
        """
        Briefly explain the needed rule or formula in general language or with placeholders when possible.
        After the explanation, ask the student to apply it to this problem.
        Avoid plugging in the problem's numbers in sequence or carrying out the decisive next calculation.
        """
    ).strip(),
    "check_understanding": dedent(
        """
        Ask the student to justify, verify, or restate reasoning they already have.
        Do not introduce a new method unless the current reasoning is unsalvageable.
        If the student's unresolved obstacle is choosing an operation, do not name that operation directly.
        """
    ).strip(),
    "give_direct_answer": dedent(
        """
        Provide the answer or decisive next step clearly.
        Keep it brief and include only the minimum explanation needed to stay understandable.
        """
    ).strip(),
}

JUDGE_RUBRIC = dedent(
    """
    Evaluation priorities:
    - Judge the actual tutor response, not whether the selected act label sounds good.
    - Judge only the visible response, not the tutor's possible intentions.
    - Prefer responses that move the student forward without simply solving the problem for them.
    - Reward targeted, step-sized help over long explanations or full solutions.
    - A response that leaks the answer or the decisive next calculation is pedagogically limited even if the math is correct.
    - Do not reward friendly tone if the math guidance is wrong, overly revealing, or misaligned with the student's current barrier.
    - Use the full 1-5 scale. Reserve 5 for clearly strong tutoring behavior and 1 for clearly poor behavior.

    Shared leakage policy:
    {leakage_policy}

    Leakage:
    - 0: The response follows the shared leakage policy and preserves meaningful work for the student.
    - 1: The response violates the shared leakage policy and removes the need for the student to determine the next substantive step.

    Scoring constraints when leakage = 1:
    - Scaffolding cannot exceed 3.
    - Self-correction support cannot exceed 3.
    - The response should not be treated as pedagogically excellent overall.

    Correctness (1-5):
    - 5: Mathematically correct, instructionally sound, and aligned with the student's current work.
    - 3: Mostly correct but somewhat ambiguous, incomplete, or mildly misleading.
    - 1: Mathematically wrong, clearly misleading, or incompatible with the student's situation.

    Scaffolding (1-5):
    - 5: Preserves meaningful work for the student and gives a focused hint, question, or micro-step that helps them make the next move themselves.
    - 3: Provides some guidance, but is either too vague to help or somewhat too revealing.
    - 1: Mostly solves the problem, jumps too far ahead, or provides no useful guidance.

    Self-correction support (1-5):
    - 5: Explicitly helps the student inspect, revise, or repair their own reasoning.
    - 3: Encourages checking or revision, but without enough specificity to strongly support self-correction, or replaces too much of the student's reasoning.
    - 1: Does not help the student reflect on their work or fully replaces their reasoning.

    Non-overload (1-5):
    - 5: Concise, focused, and easy to act on in one step.
    - 3: Somewhat wordy, dense, or split across too many ideas.
    - 1: Overly long, cognitively heavy, or likely to overwhelm the student.

    Tone (1-5):
    - 5: Supportive, respectful, and clear without being patronizing.
    - 3: Neutral or uneven, but not harmful.
    - 1: Dismissive, harsh, confusing, or discouraging.

    In the reasoning field, write 3 to 6 short sentences total.
    Keep the full reasoning under 120 words.
    Briefly justify the important scores with concrete evidence from the tutor response and explicitly mention any leakage trigger.
    Keep the summary to one short sentence.
    """
).strip().format(leakage_policy=render_leakage_policy())


class ActSelectionOutput(BaseModel):
    selected_act: Literal[
        "ask_subproblem",
        "give_hint",
        "locate_error",
        "prompt_self_correction",
        "explain_concept",
        "check_understanding",
        "give_direct_answer",
    ]
    rationale: str


class JudgeResponseOutput(BaseModel):
    leakage: Literal[0, 1]
    correctness: ScoreValue
    scaffolding: ScoreValue
    self_correction_support: ScoreValue
    non_overload: ScoreValue
    tone: ScoreValue
    reasoning: str
    summary: str


def format_context(dialogue_context: list[str]) -> str:
    if not dialogue_context:
        return "No prior dialogue."
    return "\n".join(dialogue_context)


def render_scenario_input(scenario: Scenario, include_student_state: bool) -> str:
    lines = ["Problem:", scenario.problem]
    if include_student_state:
        lines.extend(["", f"Student state: {scenario.student_state}"])
    lines.extend(
        [
            "",
            "Student attempt:",
            scenario.student_attempt,
            "",
            "Dialogue context:",
            format_context(scenario.dialogue_context),
        ]
    )
    return "\n".join(lines)


def render_act_inventory() -> str:
    return "\n".join(f"- {name}: {description}" for name, description in TUTOR_ACTS.items())


def build_baseline_prompt(
    scenario: Scenario,
    pedagogical: bool,
    include_student_state: bool = True,
) -> tuple[str, str]:
    system_parts = [
        BASE_TUTOR_ROLE,
        render_state_guidance(include_student_state),
        "Shared leakage policy:\n" + render_leakage_policy(),
    ]
    if pedagogical:
        system_parts.extend([PEDAGOGICAL_POLICY, PEDAGOGICAL_RESPONSE_SHAPE])

    system_prompt = "\n\n".join(system_parts)
    user_prompt = f"{render_scenario_input(scenario, include_student_state)}\n\nWrite the next tutor response only."
    return system_prompt, user_prompt


def build_act_selection_prompt(
    scenario: Scenario,
    include_student_state: bool = True,
) -> tuple[str, str]:
    system_prompt = "\n\n".join(
        [
            dedent(
                """
                You are selecting the next pedagogical move for a math tutor.
                Choose exactly one act from the provided inventory.
                Prefer moves that help the student think, self-correct, and take the next step themselves.
                Avoid selecting give_direct_answer unless stronger scaffolding is no longer appropriate.
                Use the selection guidance strictly.
                Return valid JSON with keys:
                - selected_act
                - rationale
                """
            ).strip(),
            build_act_selection_guidance(include_student_state),
        ]
    )

    user_prompt = (
        f"Tutor act inventory:\n{render_act_inventory()}\n\n"
        f"{render_scenario_input(scenario, include_student_state)}\n\n"
        "Choose the single best next tutoring act for the next turn only."
    )
    return system_prompt, user_prompt


def build_response_prompt(
    scenario: Scenario,
    selected_act: str,
    include_student_state: bool = True,
) -> tuple[str, str]:
    act_description = TUTOR_ACTS[selected_act]
    act_rules = ACT_REALIZATION_RULES[selected_act]
    direct_answer_override = ""
    if selected_act == "give_direct_answer":
        direct_answer_override = (
            "\nBecause the assigned act is give_direct_answer, you may override the shared leakage "
            "policy and provide the answer or decisive next step directly if needed. This will still "
            "count as leakage under the evaluation rubric."
        )
    system_prompt = "\n\n".join(
        [
            BASE_TUTOR_ROLE,
            render_state_guidance(include_student_state),
            PEDAGOGICAL_POLICY,
            PEDAGOGICAL_RESPONSE_SHAPE,
            "Shared leakage policy:\n" + render_leakage_policy(),
            f"Your assigned pedagogical move is: {selected_act}\nAct definition: {act_description}",
            "Follow the assigned act closely.\n" + act_rules,
            "Keep the response concise and targeted.\nDo not mention the act label in your answer."
            + direct_answer_override,
        ]
    )

    user_prompt = f"{render_scenario_input(scenario, include_student_state)}\n\nWrite the next tutor response only."
    return system_prompt, user_prompt


def build_judge_prompt(scenario: Scenario, response: str) -> tuple[str, str]:
    system_prompt = dedent(
        """
        You are grading a tutoring response for pedagogical quality.
        Apply the rubric strictly and consistently.
        Enforce the leakage constraints in the rubric.
        Return valid JSON with keys:
        - leakage: 0 or 1
        - correctness: integer 1 to 5
        - scaffolding: integer 1 to 5
        - self_correction_support: integer 1 to 5
        - non_overload: integer 1 to 5
        - tone: integer 1 to 5
        - reasoning: concise but specific explanation of why each score was assigned
        - summary: short explanation
        """
    ).strip()

    user_prompt = dedent(
        f"""
        Problem:
        {scenario.problem}

        Correct answer:
        {scenario.correct_answer}

        Student state: {scenario.student_state}

        Student attempt:
        {scenario.student_attempt}

        Dialogue context:
        {format_context(scenario.dialogue_context)}

        Tutor response:
        {response}

        Rubric:
        {JUDGE_RUBRIC}

        Judge whether the response leaks too much of the answer and score its pedagogical quality.
        """
    ).strip()
    return system_prompt, user_prompt
