from __future__ import annotations

from textwrap import dedent
from typing import Literal

from pydantic import BaseModel

from src.data import Scenario


TUTOR_ACTS: dict[str, str] = {
    "ask_subproblem": "Ask a smaller leading question that helps the student figure out the next step.",
    "give_hint": "Provide a partial hint without revealing the full answer or full next step.",
    "locate_error": "Point out the specific mistake or incorrect assumption in the student's work.",
    "prompt_self_correction": "Prompt the student to revise or fix their own reasoning.",
    "explain_concept": "Briefly explain the concept, rule, or formula needed for the problem.",
    "check_understanding": "Ask the student to justify, verify, or restate their reasoning.",
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
    Prefer hints, leading questions, and support for self-correction.
    Avoid giving away the final answer or a decisive next step unless it is truly necessary.
    Keep the help targeted to the student's current mistake or uncertainty.
    """
).strip()

JUDGE_RUBRIC = dedent(
    """
    Evaluation priorities:
    - Judge the actual tutor response, not whether the selected act label sounds good.
    - Prefer responses that move the student forward without simply solving the problem for them.
    - Reward targeted, step-sized help over long explanations or full solutions.
    - Do not reward friendly tone if the math guidance is wrong or misleading.
    - Use the full 1-5 scale. Reserve 5 for clearly strong tutoring behavior and 1 for clearly poor behavior.

    Leakage:
    - 0: The response does not reveal the final answer or a decisive next step that removes the student's work.
    - 1: The response gives away the final answer, most of the solution, or a decisive next step that substantially solves the problem for the student.

    Correctness (1-5):
    - 5: Mathematically correct, instructionally sound, and aligned with the student's current work.
    - 3: Mostly correct but somewhat ambiguous, incomplete, or mildly misleading.
    - 1: Mathematically wrong, clearly misleading, or incompatible with the student's situation.

    Scaffolding (1-5):
    - 5: Gives a focused hint, question, or micro-step that helps the student make the next move themselves.
    - 3: Provides some guidance, but is either too vague to help or too direct to count as strong scaffolding.
    - 1: Mostly solves the problem, jumps too far ahead, or provides no useful guidance.

    Self-correction support (1-5):
    - 5: Explicitly helps the student inspect, revise, or repair their own reasoning.
    - 3: Encourages checking or revision, but without enough specificity to strongly support self-correction.
    - 1: Does not help the student reflect on their work or replaces their reasoning entirely.

    Non-overload (1-5):
    - 5: Concise, focused, and easy to act on in one step.
    - 3: Somewhat wordy, dense, or split across too many ideas.
    - 1: Overly long, cognitively heavy, or likely to overwhelm the student.

    Tone (1-5):
    - 5: Supportive, respectful, and clear without being patronizing.
    - 3: Neutral or uneven, but not harmful.
    - 1: Dismissive, harsh, confusing, or discouraging.

    In the reasoning field, briefly justify each score with concrete evidence from the tutor response.
    """
).strip()


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


def render_act_inventory() -> str:
    return "\n".join(f"- {name}: {description}" for name, description in TUTOR_ACTS.items())


def build_baseline_prompt(scenario: Scenario, pedagogical: bool) -> tuple[str, str]:
    system_prompt = BASE_TUTOR_ROLE
    if pedagogical:
        system_prompt += "\n" + PEDAGOGICAL_POLICY

    user_prompt = dedent(
        f"""
        Problem:
        {scenario.problem}

        Student state: {scenario.student_state}

        Student attempt:
        {scenario.student_attempt}

        Dialogue context:
        {format_context(scenario.dialogue_context)}

        Write the next tutor response only.
        """
    ).strip()
    return system_prompt, user_prompt


def build_act_selection_prompt(scenario: Scenario) -> tuple[str, str]:
    system_prompt = dedent(
        """
        You are selecting the next pedagogical move for a math tutor.
        Choose exactly one act from the provided inventory.
        Prefer moves that help the student think, self-correct, and take the next step themselves.
        Avoid selecting give_direct_answer unless stronger scaffolding is no longer appropriate.
        Return valid JSON with keys:
        - selected_act
        - rationale
        """
    ).strip()

    user_prompt = dedent(
        f"""
        Tutor act inventory:
        {render_act_inventory()}

        Problem:
        {scenario.problem}

        Student state: {scenario.student_state}

        Student attempt:
        {scenario.student_attempt}

        Dialogue context:
        {format_context(scenario.dialogue_context)}

        Choose the best next tutoring act.
        """
    ).strip()
    return system_prompt, user_prompt


def build_response_prompt(scenario: Scenario, selected_act: str) -> tuple[str, str]:
    act_description = TUTOR_ACTS[selected_act]
    system_prompt = dedent(
        f"""
        {BASE_TUTOR_ROLE}
        {PEDAGOGICAL_POLICY}

        Your assigned pedagogical move is: {selected_act}
        Act definition: {act_description}

        Follow the assigned act closely.
        Keep the response concise and targeted.
        If the assigned act is not give_direct_answer, do not reveal the final answer or a decisive next step.
        Do not mention the act label in your answer.
        """
    ).strip()

    user_prompt = dedent(
        f"""
        Problem:
        {scenario.problem}

        Student state: {scenario.student_state}

        Student attempt:
        {scenario.student_attempt}

        Dialogue context:
        {format_context(scenario.dialogue_context)}

        Write the next tutor response only.
        """
    ).strip()
    return system_prompt, user_prompt


def build_judge_prompt(scenario: Scenario, response: str) -> tuple[str, str]:
    system_prompt = dedent(
        """
        You are grading a tutoring response for pedagogical quality.
        Apply the rubric strictly and consistently.
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
