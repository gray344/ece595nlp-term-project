from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Scenario:
    scenario_id: str
    problem: str
    correct_answer: str
    difficulty: str
    topic: str
    student_state: str
    student_attempt: str
    dialogue_context: list[str]
    ideal_move: str


def load_scenarios(path: str) -> list[Scenario]:
    scenarios: list[Scenario] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            scenarios.append(
                Scenario(
                    scenario_id=record["scenario_id"],
                    problem=record["problem"],
                    correct_answer=str(record["correct_answer"]),
                    difficulty=record.get("difficulty", "unspecified"),
                    topic=record.get("topic", "unspecified"),
                    student_state=record["student_state"],
                    student_attempt=record["student_attempt"],
                    dialogue_context=list(record.get("dialogue_context", [])),
                    ideal_move=record["ideal_move"],
                )
            )
    return scenarios


def save_jsonl(path: str, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def scenario_to_dict(scenario: Scenario) -> dict:
    return asdict(scenario)
