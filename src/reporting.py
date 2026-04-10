from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

SYSTEM_ORDER = ["direct", "pedagogical", "act_conditioned"]
DIFFICULTY_ORDER = ["easy", "medium", "hard", "deep"]


def normalize_leakage(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"none", "decisive_step", "full_answer"}:
        return text
    if text in {"0", ""}:
        return "none"
    if text == "1":
        return "decisive_step"
    return text


def preview(text: str | None, limit: int = 120) -> str:
    if not text:
        return ""
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {}

    leakage_counts = Counter(normalize_leakage(item["judgment"].get("leakage", "none")) for item in records)
    leakage_rate = (
        (leakage_counts.get("decisive_step", 0) + leakage_counts.get("full_answer", 0)) / len(records)
    )
    decisive_step_rate = leakage_counts.get("decisive_step", 0) / len(records)
    full_answer_rate = leakage_counts.get("full_answer", 0) / len(records)
    pedagogy_raw_mean = sum(
        item["judgment"].get("pedagogy_raw_mean", item["judgment"].get("pedagogy_mean", 0.0))
        for item in records
    ) / len(records)
    pedagogy_mean = sum(item["judgment"]["pedagogy_mean"] for item in records) / len(records)
    correctness_mean = sum(item["judgment"]["correctness"] for item in records) / len(records)
    act_counts = Counter(
        item["selected_act"] for item in records if item.get("selected_act") is not None
    )

    selected_records = [item for item in records if item.get("selected_act") is not None]
    act_match_rate = None
    if selected_records:
        matches = sum(item["selected_act"] == item.get("ideal_move") for item in selected_records)
        act_match_rate = matches / len(selected_records)

    return {
        "count": len(records),
        "leakage_rate": leakage_rate,
        "decisive_step_rate": decisive_step_rate,
        "full_answer_rate": full_answer_rate,
        "leakage_counts": dict(leakage_counts),
        "pedagogy_raw_mean": pedagogy_raw_mean,
        "pedagogy_mean": pedagogy_mean,
        "correctness_mean": correctness_mean,
        "act_counts": dict(act_counts),
        "act_match_rate": act_match_rate,
        "difficulty_counts": dict(Counter(item.get("difficulty", "unspecified") for item in records)),
    }


def summarize_by_system(records: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["system"]].append(record)
    return {system: aggregate(system_records) for system, system_records in grouped.items()}


def summarize_by_system_and_difficulty(records: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        difficulty = str(record.get("difficulty", "unspecified"))
        grouped[difficulty][record["system"]].append(record)
    return {
        difficulty: {
            system_name: aggregate(system_records)
            for system_name, system_records in system_groups.items()
        }
        for difficulty, system_groups in grouped.items()
    }


def ordered_system_names(system_names: list[str] | set[str] | dict[str, dict]) -> list[str]:
    names = list(system_names)
    priority = {name: index for index, name in enumerate(SYSTEM_ORDER)}
    return sorted(names, key=lambda name: (priority.get(name, len(priority)), name))


def ordered_difficulty_names(
    difficulty_names: list[str] | set[str] | dict[str, dict],
) -> list[str]:
    names = list(difficulty_names)
    priority = {name: index for index, name in enumerate(DIFFICULTY_ORDER)}
    return sorted(names, key=lambda name: (priority.get(name, len(priority)), name))


def system_summary_lines(system_summaries: dict[str, dict]) -> list[str]:
    lines: list[str] = []
    for system_name in ordered_system_names(system_summaries):
        summary = system_summaries[system_name]
        act_match = summary.get("act_match_rate")
        act_match_text = f"{act_match:.2%}" if act_match is not None else "-"
        lines.append(
            f"{system_name:<15} n={summary['count']:<3} "
            f"leaks(any/step/full)={summary['leakage_rate']:.2%}/"
            f"{summary['decisive_step_rate']:.2%}/{summary['full_answer_rate']:.2%} "
            f"pedagogy(raw/capped)={summary['pedagogy_raw_mean']:.2f}/{summary['pedagogy_mean']:.2f} "
            f"correctness={summary['correctness_mean']:.2f} "
            f"act_match={act_match_text}"
        )
    return lines


def trial_key(record: dict) -> tuple[str, int, str]:
    return (
        str(record.get("run_id", "latest")),
        int(record.get("repetition_index", 1)),
        str(record.get("scenario_id", "")),
    )


def group_by_trial(records: list[dict]) -> dict[tuple[str, int, str], dict[str, dict]]:
    grouped: dict[tuple[str, int, str], dict[str, dict]] = defaultdict(dict)
    for record in records:
        grouped[trial_key(record)][record["system"]] = record
    return grouped


def compare_systems_by_scenario(records: list[dict], baseline: str, candidate: str) -> list[dict]:
    comparisons: list[dict] = []
    for (run_id, repetition_index, scenario_id), system_records in group_by_trial(records).items():
        if baseline not in system_records or candidate not in system_records:
            continue
        base_record = system_records[baseline]
        candidate_record = system_records[candidate]
        delta = (
            candidate_record["judgment"]["pedagogy_mean"] - base_record["judgment"]["pedagogy_mean"]
        )
        comparisons.append(
            {
                "run_id": run_id,
                "repetition_index": repetition_index,
                "scenario_id": scenario_id,
                "delta": delta,
                "baseline": base_record,
                "candidate": candidate_record,
            }
        )
    comparisons.sort(key=lambda item: item["delta"], reverse=True)
    return comparisons


def select_examples(
    records: list[dict],
    *,
    system_name: str,
    top_k: int = 3,
) -> dict[str, list[dict]]:
    filtered = [record for record in records if record["system"] == system_name]
    ordered = sorted(filtered, key=lambda item: item["judgment"]["pedagogy_mean"], reverse=True)
    leakage_cases = [
        record
        for record in filtered
        if normalize_leakage(record["judgment"].get("leakage", "none")) != "none"
    ]
    return {
        "best": ordered[:top_k],
        "worst": list(reversed(ordered[-top_k:])) if ordered else [],
        "leakage": leakage_cases[:top_k],
    }


def render_example(record: dict) -> str:
    response = preview(record.get("response"), 220)
    problem = preview(record.get("problem"), 160)
    student_attempt = preview(record.get("student_attempt"), 160)
    selected_act = record.get("selected_act") or "-"
    run_id = record.get("run_id", "latest")
    repetition_index = record.get("repetition_index", 1)
    return (
        f"- `{record['scenario_id']}` "
        f"(run={run_id}, rep={repetition_index}, "
        f"difficulty={record.get('difficulty')}, topic={record.get('topic')}, "
        f"state={record.get('student_state')}, ideal={record.get('ideal_move')}, "
        f"selected={selected_act}, pedagogy_raw={record['judgment'].get('pedagogy_raw_mean', record['judgment']['pedagogy_mean']):.2f}, "
        f"pedagogy_capped={record['judgment']['pedagogy_mean']:.2f}, "
        f"leakage={normalize_leakage(record['judgment'].get('leakage', 'none'))})\n"
        f"  problem: {problem}\n"
        f"  student: {student_attempt}\n"
        f"  response: {response}\n"
        f"  judge: {preview(record['judgment'].get('summary'), 180)}"
    )


def score_bundle_text(judgment: dict) -> str:
    pedagogy_raw_mean = judgment.get("pedagogy_raw_mean", "")
    scaffolding = judgment.get("scaffolding", "")
    scaffolding_capped = judgment.get("scaffolding_capped", "")
    self_correction_support = judgment.get("self_correction_support", "")
    self_correction_support_capped = judgment.get("self_correction_support_capped", "")
    scaffolding_text = (
        f"{scaffolding}->{scaffolding_capped}"
        if scaffolding != "" and scaffolding_capped != "" and scaffolding != scaffolding_capped
        else str(scaffolding)
    )
    self_correction_text = (
        f"{self_correction_support}->{self_correction_support_capped}"
        if self_correction_support != ""
        and self_correction_support_capped != ""
        and self_correction_support != self_correction_support_capped
        else str(self_correction_support)
    )
    return (
        f"leakage={judgment.get('leakage', '')}; "
        f"correctness={judgment.get('correctness', '')}; "
        f"scaffolding={scaffolding_text}; "
        f"self_correction_support={self_correction_text}; "
        f"non_overload={judgment.get('non_overload', '')}; "
        f"tone={judgment.get('tone', '')}; "
        f"pedagogy_raw_mean={pedagogy_raw_mean}; "
        f"pedagogy_mean={judgment.get('pedagogy_mean', '')}"
    )


def table_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        judgment = record.get("judgment", {})
        rows.append(
            {
                "run_id": record.get("run_id", ""),
                "run_label": record.get("run_label", ""),
                "repetition_index": record.get("repetition_index", ""),
                "timestamp": record.get("timestamp", ""),
                "generation_model": record.get("generation_model", ""),
                "act_selection_model": record.get("act_selection_model", ""),
                "judge_model": record.get("judge_model", ""),
                "scenario_id": record.get("scenario_id", ""),
                "system": record.get("system", ""),
                "difficulty": record.get("difficulty", ""),
                "topic": record.get("topic", ""),
                "student_state": record.get("student_state", ""),
                "ideal_move": record.get("ideal_move", ""),
                "selected_act": record.get("selected_act") or "",
                "problem": record.get("problem", ""),
                "student_response": record.get("student_attempt", ""),
                "teacher_response": record.get("response", ""),
                "judge_reasoning": judgment.get("reasoning", judgment.get("summary", "")),
                "judge_summary": judgment.get("summary", ""),
                "leakage": normalize_leakage(judgment.get("leakage", "")),
                "correctness": judgment.get("correctness", ""),
                "scaffolding": judgment.get("scaffolding", ""),
                "scaffolding_capped": judgment.get("scaffolding_capped", ""),
                "self_correction_support": judgment.get("self_correction_support", ""),
                "self_correction_support_capped": judgment.get("self_correction_support_capped", ""),
                "non_overload": judgment.get("non_overload", ""),
                "tone": judgment.get("tone", ""),
                "pedagogy_raw_mean": judgment.get("pedagogy_raw_mean", ""),
                "pedagogy_mean": judgment.get("pedagogy_mean", ""),
            }
        )
    return rows


def _compact_cell(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _markdown_escape(value: object) -> str:
    text = _compact_cell(value)
    return text.replace("|", "\\|")


def write_csv_table(records: list[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    rows = table_rows(records)
    fieldnames = [
        "run_id",
        "run_label",
        "repetition_index",
        "timestamp",
        "generation_model",
        "act_selection_model",
        "judge_model",
        "scenario_id",
        "system",
        "difficulty",
        "topic",
        "student_state",
        "ideal_move",
        "selected_act",
        "problem",
        "student_response",
        "teacher_response",
        "judge_reasoning",
        "judge_summary",
        "leakage",
        "correctness",
        "scaffolding",
        "scaffolding_capped",
        "self_correction_support",
        "self_correction_support_capped",
        "non_overload",
        "tone",
        "pedagogy_raw_mean",
        "pedagogy_mean",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _compact_cell(row.get(key, "")) for key in fieldnames})
    return output_path


def difficulty_summary_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    by_difficulty = summarize_by_system_and_difficulty(records)
    for difficulty in ordered_difficulty_names(by_difficulty):
        for system_name in ordered_system_names(by_difficulty[difficulty]):
            summary = by_difficulty[difficulty][system_name]
            act_match = summary.get("act_match_rate")
            rows.append(
                {
                    "difficulty": difficulty,
                    "system": system_name,
                    "count": summary["count"],
                    "leakage_rate": summary["leakage_rate"],
                    "decisive_step_rate": summary["decisive_step_rate"],
                    "full_answer_rate": summary["full_answer_rate"],
                    "pedagogy_raw_mean": summary["pedagogy_raw_mean"],
                    "pedagogy_mean": summary["pedagogy_mean"],
                    "correctness_mean": summary["correctness_mean"],
                    "act_match_rate": act_match if act_match is not None else "",
                }
            )
    return rows


def write_difficulty_summary_csv(records: list[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    rows = difficulty_summary_rows(records)
    fieldnames = [
        "difficulty",
        "system",
        "count",
        "leakage_rate",
        "decisive_step_rate",
        "full_answer_rate",
        "pedagogy_raw_mean",
        "pedagogy_mean",
        "correctness_mean",
        "act_match_rate",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def write_markdown_table(records: list[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    rows = table_rows(records)
    headers = [
        "run_id",
        "run_label",
        "rep",
        "timestamp",
        "generation_model",
        "act_selection_model",
        "judge_model",
        "scenario_id",
        "system",
        "difficulty",
        "topic",
        "student_state",
        "ideal_move",
        "selected_act",
        "problem",
        "student_response",
        "teacher_response",
        "judge_reasoning",
        "judge_scores",
    ]
    lines = [
        "# Scenario Results Table",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        judgment = {
            "leakage": row["leakage"],
            "correctness": row["correctness"],
            "scaffolding": row["scaffolding"],
            "scaffolding_capped": row["scaffolding_capped"],
            "self_correction_support": row["self_correction_support"],
            "self_correction_support_capped": row["self_correction_support_capped"],
            "non_overload": row["non_overload"],
            "tone": row["tone"],
            "pedagogy_raw_mean": row["pedagogy_raw_mean"],
            "pedagogy_mean": row["pedagogy_mean"],
        }
        cells = [
            row["run_id"],
            row["run_label"],
            row["repetition_index"],
            row["timestamp"],
            row["generation_model"],
            row["act_selection_model"],
            row["judge_model"],
            row["scenario_id"],
            row["system"],
            row["difficulty"],
            row["topic"],
            row["student_state"],
            row["ideal_move"],
            row["selected_act"],
            row["problem"],
            row["student_response"],
            row["teacher_response"],
            row["judge_reasoning"],
            score_bundle_text(judgment),
        ]
        lines.append("| " + " | ".join(_markdown_escape(cell) for cell in cells) + " |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path


def write_markdown_report(records: list[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    summaries = summarize_by_system(records)
    difficulty_summaries = summarize_by_system_and_difficulty(records)
    example_pack = select_examples(records, system_name="act_conditioned", top_k=3)
    comparisons = compare_systems_by_scenario(records, baseline="pedagogical", candidate="act_conditioned")

    lines: list[str] = ["# Evaluation Report", ""]
    lines.append("## System Summary")
    lines.append("")
    lines.append("| System | N | Leak Any | Leak Step | Leak Full | Ped Raw | Ped Capped | Correctness | Act Match |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for system_name in ordered_system_names(summaries):
        summary = summaries[system_name]
        act_match = summary.get("act_match_rate")
        act_match_text = f"{act_match:.2%}" if act_match is not None else "-"
        lines.append(
            f"| {system_name} | {summary['count']} | {summary['leakage_rate']:.2%} | "
            f"{summary['decisive_step_rate']:.2%} | {summary['full_answer_rate']:.2%} | "
            f"{summary['pedagogy_raw_mean']:.2f} | {summary['pedagogy_mean']:.2f} | "
            f"{summary['correctness_mean']:.2f} | {act_match_text} |"
        )
    lines.append("")
    lines.append("## System Summary by Difficulty")
    lines.append("")
    lines.append("| Difficulty | System | N | Leak Any | Leak Step | Leak Full | Ped Raw | Ped Capped | Correctness | Act Match |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for difficulty in ordered_difficulty_names(difficulty_summaries):
        for system_name in ordered_system_names(difficulty_summaries[difficulty]):
            summary = difficulty_summaries[difficulty][system_name]
            act_match = summary.get("act_match_rate")
            act_match_text = f"{act_match:.2%}" if act_match is not None else "-"
            lines.append(
                f"| {difficulty} | {system_name} | {summary['count']} | "
                f"{summary['leakage_rate']:.2%} | {summary['decisive_step_rate']:.2%} | "
                f"{summary['full_answer_rate']:.2%} | {summary['pedagogy_raw_mean']:.2f} | "
                f"{summary['pedagogy_mean']:.2f} | {summary['correctness_mean']:.2f} | {act_match_text} |"
            )
    lines.append("")
    scenario_index: dict[str, dict] = {}
    for record in records:
        scenario_index.setdefault(record.get("scenario_id", ""), record)
    dataset_difficulties = Counter(
        record.get("difficulty", "unspecified") for record in scenario_index.values()
    )
    lines.append("## Dataset Composition")
    lines.append("")
    lines.append(
        "- Difficulty counts: "
        + ", ".join(
            f"{name}={dataset_difficulties[name]}"
            for name in ordered_difficulty_names(dataset_difficulties)
        )
    )
    run_ids = sorted({record.get("run_id", "latest") for record in records})
    repetition_count = len(
        {
            (record.get("run_id", "latest"), int(record.get("repetition_index", 1)))
            for record in records
        }
    )
    lines.append(f"- Runs represented: {len(run_ids)}")
    lines.append(f"- Trial repetitions represented: {repetition_count}")
    lines.append("")

    if comparisons:
        lines.append("## Act-Conditioned vs Pedagogical Baseline")
        lines.append("")
        for item in comparisons[:5]:
            candidate = item["candidate"]
            baseline = item["baseline"]
            lines.append(
                f"- `{item['scenario_id']}` run={item['run_id']} rep={item['repetition_index']} "
                f"delta={item['delta']:+.2f} "
                f"(act_conditioned={candidate['judgment']['pedagogy_mean']:.2f}, "
                f"pedagogical={baseline['judgment']['pedagogy_mean']:.2f})"
            )
        lines.append("")

    lines.append("## Best Act-Conditioned Examples")
    lines.append("")
    for record in example_pack["best"]:
        lines.append(render_example(record))
        lines.append("")

    lines.append("## Worst Act-Conditioned Examples")
    lines.append("")
    for record in example_pack["worst"]:
        lines.append(render_example(record))
        lines.append("")

    if example_pack["leakage"]:
        lines.append("## Leakage Cases")
        lines.append("")
        for record in example_pack["leakage"]:
            lines.append(render_example(record))
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path


def coverage_rows(records: list[dict]) -> list[dict]:
    counts: dict[str, Counter] = defaultdict(Counter)
    metadata: dict[str, dict] = {}
    for record in records:
        scenario_id = record.get("scenario_id", "")
        counts[scenario_id][record.get("system", "")] += 1
        metadata.setdefault(
            scenario_id,
            {
                "difficulty": record.get("difficulty", ""),
                "topic": record.get("topic", ""),
            },
        )

    rows: list[dict] = []
    for scenario_id in sorted(counts):
        row = {
            "scenario_id": scenario_id,
            "difficulty": metadata.get(scenario_id, {}).get("difficulty", ""),
            "topic": metadata.get(scenario_id, {}).get("topic", ""),
        }
        total = 0
        for system_name in SYSTEM_ORDER:
            value = counts[scenario_id].get(system_name, 0)
            row[f"{system_name}_count"] = value
            total += value
        row["total_records"] = total
        rows.append(row)
    return rows


def write_coverage_csv(records: list[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    rows = coverage_rows(records)
    fieldnames = [
        "scenario_id",
        "difficulty",
        "topic",
        "direct_count",
        "pedagogical_count",
        "act_conditioned_count",
        "total_records",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path
