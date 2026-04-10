from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.act_selector import aselect_act
from src.baselines import agenerate_direct_baseline, agenerate_pedagogical_baseline
from src.config import load_config
from src.data import Scenario, append_jsonl, load_scenarios, save_jsonl
from src.judge import ajudge_response
from src.models.openai_client import OpenAITextClient
from src.reporting import (
    aggregate,
    load_jsonl,
    preview,
    summarize_by_system,
    system_summary_lines,
    write_coverage_csv,
    write_csv_table,
    write_difficulty_summary_csv,
    write_markdown_report,
    write_markdown_table,
)
from src.response_generator import agenerate_act_conditioned_response

SYSTEM_PRIORITY = {
    "direct": 0,
    "pedagogical": 1,
    "act_conditioned": 2,
}


class PartialRunError(RuntimeError):
    def __init__(
        self,
        *,
        system_name: str,
        repetition_index: int,
        partial_records: list[dict],
        original_error: BaseException,
    ) -> None:
        self.system_name = system_name
        self.repetition_index = repetition_index
        self.partial_records = partial_records
        self.original_error = original_error
        message = (
            f"{system_name} failed during repetition {repetition_index} after saving "
            f"{len(partial_records)} completed record(s). "
            f"Original error: {type(original_error).__name__}: {original_error}"
        )
        super().__init__(message)


def print_rule(char: str = "=", width: int = 80) -> None:
    print(char * width, flush=True)


def print_run_header(
    *,
    run_id: str,
    run_label: str,
    systems: list[str],
    scenario_count: int,
    repetitions: int,
    output_dir: Path,
    debug: bool,
    concurrency: int,
    include_student_state: bool,
) -> None:
    print_rule()
    print("Tutoring Evaluation Run", flush=True)
    print_rule()
    print(f"Run ID     : {run_id}", flush=True)
    print(f"Run label  : {run_label or '-'}", flush=True)
    print(f"Systems    : {', '.join(systems)}", flush=True)
    print(f"Scenarios  : {scenario_count}", flush=True)
    print(f"Repetitions: {repetitions}", flush=True)
    print(f"Total evals: {scenario_count * repetitions * len(systems)}", flush=True)
    print(f"Output dir : {output_dir}", flush=True)
    print(f"Concurrency: {concurrency}", flush=True)
    print(f"Debug      : {'on' if debug else 'off'}", flush=True)
    print(f"State input: {'on' if include_student_state else 'off'}", flush=True)
    print_rule()


def print_system_header(system_name: str, scenario_count: int) -> None:
    print_rule()
    print(f"System: {system_name} | scenarios: {scenario_count}", flush=True)
    print_rule()


def print_scenario_header(
    *,
    system_name: str,
    scenario: Scenario,
    index: int,
    total: int,
    repetition_index: int,
    repetitions: int,
) -> None:
    print(
        f"[{index}/{total}] {system_name} on {scenario.scenario_id} "
        f"(rep {repetition_index}/{repetitions})",
        flush=True,
    )
    print(f"  difficulty    : {scenario.difficulty}", flush=True)
    print(f"  topic         : {scenario.topic}", flush=True)
    print(f"  student_state : {scenario.student_state}", flush=True)
    print(f"  ideal_move    : {scenario.ideal_move}", flush=True)
    print(f"  problem       : {preview(scenario.problem, 160)}", flush=True)
    print(f"  student       : {preview(scenario.student_attempt, 160)}", flush=True)
    print("-" * 80, flush=True)


def print_record_summary(record: dict) -> None:
    judgment = record["judgment"]
    print(f"  selected_act  : {record.get('selected_act') or '-'}", flush=True)
    if record.get("act_rationale"):
        print(f"  act_rationale : {preview(record['act_rationale'], 180)}", flush=True)
    print(f"  response      : {preview(record['response'], 220)}", flush=True)
    print(
        "  scores        : "
        f"leakage={judgment['leakage']} "
        f"pedagogy_raw={judgment['pedagogy_raw_mean']:.2f} "
        f"pedagogy_capped={judgment['pedagogy_mean']:.2f} "
        f"correctness={judgment['correctness']} "
        f"scaffolding={judgment['scaffolding']} "
        f"self_correction={judgment['self_correction_support']}",
        flush=True,
    )
    print(f"  judge_reason  : {preview(judgment.get('reasoning'), 260)}", flush=True)
    print(f"  judge_summary : {preview(judgment.get('summary'), 220)}", flush=True)
    print_rule("-")


def record_sort_key(record: dict) -> tuple[int, int, str, str]:
    return (
        SYSTEM_PRIORITY.get(str(record.get("system", "")), len(SYSTEM_PRIORITY)),
        int(record.get("repetition_index", 0) or 0),
        str(record.get("scenario_id", "")),
        str(record.get("timestamp", "")),
    )


def record_identity_key(record: dict) -> tuple[str, str, int, str, str]:
    return (
        str(record.get("run_id", "")),
        str(record.get("system", "")),
        int(record.get("repetition_index", 0) or 0),
        str(record.get("scenario_id", "")),
        str(record.get("timestamp", "")),
    )


def merge_records(*record_groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str, int, str, str]] = set()
    for group in record_groups:
        for record in group:
            key = record_identity_key(record)
            if key in seen:
                continue
            seen.add(key)
            merged.append(record)
    return merged


def load_jsonl_if_exists(path: str | Path) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    return load_jsonl(target)


def initialize_run_output_files(*, output_dir: Path, run_dir: Path, systems: list[str]) -> None:
    save_jsonl(output_dir / "all_results.jsonl", [])
    save_jsonl(run_dir / "all_results.jsonl", [])
    for system_name in systems:
        save_jsonl(output_dir / f"{system_name}_results.jsonl", [])
        save_jsonl(run_dir / f"{system_name}_results.jsonl", [])


def checkpoint_record(*, record: dict, output_dir: Path, run_dir: Path) -> None:
    system_name = str(record["system"])
    append_jsonl(output_dir / f"{system_name}_results.jsonl", [record])
    append_jsonl(run_dir / f"{system_name}_results.jsonl", [record])
    append_jsonl(output_dir / "all_results.jsonl", [record])
    append_jsonl(run_dir / "all_results.jsonl", [record])


def build_run_manifest_entry(
    *,
    run_id: str,
    run_label: str,
    systems: list[str],
    scenario_count: int,
    repetitions: int,
    config,
    total_records: int,
    status: str,
    error: BaseException | None,
) -> dict:
    payload = {
        "run_id": run_id,
        "run_label": run_label,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "systems": systems,
        "scenario_count": scenario_count,
        "repetitions": repetitions,
        "planned_records": scenario_count * repetitions * len(systems),
        "total_records": total_records,
        "status": status,
        "scenario_path": config.scenario_path,
        "generation_model": config.generation_model,
        "act_selection_model": config.act_selection_model,
        "judge_model": config.judge_model,
        "include_student_state": config.include_student_state,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "judge_max_output_tokens": config.judge_max_output_tokens,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
    return payload


def finalize_run_outputs(
    *,
    all_records: list[dict],
    systems: list[str],
    output_dir: Path,
    run_dir: Path,
    history_dir: Path,
    manifest_entry: dict,
    append_history: bool,
) -> dict[str, Path]:
    ordered_all_records = sorted(all_records, key=record_sort_key)
    save_jsonl(output_dir / "all_results.jsonl", ordered_all_records)
    save_jsonl(run_dir / "all_results.jsonl", ordered_all_records)
    for system_name in systems:
        system_records = [record for record in ordered_all_records if record.get("system") == system_name]
        save_jsonl(output_dir / f"{system_name}_results.jsonl", system_records)
        save_jsonl(run_dir / f"{system_name}_results.jsonl", system_records)

    report_path = write_markdown_report(ordered_all_records, output_dir / "summary.md")
    run_report_path = write_markdown_report(ordered_all_records, run_dir / "summary.md")
    markdown_table_path = write_markdown_table(ordered_all_records, output_dir / "scenario_results_table.md")
    run_markdown_table_path = write_markdown_table(
        ordered_all_records,
        run_dir / "scenario_results_table.md",
    )
    csv_table_path = write_csv_table(ordered_all_records, output_dir / "scenario_results_table.csv")
    run_csv_table_path = write_csv_table(ordered_all_records, run_dir / "scenario_results_table.csv")
    difficulty_csv_path = write_difficulty_summary_csv(
        ordered_all_records,
        output_dir / "difficulty_summary.csv",
    )
    run_difficulty_csv_path = write_difficulty_summary_csv(
        ordered_all_records,
        run_dir / "difficulty_summary.csv",
    )

    history_path = history_dir / "all_results_history.jsonl"
    if append_history and ordered_all_records:
        append_jsonl(history_path, ordered_all_records)
    history_records = load_jsonl_if_exists(history_path)
    history_csv_path = write_csv_table(history_records, history_dir / "all_results_history.csv")
    coverage_csv_path = write_coverage_csv(history_records, history_dir / "coverage.csv")

    manifest_path = history_dir / "run_manifest.jsonl"
    append_jsonl(manifest_path, [manifest_entry])
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest_path.write_text(json.dumps(manifest_entry, indent=2) + "\n", encoding="utf-8")

    return {
        "report_path": report_path,
        "run_report_path": run_report_path,
        "markdown_table_path": markdown_table_path,
        "run_markdown_table_path": run_markdown_table_path,
        "csv_table_path": csv_table_path,
        "run_csv_table_path": run_csv_table_path,
        "difficulty_csv_path": difficulty_csv_path,
        "run_difficulty_csv_path": run_difficulty_csv_path,
        "history_csv_path": history_csv_path,
        "coverage_csv_path": coverage_csv_path,
        "manifest_path": manifest_path,
        "run_manifest_path": run_manifest_path,
    }


async def run_single_system(
    *,
    system_name: str,
    scenario: Scenario,
    client: OpenAITextClient,
    config,
    run_id: str,
    run_label: str,
    repetition_index: int,
) -> dict:
    selected_act: str | None = None
    rationale = ""

    if system_name == "direct":
        response = await agenerate_direct_baseline(client, scenario, config)
    elif system_name == "pedagogical":
        response = await agenerate_pedagogical_baseline(client, scenario, config)
    elif system_name == "act_conditioned":
        act_result = await aselect_act(client, scenario, config)
        selected_act = act_result["selected_act"]
        rationale = act_result["rationale"]
        response = await agenerate_act_conditioned_response(
            client,
            scenario,
            selected_act,
            config,
        )
    else:
        raise ValueError(f"Unknown system: {system_name}")

    judgment = await ajudge_response(
        client=client,
        scenario=scenario,
        response=response,
        config=config,
        selected_act=selected_act,
    )

    return {
        "run_id": run_id,
        "run_label": run_label,
        "repetition_index": repetition_index,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "generation_model": config.generation_model,
        "act_selection_model": config.act_selection_model,
        "judge_model": config.judge_model,
        "student_state_visible_to_tutor": config.include_student_state,
        "system": system_name,
        "scenario_id": scenario.scenario_id,
        "problem": scenario.problem,
        "difficulty": scenario.difficulty,
        "topic": scenario.topic,
        "student_state": scenario.student_state,
        "student_attempt": scenario.student_attempt,
        "dialogue_context": scenario.dialogue_context,
        "ideal_move": scenario.ideal_move,
        "selected_act": selected_act,
        "act_rationale": rationale,
        "response": response,
        "judgment": judgment.to_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tutoring evaluation pipeline.")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["direct", "pedagogical", "act_conditioned"],
        choices=["direct", "pedagogical", "act_conditioned"],
        help="Systems to run.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional number of scenarios to run.")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=0,
        help="Number of repeated trials per scenario and system. Defaults to DEFAULT_REPETITIONS from .env.",
    )
    parser.add_argument(
        "--run-label",
        default="",
        help="Optional short label to store with this run in history outputs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-scenario progress and API call feedback.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug logging even if DEBUG_API is enabled in .env.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Maximum number of in-flight scenario evaluations per system. Defaults to MAX_CONCURRENCY from .env.",
    )
    student_state_group = parser.add_mutually_exclusive_group()
    student_state_group.add_argument(
        "--include-student-state",
        dest="include_student_state",
        action="store_true",
        help="Pass the scenario's student_state label to the tutor systems.",
    )
    student_state_group.add_argument(
        "--omit-student-state",
        dest="include_student_state",
        action="store_false",
        help="Do not pass the scenario's student_state label to the tutor systems.",
    )
    parser.set_defaults(include_student_state=None)
    return parser.parse_args()


async def run_system_records(
    *,
    system_name: str,
    scenarios: list[Scenario],
    client: OpenAITextClient,
    config,
    run_id: str,
    run_label: str,
    repetition_index: int,
    repetitions: int,
    on_record: Callable[[dict], None] | None = None,
) -> list[dict]:
    concurrency = max(1, config.max_concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(index: int, scenario: Scenario) -> tuple[int, dict]:
        async with semaphore:
            if config.debug_api:
                print_scenario_header(
                    system_name=system_name,
                    scenario=scenario,
                    index=index,
                    total=len(scenarios),
                    repetition_index=repetition_index,
                    repetitions=repetitions,
                )
            record = await run_single_system(
                system_name=system_name,
                scenario=scenario,
                client=client,
                config=config,
                run_id=run_id,
                run_label=run_label,
                repetition_index=repetition_index,
            )
            if config.debug_api:
                print_record_summary(record)
            return index, record

    indexed_records: list[tuple[int, dict]] = []
    scenario_iter = iter(enumerate(scenarios, start=1))
    pending: set[asyncio.Task] = set()
    task_metadata: dict[asyncio.Task, tuple[int, Scenario]] = {}
    first_error: BaseException | None = None

    def schedule(index: int, scenario: Scenario) -> None:
        task = asyncio.create_task(worker(index, scenario))
        pending.add(task)
        task_metadata[task] = (index, scenario)

    for _ in range(min(concurrency, len(scenarios))):
        try:
            next_index, next_scenario = next(scenario_iter)
        except StopIteration:
            break
        schedule(next_index, next_scenario)

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            index, scenario = task_metadata.pop(task)
            try:
                _, record = await task
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                    print(
                        f"Error in {system_name} repetition {repetition_index} on {scenario.scenario_id}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                continue
            indexed_records.append((index, record))
            if on_record is not None:
                on_record(record)

        if first_error is not None:
            continue

        while len(pending) < concurrency:
            try:
                next_index, next_scenario = next(scenario_iter)
            except StopIteration:
                break
            schedule(next_index, next_scenario)

    indexed_records.sort(key=lambda item: item[0])
    ordered_records = [record for _, record in indexed_records]
    if first_error is not None:
        raise PartialRunError(
            system_name=system_name,
            repetition_index=repetition_index,
            partial_records=ordered_records,
            original_error=first_error,
        ) from first_error
    return ordered_records


async def amain() -> None:
    args = parse_args()
    config = load_config()
    debug_enabled = config.debug_api
    if args.debug:
        debug_enabled = True
    if args.no_debug:
        debug_enabled = False
    concurrency = config.max_concurrency if args.concurrency <= 0 else args.concurrency
    repetitions = config.default_repetitions if args.repetitions <= 0 else args.repetitions
    include_student_state = config.include_student_state
    if args.include_student_state is not None:
        include_student_state = bool(args.include_student_state)
    config = replace(
        config,
        debug_api=debug_enabled,
        max_concurrency=max(1, concurrency),
        include_student_state=include_student_state,
    )

    scenarios = load_scenarios(config.scenario_path)
    if args.limit > 0:
        scenarios = scenarios[: args.limit]

    client = OpenAITextClient(debug=config.debug_api)
    output_dir = Path(config.output_dir)
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    run_label = args.run_label.strip()
    run_dir = output_dir / "runs" / run_id
    history_dir = output_dir / "history"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    initialize_run_output_files(output_dir=output_dir, run_dir=run_dir, systems=args.systems)
    print_run_header(
        run_id=run_id,
        run_label=run_label,
        systems=args.systems,
        scenario_count=len(scenarios),
        repetitions=repetitions,
        output_dir=output_dir,
        debug=config.debug_api,
        concurrency=config.max_concurrency,
        include_student_state=config.include_student_state,
    )
    if config.debug_api and config.max_concurrency > 1:
        print("Note: debug output may interleave when concurrency is greater than 1.", flush=True)
        print_rule()

    all_records: list[dict] = []
    run_error: BaseException | None = None
    finalized_paths: dict[str, Path] = {}
    current_system_name = ""
    current_repetition_index = 0

    try:
        for system_name in args.systems:
            current_system_name = system_name
            print_system_header(system_name, len(scenarios) * repetitions)
            system_records: list[dict] = [
                record for record in all_records if record.get("system") == system_name
            ]
            for repetition_index in range(1, repetitions + 1):
                current_repetition_index = repetition_index
                if repetitions > 1:
                    print(f"Repetition {repetition_index}/{repetitions}", flush=True)
                try:
                    repetition_records = await run_system_records(
                        system_name=system_name,
                        scenarios=scenarios,
                        client=client,
                        config=config,
                        run_id=run_id,
                        run_label=run_label,
                        repetition_index=repetition_index,
                        repetitions=repetitions,
                        on_record=lambda record, output_dir=output_dir, run_dir=run_dir: checkpoint_record(
                            record=record,
                            output_dir=output_dir,
                            run_dir=run_dir,
                        ),
                    )
                except PartialRunError as exc:
                    system_records.extend(exc.partial_records)
                    all_records.extend(exc.partial_records)
                    run_error = exc.original_error
                    print(exc, flush=True)
                    print("No new scenarios will be scheduled for this run; finalized outputs will be written.", flush=True)
                    print_rule("-")
                    break
                system_records.extend(repetition_records)
                all_records.extend(repetition_records)

            if system_records:
                summary = aggregate(system_records)
                label = "Partial Summary" if run_error is not None else "Summary"
                print(label, flush=True)
                for line in system_summary_lines({system_name: summary}):
                    print(f"  {line}", flush=True)
                print_rule("-")
            else:
                print("No completed records for this system.", flush=True)
                print_rule("-")

            if run_error is not None:
                break
    except BaseException as exc:
        run_error = exc

    checkpointed_records = load_jsonl_if_exists(run_dir / "all_results.jsonl")
    all_records = merge_records(all_records, checkpointed_records)
    manifest_entry = build_run_manifest_entry(
        run_id=run_id,
        run_label=run_label,
        systems=args.systems,
        scenario_count=len(scenarios),
        repetitions=repetitions,
        config=config,
        total_records=len(all_records),
        status="failed" if run_error is not None else "completed",
        error=run_error,
    )
    finalized_paths = finalize_run_outputs(
        all_records=all_records,
        systems=args.systems,
        output_dir=output_dir,
        run_dir=run_dir,
        history_dir=history_dir,
        manifest_entry=manifest_entry,
        append_history=run_error is None,
    )

    system_summaries = summarize_by_system(all_records)
    print("Final Summary", flush=True)
    print_rule()
    for line in system_summary_lines(system_summaries):
        print(line, flush=True)
    print_rule()
    print(f"Run status  : {manifest_entry['status']}", flush=True)
    print(f"Saved report: {finalized_paths['report_path']}", flush=True)
    print(f"Saved run report: {finalized_paths['run_report_path']}", flush=True)
    print(f"Saved markdown table: {finalized_paths['markdown_table_path']}", flush=True)
    print(f"Saved run markdown table: {finalized_paths['run_markdown_table_path']}", flush=True)
    print(f"Saved csv table: {finalized_paths['csv_table_path']}", flush=True)
    print(f"Saved run csv table: {finalized_paths['run_csv_table_path']}", flush=True)
    print(f"Saved difficulty csv: {finalized_paths['difficulty_csv_path']}", flush=True)
    print(f"Saved run difficulty csv: {finalized_paths['run_difficulty_csv_path']}", flush=True)
    print(f"Saved cumulative history csv: {finalized_paths['history_csv_path']}", flush=True)
    print(f"Saved coverage csv: {finalized_paths['coverage_csv_path']}", flush=True)
    print(f"Saved history manifest: {finalized_paths['manifest_path']}", flush=True)
    print(f"Saved run manifest: {finalized_paths['run_manifest_path']}", flush=True)
    if run_error is not None:
        print(
            f"Run stopped while processing system={current_system_name or '-'} "
            f"repetition={current_repetition_index or '-'} "
            f"after salvaging {len(all_records)} completed record(s).",
            flush=True,
        )
        print("Partial run records were not appended to cumulative history.", flush=True)
        raise run_error


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
