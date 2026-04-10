from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime
from pathlib import Path

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

    if concurrency == 1:
        ordered_records: list[dict] = []
        for index, scenario in enumerate(scenarios, start=1):
            _, record = await worker(index, scenario)
            ordered_records.append(record)
        return ordered_records

    tasks = [
        asyncio.create_task(worker(index, scenario))
        for index, scenario in enumerate(scenarios, start=1)
    ]
    indexed_records = await asyncio.gather(*tasks)
    indexed_records.sort(key=lambda item: item[0])
    return [record for _, record in indexed_records]


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
    run_dir.mkdir(parents=True, exist_ok=True)
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
    for system_name in args.systems:
        print_system_header(system_name, len(scenarios) * repetitions)
        system_records: list[dict] = []
        for repetition_index in range(1, repetitions + 1):
            if repetitions > 1:
                print(f"Repetition {repetition_index}/{repetitions}", flush=True)
            repetition_records = await run_system_records(
                system_name=system_name,
                scenarios=scenarios,
                client=client,
                config=config,
                run_id=run_id,
                run_label=run_label,
                repetition_index=repetition_index,
                repetitions=repetitions,
            )
            system_records.extend(repetition_records)
        save_jsonl(output_dir / f"{system_name}_results.jsonl", system_records)
        save_jsonl(run_dir / f"{system_name}_results.jsonl", system_records)
        summary = aggregate(system_records)
        print("Summary", flush=True)
        for line in system_summary_lines({system_name: summary}):
            print(f"  {line}", flush=True)
        print_rule("-")
        all_records.extend(system_records)

    save_jsonl(output_dir / "all_results.jsonl", all_records)
    save_jsonl(run_dir / "all_results.jsonl", all_records)
    append_jsonl(history_dir / "all_results_history.jsonl", all_records)
    append_jsonl(
        history_dir / "run_manifest.jsonl",
        [
            {
                "run_id": run_id,
                "run_label": run_label,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "systems": args.systems,
                "scenario_count": len(scenarios),
                "repetitions": repetitions,
                "total_records": len(all_records),
                "scenario_path": config.scenario_path,
                "generation_model": config.generation_model,
                "act_selection_model": config.act_selection_model,
                "judge_model": config.judge_model,
                "include_student_state": config.include_student_state,
                "temperature": config.temperature,
                "max_output_tokens": config.max_output_tokens,
                "judge_max_output_tokens": config.judge_max_output_tokens,
            }
        ],
    )
    report_path = write_markdown_report(all_records, output_dir / "summary.md")
    run_report_path = write_markdown_report(all_records, run_dir / "summary.md")
    markdown_table_path = write_markdown_table(all_records, output_dir / "scenario_results_table.md")
    run_markdown_table_path = write_markdown_table(all_records, run_dir / "scenario_results_table.md")
    csv_table_path = write_csv_table(all_records, output_dir / "scenario_results_table.csv")
    run_csv_table_path = write_csv_table(all_records, run_dir / "scenario_results_table.csv")
    difficulty_csv_path = write_difficulty_summary_csv(all_records, output_dir / "difficulty_summary.csv")
    run_difficulty_csv_path = write_difficulty_summary_csv(all_records, run_dir / "difficulty_summary.csv")
    history_records = load_jsonl(history_dir / "all_results_history.jsonl")
    history_csv_path = write_csv_table(history_records, history_dir / "all_results_history.csv")
    coverage_csv_path = write_coverage_csv(history_records, history_dir / "coverage.csv")
    manifest_path = history_dir / "run_manifest.jsonl"
    system_summaries = summarize_by_system(all_records)
    print("Final Summary", flush=True)
    print_rule()
    for line in system_summary_lines(system_summaries):
        print(line, flush=True)
    print_rule()
    print(f"Saved report: {report_path}", flush=True)
    print(f"Saved run report: {run_report_path}", flush=True)
    print(f"Saved markdown table: {markdown_table_path}", flush=True)
    print(f"Saved run markdown table: {run_markdown_table_path}", flush=True)
    print(f"Saved csv table: {csv_table_path}", flush=True)
    print(f"Saved run csv table: {run_csv_table_path}", flush=True)
    print(f"Saved difficulty csv: {difficulty_csv_path}", flush=True)
    print(f"Saved run difficulty csv: {run_difficulty_csv_path}", flush=True)
    print(f"Saved cumulative history csv: {history_csv_path}", flush=True)
    print(f"Saved coverage csv: {coverage_csv_path}", flush=True)
    print(f"Saved run manifest: {manifest_path}", flush=True)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
