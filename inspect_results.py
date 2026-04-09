from __future__ import annotations

import argparse

from src.reporting import (
    load_jsonl,
    render_example,
    select_examples,
    summarize_by_system,
    system_summary_lines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect saved tutoring evaluation results.")
    parser.add_argument(
        "--path",
        default="outputs/all_results.jsonl",
        help="Path to a JSONL results file.",
    )
    parser.add_argument(
        "--system",
        default="act_conditioned",
        help="System to inspect in detail.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of best/worst examples to show.",
    )
    parser.add_argument(
        "--scenario-id",
        default="",
        help="Optional scenario id to filter to a single example.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.path)

    if args.scenario_id:
        filtered = [record for record in records if record["scenario_id"] == args.scenario_id]
        if not filtered:
            print(f"No records found for scenario_id={args.scenario_id}")
            return
        for record in filtered:
            print(f"{record['system']} | {record['scenario_id']}")
            print(render_example(record))
            print()
        return

    summaries = summarize_by_system(records)
    print("System Summary")
    print("=" * 80)
    for line in system_summary_lines(summaries):
        print(line)
    print("=" * 80)

    selected_examples = select_examples(records, system_name=args.system, top_k=args.top)
    print(f"Best {args.system} examples")
    print("-" * 80)
    for record in selected_examples["best"]:
        print(render_example(record))
        print()

    print(f"Worst {args.system} examples")
    print("-" * 80)
    for record in selected_examples["worst"]:
        print(render_example(record))
        print()

    if selected_examples["leakage"]:
        print(f"Leakage cases for {args.system}")
        print("-" * 80)
        for record in selected_examples["leakage"]:
            print(render_example(record))
            print()
    else:
        print(f"No leakage cases found for {args.system}.")


if __name__ == "__main__":
    main()
