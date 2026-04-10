This repository contains a lightweight implementation of AI NLP components from my selected papers (referenced in `gray344.bib`). It involves an AI math tutoring pipeline tested on 3 different system configurations to including BIPED's act-controlled approach. An evaluation system influenced by MathTutorBench's approach to measuring tutoring quality, is used to grade the 3 systems outcomes.

## Requirements
- Python `3.10+`
- An `OPENAI_API_KEY`
- Internet access for OpenAI API calls
The repo already includes the scenario data. There is no separate dataset download step.

## Setup
1. Install dependencies via `requirements.txt`
2. Set up `.env` file using `.env.example` as reference
3. Obtain OpenAI API key and set to `OPENAI_API_KEY` env variable

## Environment Variables
- `GENERATION_MODEL`: model used for tutor response generation
- `ACT_SELECTION_MODEL`: model used by the act selector
- `JUDGE_MODEL`: model used by the evaluator / judge
- `MAX_CONCURRENCY`: number of parallel scenario evaluations
- `DEFAULT_REPETITIONS`: repeated trials per system
- `INCLUDE_STUDENT_STATE`: whether tutor prompts receive the scenario's `student_state`
- `SCENARIO_PATH`: input JSONL scenario file
- `OUTPUT_DIR`: where reports and run artifacts are written

## How to Run Pipeline
For a quick smoke test:
```powershell
python run_eval.py --limit 3 --run-label smoke
```

For a fuller run:
```powershell
python run_eval.py --repetitions 3 --run-label main
```

`run_eval.py` flags:
- `--limit N`: run only the first `N` scenarios
- `--systems ...`: choose any subset of `direct`, `pedagogical`, `act_conditioned`
- `--repetitions N`: override `DEFAULT_REPETITIONS`
- `--concurrency N`: override `MAX_CONCURRENCY`
- `--debug`: print detailed per-scenario progress and API activity
- `--omit-student-state`: hide `student_state` from tutor prompts
- `--include-student-state`: explicitly pass `student_state` to tutor prompts

## Outputs
Each run writes both "latest" outputs and a timestamped run directory, default to `outputs/`.
Generated files:
- `outputs/all_results.jsonl`
- `outputs/direct_results.jsonl`
- `outputs/pedagogical_results.jsonl`
- `outputs/act_conditioned_results.jsonl`
- `outputs/summary.md`
- `outputs/scenario_results_table.md`
- `outputs/scenario_results_table.csv`
- `outputs/difficulty_summary.csv`
- `outputs/runs/<run_id>/...`
- `outputs/history/all_results_history.jsonl`
- `outputs/history/run_manifest.jsonl`

## Inspect Saved Results
Inspect the latest aggregate results:
```powershell
python inspect_results.py
```

Inspect a specific run file:
```powershell
python inspect_results.py --path outputs/runs/<run_id>/all_results.jsonl --system act_conditioned --top 5
```

Inspect a single scenario across systems:
```powershell
python inspect_results.py --path outputs/all_results.jsonl --scenario-id <scenario_id>
```

## Inspect The Prompts
Render a markdown snapshot in `outputs/` of the fully assembled prompts:
```powershell
python inspect_prompts.py --scenario-id <scenario_id>
```

```powershell
python inspect_prompts.py --scenario-id <scenario_id> --omit-student-state
```

## Scenario Data Format
The default dataset is `data/scenarios.jsonl`. Each JSONL row should contain:
If you want to use your own scenario set, create a JSONL file with that schema and point `SCENARIO_PATH` to it.

## Common Issues
- `OPENAI_API_KEY is not set`: copy `.env.example` to `.env` and add your key.
- `ImportError` for `openai` or `dotenv`: activate the virtual environment and run `pip install -r requirements.txt`.
- `openai.PermissionDeniedError: Error code: 403`: your API key's project likely does not have access to one of the configured models. The default suspect is `JUDGE_MODEL=gpt-5-mini`; if needed, switch it to a model your project can access, such as `gpt-4.1-mini-2025-04-14`.
- Slow or expensive runs: lower `--limit`, lower `--repetitions`, or reduce `MAX_CONCURRENCY`.
- Debug logs interleave: this is expected when concurrency is greater than `1`; use `--concurrency 1` for cleaner step-by-step logs.