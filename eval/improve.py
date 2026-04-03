"""
Prompt improvement loop: evaluates current prompts, identifies failure patterns,
generates improved prompts via Claude, and iterates until correctness >= 90%.

Usage:
    uv run python -m eval.improve [--target 90] [--max-iterations 5]

Requires the server to be running with 'make dev' (--reload mode).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv
from rich.console import Console

load_dotenv(override=True)

from eval.run_eval import CaseResult, EvalRun, check_server_health, run_eval

console = Console()

MAX_ITERATIONS = 5
TARGET_PCT = 90.0

PROMPT_FILE_MAP = {
    "analyze": "app/agent/nodes/analyze.py",
    "grade": "app/agent/nodes/grade.py",
    "rewrite": "app/agent/nodes/rewrite.py",
    "generate": "app/agent/nodes/generate.py",
}

PROMPT_CONSTANT_MAP = {
    "analyze": "SYSTEM_PROMPT",
    "grade": "GRADE_SYSTEM",
    "rewrite": "REWRITE_SYSTEM",
    "generate": "GENERATE_SYSTEM",
}

# Maps failure categories (from judge) to which prompts to improve
FAILURE_CATEGORY_TO_PROMPTS: dict[str, list[str]] = {
    "wrong_cards": ["grade"],
    "negative_result_returned": ["grade"],
    "missing_key_variant": ["grade", "analyze"],
    "empty_result": ["rewrite", "grade"],
    "hallucinated_answer": ["generate"],
}

# Maps dataset categories to which prompts to improve
DATASET_CATEGORY_TO_PROMPTS: dict[str, list[str]] = {
    "composite_query": ["analyze"],
    "slang_alias": ["analyze", "generate"],
    "special_characters": ["analyze", "generate"],
    "error_tolerance": ["rewrite"],
    "identification": ["generate"],
    "disambiguation": ["analyze"],
    "name_mechanic_lookup": ["analyze", "grade"],
}


@dataclass
class FailureAnalysis:
    total_failures: int
    by_failure_category: dict[str, list[CaseResult]]
    by_dataset_category: dict[str, list[CaseResult]]
    primary_pattern: str
    recommended_prompt_targets: list[str]
    representative_failures: list[CaseResult]


def analyze_failures(failed_cases: list[CaseResult]) -> FailureAnalysis:
    by_failure: dict[str, list[CaseResult]] = {}
    for c in failed_cases:
        key = c.failure_category or "unknown"
        by_failure.setdefault(key, []).append(c)

    by_dataset: dict[str, list[CaseResult]] = {}
    for c in failed_cases:
        by_dataset.setdefault(c.category, []).append(c)

    # Determine recommended prompts by vote: each failure contributes votes
    prompt_votes: dict[str, int] = {}

    for failure_cat, cases in by_failure.items():
        for prompt in FAILURE_CATEGORY_TO_PROMPTS.get(failure_cat, []):
            prompt_votes[prompt] = prompt_votes.get(prompt, 0) + len(cases)

    for dataset_cat, cases in by_dataset.items():
        for prompt in DATASET_CATEGORY_TO_PROMPTS.get(dataset_cat, []):
            prompt_votes[prompt] = prompt_votes.get(prompt, 0) + len(cases)

    # Sort by vote count, take top 2
    recommended = sorted(prompt_votes, key=lambda p: -prompt_votes[p])[:2]
    if not recommended:
        recommended = ["grade"]  # fallback

    # Primary pattern description
    if by_failure:
        top_failure = max(by_failure, key=lambda k: len(by_failure[k]))
        primary_pattern = (
            f"Most common failure: '{top_failure}' in {len(by_failure[top_failure])} cases. "
            f"Top dataset categories with failures: {', '.join(list(by_dataset.keys())[:3])}."
        )
    else:
        primary_pattern = "Various failures across categories."

    # Representative failures: pick diverse set (by category and failure type), up to 5
    seen_cats: set[str] = set()
    representative: list[CaseResult] = []
    # First pass: one per category
    for c in sorted(failed_cases, key=lambda x: x.score):
        if c.category not in seen_cats and len(representative) < 5:
            representative.append(c)
            seen_cats.add(c.category)
    # Second pass: fill remaining by lowest score
    for c in sorted(failed_cases, key=lambda x: x.score):
        if c not in representative and len(representative) < 5:
            representative.append(c)

    return FailureAnalysis(
        total_failures=len(failed_cases),
        by_failure_category=by_failure,
        by_dataset_category=by_dataset,
        primary_pattern=primary_pattern,
        recommended_prompt_targets=recommended,
        representative_failures=representative,
    )


def read_prompt_constant(prompt_name: str) -> str:
    file_path = Path(PROMPT_FILE_MAP[prompt_name])
    constant_name = PROMPT_CONSTANT_MAP[prompt_name]
    text = file_path.read_text()
    pattern = rf'{re.escape(constant_name)}\s*=\s*"""(.*?)"""'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find {constant_name} in {file_path}")
    return match.group(1)


def replace_prompt_constant(file_text: str, constant_name: str, new_prompt: str) -> str:
    pattern = rf'({re.escape(constant_name)}\s*=\s*""")(.*?)(""")'
    replacement = rf'\g<1>{new_prompt}\g<3>'
    result = re.sub(pattern, replacement, file_text, flags=re.DOTALL)
    if result == file_text:
        raise ValueError(f"Replacement had no effect for {constant_name}")
    return result


def write_prompt_constant(prompt_name: str, new_prompt_text: str) -> None:
    file_path = Path(PROMPT_FILE_MAP[prompt_name])
    constant_name = PROMPT_CONSTANT_MAP[prompt_name]
    original = file_path.read_text()
    updated = replace_prompt_constant(original, constant_name, new_prompt_text)
    file_path.write_text(updated)


def format_failure_for_meta_prompt(case: CaseResult) -> str:
    return (
        f"  Case {case.case_id} ({case.category}, {case.difficulty}):\n"
        f"    Query: {case.query!r}\n"
        f"    Score: {case.score:.2f} | Failure: {case.failure_category}\n"
        f"    Cards returned: {case.actual_cards_count}\n"
        f"    Answer snippet: {case.actual_answer_snippet[:150]!r}\n"
        f"    Judge reasoning: {case.reasoning[:200]}"
    )


_IMPROVE_SYSTEM = """You are an expert at writing system prompts for LLM-powered search systems.

You will be given:
1. The current system prompt for a node in a Pokemon card search pipeline
2. Representative failing test cases (what the user searched, what was returned, why it failed)
3. The failure pattern diagnosis

Your task: rewrite the system prompt to address the failures while preserving all existing capabilities.

Rules:
- Return ONLY the new prompt text — no explanation, no markdown fences, no preamble
- Preserve the exact JSON output format specified in the current prompt
- Do not add capabilities the system doesn't have (e.g., don't promise to filter by grade or edition)
- Make targeted improvements: fix the identified failure pattern without breaking passing cases
- Keep the prompt concise — avoid padding or redundant instructions"""


async def generate_improved_prompt(
    prompt_name: str,
    current_prompt: str,
    analysis: FailureAnalysis,
    client: anthropic.AsyncAnthropic,
) -> str:
    failure_examples = "\n\n".join(
        format_failure_for_meta_prompt(c) for c in analysis.representative_failures
    )

    user_message = (
        f"CURRENT PROMPT FOR '{prompt_name}':\n"
        f"---\n{current_prompt}\n---\n\n"
        f"FAILURE PATTERN:\n{analysis.primary_pattern}\n\n"
        f"REPRESENTATIVE FAILING CASES:\n{failure_examples}\n\n"
        f"Rewrite the prompt to reduce these failures. Return only the new prompt text."
    )

    message = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=_IMPROVE_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )

    new_text = message.content[0].text.strip()
    # Remove any accidental markdown code fences
    if new_text.startswith("```"):
        parts = new_text.split("```")
        new_text = parts[1] if len(parts) > 1 else new_text
        if new_text.startswith("text") or new_text.startswith("prompt"):
            new_text = new_text[new_text.index("\n") + 1:]
    return new_text.strip()


async def wait_for_reload(api_base: str, retries: int = 15, delay: float = 1.5) -> bool:
    """Poll health endpoint until server responds (handles uvicorn --reload)."""
    # Brief initial wait to let reload start
    await asyncio.sleep(2.0)
    async with httpx.AsyncClient() as client:
        for _ in range(retries):
            if await check_server_health(api_base, client):
                return True
            await asyncio.sleep(delay)
    return False


def save_report(
    iterations: list[dict],
    start_pct: float,
    final_pct: float,
    target_pct: float,
    output_dir: str,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"improve_report_{ts}.md"

    lines = [
        f"# Prompt Improvement Run — {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        f"- Starting correctness: {start_pct:.1f}%",
        f"- Final correctness: {final_pct:.1f}%",
        f"- Iterations: {len(iterations)}",
        f"- Target ({target_pct}%) reached: {'Yes' if final_pct >= target_pct else 'No'}",
        "",
    ]

    for i, it in enumerate(iterations, 1):
        delta = it["score_after"] - it["score_before"]
        sign = "+" if delta >= 0 else ""
        lines += [
            f"## Iteration {i}: {it['score_before']:.1f}% → {it['score_after']:.1f}% ({sign}{delta:.1f}%)",
            f"- Prompts modified: {', '.join(it['prompts_modified'])}",
            f"- Change kept: {'Yes' if it['kept'] else 'No (reverted)'}",
            "",
        ]
        for pname, pdata in it.get("prompt_changes", {}).items():
            lines += [
                f"### {pname}.py — {PROMPT_CONSTANT_MAP[pname]}",
                f"**Why:** {pdata['failure_pattern']}",
                "",
                "**New prompt:**",
                "```",
                pdata["new_prompt"][:800],
                "```",
                "",
            ]

    report_path.write_text("\n".join(lines))
    return report_path


async def improve(
    dataset_path: str = "eval/pokemon-tcg-search-golden-dataset.json",
    api_base: str = "http://localhost:8000",
    output_dir: str = "eval/results",
    target_pct: float = TARGET_PCT,
    max_iterations: int = MAX_ITERATIONS,
    concurrency: int = 3,
) -> None:
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        console.print("[red]ANTHROPIC_API_KEY not set[/red]")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=anthropic_key)

    console.print(f"[bold]Prompt Improvement Loop[/bold] — target: {target_pct}%, max iterations: {max_iterations}")
    console.print("Server must be running with 'make dev' (--reload mode)\n")

    # Initial evaluation
    console.print("[bold cyan]--- Initial Evaluation ---[/bold cyan]")
    current_run = await run_eval(
        dataset_path=dataset_path,
        api_base=api_base,
        output_dir=output_dir,
        concurrency=concurrency,
    )

    start_pct = current_run.correctness_pct
    iterations_log: list[dict] = []

    if current_run.correctness_pct >= target_pct:
        console.print(f"[green]Already at {current_run.correctness_pct:.1f}% — target reached![/green]")
        return

    for iteration in range(1, max_iterations + 1):
        console.print(f"\n[bold cyan]--- Iteration {iteration}/{max_iterations} ---[/bold cyan]")
        console.print(f"Current correctness: {current_run.correctness_pct:.1f}%")

        failed_cases = [c for c in current_run.cases if not c.passed]
        if not failed_cases:
            console.print("[green]No failures remaining![/green]")
            break

        analysis = analyze_failures(failed_cases)
        console.print(f"Failure analysis: {analysis.primary_pattern}")
        console.print(f"Targeting prompts: {analysis.recommended_prompt_targets}")

        # Back up current prompts
        backups: dict[str, str] = {}
        prompt_changes: dict[str, dict] = {}

        for prompt_name in analysis.recommended_prompt_targets:
            current_prompt = read_prompt_constant(prompt_name)
            backups[prompt_name] = current_prompt

            console.print(f"  Generating improved prompt for '{prompt_name}'...")
            try:
                new_prompt = await generate_improved_prompt(
                    prompt_name=prompt_name,
                    current_prompt=current_prompt,
                    analysis=analysis,
                    client=client,
                )
                write_prompt_constant(prompt_name, new_prompt)
                console.print(f"  [green]Updated {PROMPT_FILE_MAP[prompt_name]}[/green]")
                prompt_changes[prompt_name] = {
                    "failure_pattern": analysis.primary_pattern,
                    "new_prompt": new_prompt,
                }
            except Exception as e:
                console.print(f"  [red]Failed to improve {prompt_name}: {e}[/red]")

        if not prompt_changes:
            console.print("[yellow]No prompts were updated. Stopping.[/yellow]")
            break

        # Wait for server to reload
        console.print("Waiting for server to reload...")
        reloaded = await wait_for_reload(api_base)
        if not reloaded:
            console.print("[red]Server did not come back healthy. Reverting.[/red]")
            for pname, original in backups.items():
                write_prompt_constant(pname, original)
            break

        # Re-evaluate
        console.print("[bold cyan]Re-evaluating...[/bold cyan]")
        new_run = await run_eval(
            dataset_path=dataset_path,
            api_base=api_base,
            output_dir=output_dir,
            concurrency=concurrency,
        )

        kept = new_run.correctness_pct > current_run.correctness_pct
        iterations_log.append({
            "score_before": current_run.correctness_pct,
            "score_after": new_run.correctness_pct,
            "prompts_modified": list(prompt_changes.keys()),
            "kept": kept,
            "prompt_changes": prompt_changes,
        })

        if kept:
            console.print(
                f"[green]Improvement: {current_run.correctness_pct:.1f}% → {new_run.correctness_pct:.1f}% (kept)[/green]"
            )
            current_run = new_run
        else:
            console.print(
                f"[yellow]No improvement ({current_run.correctness_pct:.1f}% → {new_run.correctness_pct:.1f}%). Reverting.[/yellow]"
            )
            for pname, original in backups.items():
                write_prompt_constant(pname, original)
            # Wait for reload after revert
            await wait_for_reload(api_base)

        if current_run.correctness_pct >= target_pct:
            console.print(f"\n[bold green]Target {target_pct}% reached! Final: {current_run.correctness_pct:.1f}%[/bold green]")
            break

    # Save report
    report_path = save_report(
        iterations=iterations_log,
        start_pct=start_pct,
        final_pct=current_run.correctness_pct,
        target_pct=target_pct,
        output_dir=output_dir,
    )
    console.print(f"\nReport saved to [bold]{report_path}[/bold]")

    delta = current_run.correctness_pct - start_pct
    sign = "+" if delta >= 0 else ""
    console.print(
        f"\n[bold]Final: {start_pct:.1f}% → {current_run.correctness_pct:.1f}% ({sign}{delta:.1f}%)[/bold]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Iteratively improve search prompts using golden dataset")
    parser.add_argument("--dataset", default="eval/pokemon-tcg-search-golden-dataset.json")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", default="eval/results")
    parser.add_argument("--target", type=float, default=TARGET_PCT, help="Target correctness %")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    asyncio.run(improve(
        dataset_path=args.dataset,
        api_base=args.api,
        output_dir=args.out,
        target_pct=args.target,
        max_iterations=args.max_iterations,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
