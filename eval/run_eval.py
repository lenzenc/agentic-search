"""
Evaluation runner: loads the golden dataset, calls the live search API,
scores each result with the LLM judge, and saves a JSON report.

Usage:
    uv run python -m eval.run_eval [--dataset ...] [--api ...] [--out ...] [--concurrency N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import httpx
from rich.console import Console
from rich.table import Table

from eval.judge import JudgeResult, judge_case

console = Console()

DEFAULT_DATASET = "data/pokemon-tcg-search-golden-dataset.json"
DEFAULT_API = "http://localhost:8000"
DEFAULT_OUT = "eval/results"
DEFAULT_CONCURRENCY = 3
DEFAULT_TIMEOUT = 60.0


@dataclass
class CaseResult:
    case_id: str
    query: str
    category: str
    difficulty: str
    score: float
    passed: bool
    reasoning: str
    failure_category: str | None
    actual_cards_count: int
    actual_answer_snippet: str
    api_error: str | None
    latency_ms: int


@dataclass
class EvalRun:
    timestamp: str
    total: int
    passed: int
    failed: int
    correctness_pct: float
    cases: list[CaseResult] = field(default_factory=list)
    summary_by_category: dict[str, dict] = field(default_factory=dict)


async def parse_sse_stream(response: httpx.Response) -> dict:
    """Read an SSE stream and return the payload from the 'result' event."""
    current_event: str = ""
    current_data_lines: list[str] = []
    result_payload: dict | None = None

    async for line in response.aiter_lines():
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_data_lines.append(line[len("data:"):].strip())
        elif line == "":
            # End of one SSE message
            if current_event == "result" and current_data_lines:
                raw = "".join(current_data_lines)
                result_payload = json.loads(raw)
            elif current_event == "error" and current_data_lines:
                raw = "".join(current_data_lines)
                try:
                    err = json.loads(raw)
                    raise RuntimeError(f"API error: {err}")
                except json.JSONDecodeError:
                    raise RuntimeError(f"API error: {raw}")
            current_event = ""
            current_data_lines = []

    return result_payload or {}


async def check_server_health(api_base: str, client: httpx.AsyncClient) -> bool:
    try:
        resp = await client.get(f"{api_base}/api/health", timeout=5.0)
        data = resp.json()
        return data.get("status") == "ok"
    except Exception:
        return False


async def run_one_case(
    case: dict,
    api_base: str,
    http_client: httpx.AsyncClient,
    judge_client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    timeout: float,
) -> CaseResult:
    case_id = case["id"]
    query = case["query"]
    category = case.get("category", "unknown")
    difficulty = case.get("difficulty", "unknown")
    intent = case.get("intent", "")
    expected_results = case.get("expected_results", [])
    negative_results = case.get("negative_results", [])

    async with sem:
        start = time.monotonic()
        api_error: str | None = None
        actual_cards: list[dict] = []
        actual_answer = ""

        try:
            async with http_client.stream(
                "POST",
                f"{api_base}/api/search",
                json={"query": query},
                timeout=timeout,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()
                result = await parse_sse_stream(response)
                actual_cards = result.get("cards", [])
                actual_answer = result.get("answer", "")
        except httpx.TimeoutException:
            api_error = "timeout"
        except httpx.HTTPStatusError as e:
            api_error = f"http_{e.response.status_code}"
        except json.JSONDecodeError:
            api_error = "parse_error"
        except RuntimeError as e:
            api_error = str(e)[:100]
        except Exception as e:
            api_error = f"error: {str(e)[:80]}"

        latency_ms = int((time.monotonic() - start) * 1000)

        if api_error:
            result_obj = JudgeResult(
                case_id=case_id,
                score=0.0,
                passed=False,
                reasoning=f"API call failed: {api_error}",
                failure_category="api_error",
            )
        else:
            result_obj = await judge_case(
                case_id=case_id,
                query=query,
                intent=intent,
                category=category,
                difficulty=difficulty,
                expected_results=expected_results,
                negative_results=negative_results,
                actual_cards=actual_cards,
                actual_answer=actual_answer,
                client=judge_client,
            )

        return CaseResult(
            case_id=case_id,
            query=query,
            category=category,
            difficulty=difficulty,
            score=result_obj.score,
            passed=result_obj.passed,
            reasoning=result_obj.reasoning,
            failure_category=result_obj.failure_category,
            actual_cards_count=len(actual_cards),
            actual_answer_snippet=actual_answer[:200],
            api_error=api_error,
            latency_ms=latency_ms,
        )


def build_category_summary(cases: list[CaseResult]) -> dict[str, dict]:
    by_cat: dict[str, list[CaseResult]] = {}
    for c in cases:
        by_cat.setdefault(c.category, []).append(c)
    summary = {}
    for cat, cat_cases in sorted(by_cat.items()):
        passed = sum(1 for c in cat_cases if c.passed)
        summary[cat] = {
            "total": len(cat_cases),
            "passed": passed,
            "pct": round(passed / len(cat_cases) * 100, 1),
        }
    return summary


def print_summary(run: EvalRun) -> None:
    console.print()
    color = "green" if run.correctness_pct >= 90 else ("yellow" if run.correctness_pct >= 70 else "red")
    console.print(
        f"[bold {color}]Overall correctness: {run.correctness_pct:.1f}% "
        f"({run.passed}/{run.total} passed)[/bold {color}]"
    )

    # Category breakdown
    cat_table = Table(title="Results by Category", show_lines=True)
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Passed", justify="right")
    cat_table.add_column("Total", justify="right")
    cat_table.add_column("Pct", justify="right")

    for cat, stats in run.summary_by_category.items():
        pct_color = "green" if stats["pct"] >= 70 else "red"
        cat_table.add_row(cat, str(stats["passed"]), str(stats["total"]), f"[{pct_color}]{stats['pct']}%[/{pct_color}]")

    console.print(cat_table)

    # Failed cases
    failed = [c for c in run.cases if not c.passed]
    if failed:
        fail_table = Table(title="Failed Cases", show_lines=True)
        fail_table.add_column("ID", style="cyan", width=8)
        fail_table.add_column("Query", width=30)
        fail_table.add_column("Score", justify="right", width=6)
        fail_table.add_column("Failure", width=25)
        fail_table.add_column("API Error", width=15)

        for c in sorted(failed, key=lambda x: x.score):
            fail_table.add_row(
                c.case_id,
                c.query[:28] + ".." if len(c.query) > 30 else c.query,
                f"{c.score:.2f}",
                c.failure_category or "",
                c.api_error or "",
            )
        console.print(fail_table)


async def run_eval(
    dataset_path: str = DEFAULT_DATASET,
    api_base: str = DEFAULT_API,
    output_dir: str = DEFAULT_OUT,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout_per_case: float = DEFAULT_TIMEOUT,
) -> EvalRun:
    # Load dataset
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        sys.exit(1)

    with open(dataset_file) as f:
        dataset: list[dict] = json.load(f)

    console.print(f"Loaded [bold]{len(dataset)}[/bold] test cases from {dataset_path}")

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        console.print("[red]ANTHROPIC_API_KEY not set[/red]")
        sys.exit(1)

    judge_client = anthropic.AsyncAnthropic(api_key=anthropic_key)

    async with httpx.AsyncClient() as http_client:
        # Pre-flight health check
        console.print(f"Checking server at {api_base}...")
        healthy = await check_server_health(api_base, http_client)
        if not healthy:
            console.print(f"[red]Server not reachable at {api_base}. Start it with 'make dev'.[/red]")
            sys.exit(1)
        console.print("[green]Server is healthy[/green]")

        sem = asyncio.Semaphore(concurrency)
        tasks = [
            run_one_case(case, api_base, http_client, judge_client, sem, timeout_per_case)
            for case in dataset
        ]

        console.print(f"Running {len(tasks)} cases (concurrency={concurrency})...")
        cases: list[CaseResult] = []
        with console.status("[bold green]Evaluating...") as status:
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                result = await coro
                cases.append(result)
                icon = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
                status.update(f"[bold green]{i+1}/{len(tasks)}[/bold green] {result.case_id} — {icon} ({result.score:.2f})")

    # Sort by case_id for consistent output
    cases.sort(key=lambda c: c.case_id)

    passed = sum(1 for c in cases if c.passed)
    total = len(cases)
    correctness_pct = round(passed / total * 100, 1) if total else 0.0

    run = EvalRun(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total=total,
        passed=passed,
        failed=total - passed,
        correctness_pct=correctness_pct,
        cases=cases,
        summary_by_category=build_category_summary(cases),
    )

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"eval_{ts}.json"

    with open(out_file, "w") as f:
        json.dump(asdict(run), f, indent=2)

    console.print(f"\nResults saved to [bold]{out_file}[/bold]")
    print_summary(run)

    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pokemon card search evaluation")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    asyncio.run(run_eval(
        dataset_path=args.dataset,
        api_base=args.api,
        output_dir=args.out,
        concurrency=args.concurrency,
        timeout_per_case=args.timeout,
    ))


if __name__ == "__main__":
    main()
