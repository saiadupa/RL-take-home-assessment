import asyncio
import json
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, TypedDict

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    """
    try:
        namespace: dict[str, Any] = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


CHURN_PROMPT = """
You are an ML engineer working on churn prediction for a subscription product.

You have a CSV file called `churn_data.csv` in the current working directory.
Use the `python_expression` tool to write and run Python code that:

1. LOAD THE DATA
   - Load `churn_data.csv` into a pandas DataFrame.
   - Parse the `signup_date` column as a date.

2. REMOVE LEAKY AND INVALID FEATURES
   - Treat the following columns as LEAKY and DO NOT use them as features:
       - `churned` (this is the label)
       - `future_revenue_3m` (depends on future outcome)
       - `last_cancellation_reason` (only defined after churn)
       - `m4_usage`, `m5_usage`, `m6_usage` (contain post-3-month behavior)
   - You may still use `churned` as the LABEL, but it must not be counted as a feature.
   - You do NOT need to train a model for this task; we only care about the statistics.

3. FILTER TO A REALISTIC COHORT
   - Only keep rows where:
       - `tenure_months` is at least 1 (>= 1), and
       - `tenure_months` is at most 24 (<= 24).
   - Perform all subsequent analysis on this FILTERED dataset.

4. COMPUTE AGGREGATE CHURN STATISTICS
   Using the FILTERED dataset:

   (a) `num_rows`:
       - The number of rows remaining after filtering by tenure.

   (b) `overall_churn_rate`:
       - The percentage of churned users in the filtered dataset,
         as a number between 0 and 100, rounded to 4 decimal places.

   (c) `per_plan_churn_rate`:
       - A mapping from each `plan_type` to its churn rate in the filtered dataset.
       - Only include plans that appear at least 20 times in the filtered data.
       - The churn rate for each plan is defined as:
           100 * mean of `churned` for that plan in the filtered data,
           rounded to 4 decimal places.
       - Represent this as a JSON object/dict, e.g.:
           {
             "basic": 17.1234,
             "pro": 21.0000,
             "enterprise": 19.5000
           }

   (d) `high_churn_plan` and `low_churn_plan`:
       - Consider ONLY the plans included in `per_plan_churn_rate`
         (i.e., those with at least 20 rows).
       - `high_churn_plan`: the plan_type with the HIGHEST churn rate.
       - `low_churn_plan`: the plan_type with the LOWEST churn rate.
       - If multiple plans tie, any one of the tied plans is acceptable.

5. PREPARE THE LEAKY COLUMNS LIST
   - `leaky_cols_removed`: a list of the column names you removed that are considered
     leaky or invalid:
       - `churned`
       - `future_revenue_3m`
       - `last_cancellation_reason`
       - `m4_usage`
       - `m5_usage`
       - `m6_usage`
   - Sort this list alphabetically before submitting.

6. SUBMIT YOUR ANSWER
   - Once you have the correct values, use the `submit_answer` tool to submit
     a JSON-serializable object with the following structure (keys MUST match exactly):

     {
       "num_rows": <int>,
       "overall_churn_rate": <float>,
       "per_plan_churn_rate": {
         "<plan_name>": <float>,   // only plans with at least 20 rows, rounded to 4 decimals
         ...
       },
       "high_churn_plan": "<plan_name>",
       "low_churn_plan": "<plan_name>",
       "leaky_cols_removed": [<str>, <str>, ...]  // sorted list of column names
     }

   - The grader will:
       - recompute the filtered dataset from `churn_data.csv`,
       - recompute the correct aggregate statistics,
       - verify your per-plan churn rates,
       - check that your `high_churn_plan` and `low_churn_plan` are correct,
       - ensure that your `leaky_cols_removed` list is complete and correct.

Use `python_expression` to do ALL the data work and compute all statistics.
Use `submit_answer` exactly once when you’re confident in your result.

IMPORTANT:
- Do NOT answer in natural language only.
- You MUST use the `python_expression` tool to inspect and process the dataset.
- You MUST finish by calling the `submit_answer` tool exactly once with the JSON described above.
- If you do not call `submit_answer`, the task will be graded as a failure.
"""


def grade_churn_task(result: Any) -> bool:
    import json as _json
    import math as _math
    import pandas as _pd

    try:
        if isinstance(result, str):
            answer = _json.loads(result)
        elif isinstance(result, dict):
            answer = result
        else:
            print("Result has wrong type:", type(result))
            return False

        num_rows_sub = int(answer["num_rows"])
        overall_churn_rate_sub = float(answer["overall_churn_rate"])
        per_plan_sub = dict(answer["per_plan_churn_rate"])
        high_plan_sub = str(answer["high_churn_plan"])
        low_plan_sub = str(answer["low_churn_plan"])
        leaky_cols_removed_sub = sorted(list(answer["leaky_cols_removed"]))
    except Exception as e:
        print("Failed to parse submitted answer:", e)
        return False

    try:
        df = _pd.read_csv("churn_data.csv", parse_dates=["signup_date"])
    except Exception as e:
        print("Failed to load churn_data.csv:", e)
        return False

    if "tenure_months" not in df.columns:
        print("tenure_months column missing from dataset.")
        return False

    df_f = df[(df["tenure_months"] >= 1) & (df["tenure_months"] <= 24)].copy()

    n = len(df_f)
    if n == 0:
        print("Filtered dataset is empty.")
        return False

    num_rows_gold = n

    overall_churn_rate_gold = float(round(100.0 * df_f["churned"].mean(), 4))

    if "plan_type" not in df_f.columns:
        print("plan_type column missing from filtered dataset.")
        return False

    per_plan_gold: dict[str, float] = {}
    for plan in df_f["plan_type"].unique():
        df_p = df_f[df_f["plan_type"] == plan]
        if len(df_p) >= 20:
            rate = float(round(100.0 * df_p["churned"].mean(), 4))
            per_plan_gold[str(plan)] = rate

    if not per_plan_gold:
        print("No plans with at least 20 rows in filtered dataset.")
        return False

    max_rate = max(per_plan_gold.values())
    min_rate = min(per_plan_gold.values())

    high_plans_gold = {p for p, r in per_plan_gold.items() if r == max_rate}
    low_plans_gold = {p for p, r in per_plan_gold.items() if r == min_rate}

    if num_rows_sub != num_rows_gold:
        print("num_rows mismatch:", num_rows_sub, "vs", num_rows_gold)
        return False

    if not _math.isclose(
        overall_churn_rate_sub,
        overall_churn_rate_gold,
        rel_tol=1e-5,
        abs_tol=1e-3,
    ):
        print(
            "overall_churn_rate mismatch:",
            overall_churn_rate_sub,
            "vs",
            overall_churn_rate_gold,
        )
        return False

    for plan, gold_rate in per_plan_gold.items():
        if plan not in per_plan_sub:
            print(f"Missing plan in per_plan_churn_rate: {plan}")
            return False
        sub_rate = float(per_plan_sub[plan])
        if not _math.isclose(sub_rate, gold_rate, rel_tol=1e-5, abs_tol=1e-3):
            print(
                f"Churn rate mismatch for {plan}:",
                sub_rate,
                "vs",
                gold_rate,
            )
            return False

    if high_plan_sub not in high_plans_gold:
        print(
            "high_churn_plan mismatch:",
            high_plan_sub,
            "not in allowed set",
            high_plans_gold,
        )
        return False

    if low_plan_sub not in low_plans_gold:
        print(
            "low_churn_plan mismatch:",
            low_plan_sub,
            "not in allowed set",
            low_plans_gold,
        )
        return False

    forbidden_cols = [
        "churned",
        "future_revenue_3m",
        "last_cancellation_reason",
        "m4_usage",
        "m5_usage",
        "m6_usage",
    ]
    leaky_cols_removed_gold = sorted(forbidden_cols)

    if leaky_cols_removed_sub != leaky_cols_removed_gold:
        print(
            "Leaky cols mismatch:",
            leaky_cols_removed_sub,
            "vs",
            leaky_cols_removed_gold,
        )
        return False

    return True


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 10,
    model: str = "claude-3-5-haiku-latest",
    verbose: bool = False,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.

    Args:
        prompt: The initial prompt for the agent
        tools: List of tool definitions for Anthropic API
        tool_handlers: Dictionary mapping tool names to their handler functions
        max_steps: Maximum number of steps before stopping (default 5)
        model: The Anthropic model to use
        verbose: Whether to print detailed output (default True)

    Returns:
        The submitted answer if submit_answer was called, otherwise None
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model,
            max_tokens=500,
            tools=tools,
            tool_choice={"type": "any"},
            messages=messages,
        )

        has_tool_use = False
        tool_results: list[dict[str, Any]] = []
        submitted_answer: Any | None = None

        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    if tool_name == "python_expression":
                        expr: str
                        if isinstance(tool_input, dict):
                            if "expression" in tool_input and isinstance(
                                tool_input["expression"], str
                            ):
                                expr = tool_input["expression"]
                            else:
                                str_vals = [
                                    v for v in tool_input.values() if isinstance(v, str)
                                ]
                                expr = str_vals[0] if str_vals else ""
                        else:
                            expr = str(tool_input)

                        if verbose:
                            print("\nInput:")
                            print("```")
                            for line in expr.split("\n"):
                                print(line)
                            print("```")

                        if not expr.strip():
                            result = {
                                "result": None,
                                "error": "Empty or invalid expression",
                            }
                        else:
                            result = handler(expr)

                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result)
                            print("```")

                    elif tool_name == "submit_answer":
                        if not (isinstance(tool_input, dict) and "answer" in tool_input):
                            result = {"answer": None, "submitted": False}
                        else:
                            result = handler(tool_input["answer"])
                            submitted_answer = result["answer"]
                    else:
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print("\nNo answer submitted by agent.")
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    grade: Callable[[Any], bool],
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=10,
        verbose=False,
    )

    success = grade(result)

    if success:
        print(f"✓ Run {run_id}: SUCCESS - Got {result}")
    else:
        print(f"✗ Run {run_id}: FAILURE - Got {result}")

    return run_id, success, result


async def main(concurrent: bool = False):
    tools: list[ToolUnionParam] = [
        {
            "name": "python_expression",
            "description": "Evaluates a Python expression in a sandboxed namespace.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python code to pass to exec(). "
                        "Use print() to output something. Returns stdout.",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "description": "The final answer to submit "
                        "(e.g. JSON-encoded string or dict)."
                    }
                },
                "required": ["answer"],
            },
        },
    ]

    tool_handlers: dict[str, Callable[..., Any]] = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }

    num_runs = 10
    prompt = CHURN_PROMPT

    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print("=" * 60)

    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            grade=grade_churn_task,
            verbose=False,
        )
        for i in range(num_runs)
    ]

    if concurrent:
        results: list[tuple[int, bool, Any]] = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
    else:
        results: list[tuple[int, bool, Any]] = []
        for task in tasks:
            result = await task
            results.append(result)

    print("\nDEBUG FINAL results list:", results)

    successes = sum(1 for _, success, _ in results if success)
    failures = num_runs - successes
    pass_rate = (successes / num_runs) * 100 if num_runs > 0 else 0.0

    print("\n================ SUMMARY ================")
    print("  SUCCESSES:", successes)
    print("  FAILURES:", failures)
    print("  PASS_RATE:", f"{pass_rate:.1f}%")
    print("=========================================")


if __name__ == "__main__":
    asyncio.run(main(concurrent=False))
