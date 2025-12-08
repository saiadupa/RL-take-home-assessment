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
   - You do NOT need to train a model for this task; we only care about the split and label statistics.

3. CHOOSE A TIME CUTOFF DATE UNDER MULTIPLE CONSTRAINTS
   - You must choose a single calendar date `cutoff_date` between:
       - 2021-03-15 (inclusive) and 2021-08-15 (inclusive).
   - Use `signup_date` as the time column.
   - Define:
       - Training set: all rows where `signup_date` is STRICTLY BEFORE `cutoff_date`.
       - Validation set: all rows where `signup_date` is ON OR AFTER `cutoff_date`.

   Your chosen `cutoff_date` must satisfy ALL of the following:

   (a) Both train and validation sets are non-empty.

   (b) The training set size is between 45% and 75% of the full dataset size.

   (c) The overall churn rate in the TRAIN set is between 15% and 30%.

   (d) The overall churn rate in the VALIDATION set is between 15% and 30%.

   (e) For each plan_type (\"basic\", \"pro\", \"enterprise\") that appears at least 30 times
       in BOTH train and validation splits:
         - The churn rate difference between TRAIN and VALIDATION for that plan_type
           must be at most 3 percentage points (absolute difference <= 3.0).

   - The churn rate is defined as:
       churn_rate = 100 * (mean of the `churned` column) in that split,
       rounded to 4 decimal places.

   - There may be many valid cutoff dates. You must search over possible dates
     programmatically (e.g. looping over dates, or using quantiles) and pick one
     that satisfies all the constraints above.

4. COMPUTE THE FOLLOWING SUMMARY STATISTICS
   After you have:
     - removed the leaky columns listed above, and
     - chosen a valid `cutoff_date` that satisfies all constraints,

   compute and record:

   - `cutoff_date`: the date you used as the split threshold, as a string in YYYY-MM-DD format.
   - `train_rows`: the number of rows in the TRAIN set.
   - `val_rows`: the number of rows in the VALIDATION set.
   - `train_churn_rate`: the percentage of churned users in the TRAIN set,
     as a number between 0 and 100, rounded to 4 decimal places.
   - `val_churn_rate`: the percentage of churned users in the VALIDATION set,
     as a number between 0 and 100, rounded to 4 decimal places.
   - `leaky_cols_removed`: a list of the column names you removed that are considered
     leaky or invalid (`churned`, `future_revenue_3m`, `last_cancellation_reason`,
     `m4_usage`, `m5_usage`, `m6_usage`). Sort this list alphabetically before submitting.

5. SUBMIT YOUR ANSWER
   - Once you have the correct values, use the `submit_answer` tool to submit a JSON string
     with the following structure (keys MUST match exactly):

     {
       "cutoff_date": "YYYY-MM-DD",
       "train_rows": <int>,
       "val_rows": <int>,
       "train_churn_rate": <float>,
       "val_churn_rate": <float>,
       "leaky_cols_removed": [<str>, <str>, ...]  // sorted list of column names
     }

   - The grader will:
       - recompute the correct values from `churn_data.csv` using your `cutoff_date`
       - verify that your split obeys all constraints (including per-plan_type constraints)
       - check your statistics against the true values
       - ensure that your `leaky_cols_removed` list is complete and correct.

Use `python_expression` to do all the data work and search over candidate cutoff dates.
Use `submit_answer` exactly once when you’re confident in your result.
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

        cutoff_date_str = str(answer["cutoff_date"])
        train_rows_sub = int(answer["train_rows"])
        val_rows_sub = int(answer["val_rows"])
        train_churn_rate_sub = float(answer["train_churn_rate"])
        val_churn_rate_sub = float(answer["val_churn_rate"])
        leaky_cols_removed_sub = sorted(list(answer["leaky_cols_removed"]))
    except Exception as e:
        print("Failed to parse submitted answer:", e)
        return False

    try:
        df = _pd.read_csv("churn_data.csv", parse_dates=["signup_date"])
    except Exception as e:
        print("Failed to load churn_data.csv:", e)
        return False

    n = len(df)
    if n == 0:
        print("Dataset is empty.")
        return False

    try:
        cutoff = _pd.to_datetime(cutoff_date_str)
    except Exception as e:
        print("Failed to parse cutoff_date:", e)
        return False

    lower_bound = _pd.Timestamp("2021-03-15")
    upper_bound = _pd.Timestamp("2021-08-15")

    if not (lower_bound <= cutoff <= upper_bound):
        print("Cutoff date out of allowed range:", cutoff)
        return False

    train_mask = df["signup_date"] < cutoff
    val_mask = df["signup_date"] >= cutoff

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()

    train_rows_gold = len(df_train)
    val_rows_gold = len(df_val)

    if train_rows_gold == 0 or val_rows_gold == 0:
        print("Degenerate split: train or val is empty.")
        return False

    frac_train = train_rows_gold / n
    if not (0.45 <= frac_train <= 0.75):
        print("Train fraction out of range:", frac_train)
        return False

    train_churn_rate_gold = float(round(100.0 * df_train["churned"].mean(), 4))
    val_churn_rate_gold = float(round(100.0 * df_val["churned"].mean(), 4))

    if not (15.0 <= train_churn_rate_gold <= 30.0):
        print("Train churn rate out of allowed range:", train_churn_rate_gold)
        return False

    if not (15.0 <= val_churn_rate_gold <= 30.0):
        print("Validation churn rate out of allowed range:", val_churn_rate_gold)
        return False

    max_diff_allowed = 3.0
    min_count = 30

    if "plan_type" not in df.columns:
        print("plan_type column missing from dataset.")
        return False

    for plan in df["plan_type"].unique():
        df_tr_p = df_train[df_train["plan_type"] == plan]
        df_va_p = df_val[df_val["plan_type"] == plan]

        if len(df_tr_p) >= min_count and len(df_va_p) >= min_count:
            tr_rate = float(round(100.0 * df_tr_p["churned"].mean(), 4))
            va_rate = float(round(100.0 * df_va_p["churned"].mean(), 4))
            diff = abs(tr_rate - va_rate)
            if diff > max_diff_allowed:
                print(
                    f"Per-plan churn mismatch for {plan}: "
                    f"train={tr_rate}, val={va_rate}, diff={diff}"
                )
                return False

    if train_rows_sub != train_rows_gold:
        print("Train rows mismatch:", train_rows_sub, "vs", train_rows_gold)
        return False

    if val_rows_sub != val_rows_gold:
        print("Val rows mismatch:", val_rows_sub, "vs", val_rows_gold)
        return False

    if not _math.isclose(
        train_churn_rate_sub,
        train_churn_rate_gold,
        rel_tol=1e-5,
        abs_tol=1e-3,
    ):
        print(
            "Train churn rate mismatch:",
            train_churn_rate_sub,
            "vs",
            train_churn_rate_gold,
        )
        return False

    if not _math.isclose(
        val_churn_rate_sub,
        val_churn_rate_gold,
        rel_tol=1e-5,
        abs_tol=1e-3,
    ):
        print(
            "Val churn rate mismatch:",
            val_churn_rate_sub,
            "vs",
            val_churn_rate_gold,
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
    max_steps: int = 20,
    model: str = "claude-3-5-haiku-latest",
    verbose: bool = True,
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
            messages=messages,
        )

        has_tool_use = False
        tool_results: list[dict[str, Any]] = []
        submitted_answer: Any | None = None

        # Process the response
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
                            result = {"result": None, "error": "Empty or invalid expression"}
                        else:
                            result = handler(expr)

                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result)
                            print("```")

                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
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

        # If we have tool uses, add them to the conversation
        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            # If an answer was submitted, return it
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            # No tool use in response, ending loop.
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


async def main(concurrent: bool = True):
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

    # Run the test 10 times and track success rate
    num_runs = 3
    prompt = CHURN_PROMPT

    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print("=" * 60)

    # Create all test coroutines
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

    # Run concurrently or sequentially based on the flag
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

    # Count successes
    successes = sum(1 for _, success, _ in results)
    failures = num_runs - successes

    # Calculate and display pass rate
    pass_rate = (successes / num_runs) * 100
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {failures}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    # Set to True for concurrent execution, False for sequential execution
    asyncio.run(main(concurrent=False))
