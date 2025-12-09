
# Reinforcement Learning Tools Project  
### Using Anthropic Messages API with Tool Use

This project demonstrates how a model interacts with tools (Python execution + answer submission) and how performance changes when the task difficulty changes.

The goal was to:

- Understand how an AI agent uses external tools.
- See how task complexity affects success rate.
- Run repeated evaluations (episodes) and calculate pass rate.

---

## Project Files

| File Name | Description |
|----------|-------------|
| `main.py` | Original file from starter zip. Simple math task with full automation. |
| `pass.py` | Modified version with a churn dataset task. The task is easier and achieves **100% success rate** in multiple runs. |
| `fail.py` | Hard version of the churn analysis task. Model fails in all 10 attempts (0% success rate). |
| `generate_churn_data.py` | Script that creates the dataset `churn_data.csv` used in churn-related tasks. |
| `churn_data.csv` | Auto-generated dataset used by `pass.py` and `fail.py`. |
| `requirements.txt` | Python dependencies. |
| Other helper files | Came from the original zip and support structure. |

---

## Installation & Setup

1. Clone or extract the project folder.
2. Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
````

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Generate the dataset:

```bash
python generate_churn_data.py
```

This creates:

```
churn_data.csv
```

---

## How the Agent Works

All main scripts (`main.py`, `pass.py`, `fail.py`) use the same logic:

### Step 1 - Give a Prompt

A task description is sent to the model.

### Step 2 - Model Calls a Tool

The model decides whether to use:

| Tool                | Purpose                                                     |
| ------------------- | ----------------------------------------------------------- |
| `python_expression` | Runs Python code using `exec()` and returns printed output. |
| `submit_answer`     | Sends final structured answer for grading.                  |

### Step 3 - Grader Checks the Result

* If the model answers correctly → **PASS**
* If result is wrong or missing → **FAIL**

### Step 4 - Repeat Multiple Times

Each script runs **10 iterations** to test stability.

---

##  File Behavior & Results

### main.py: Baseline Example (Simple Task)

* The model solves a single math expression.
* Extremely easy and predictable.

#### Result: 100% Pass Rate

`images/run_main.png`

```
Passed: 10/10  
Pass Rate: 100.0%
```

---

### pass.py: Simple Churn Dataset Task

* Reads `churn_data.csv`
* Removes "leaky" columns:

  ```
  churned, future_revenue_3m, last_cancellation_reason,
  m4_usage, m5_usage, m6_usage
  ```
* Computes:

  * Total rows
  * Overall churn rate
* Submits structured JSON.

#### Result: **100% Pass Rate**

*Screenshot Example (`images/run_pass.png`)*

```
Passed: 10/10  
Pass Rate: 100.0%
```

This shows the agent can handle slightly more structured tasks.

---

###  fail.py: High-Difficulty Churn Task

This final test increases complexity:

| Requirement                        | Difficulty   |
| ---------------------------------- | ------------ |
| Load dataset                       |  Easy      |
| Remove leak columns                |  Easy      |
| Filter rows using `tenure_months`  |  Medium    |
| Compute churn per plan             |  Hard      |
| Pick highest and lowest churn plan |  Very Hard |
| Submit structured JSON exactly     |  Strict    |

Because the grader is strict and the agent must follow many rules perfectly, it fails consistently.

#### Result: **0% Success**

*Screenshot Example (`images/run_fail.png`)*

```
Passed: 0/10  
Failed: 10/10  
Pass Rate: 0.0%
```

---

## Comparison

| Script    | Task Type                           | Difficulty | Expected Behavior | Result |
| --------- | ----------------------------------- | ---------- | ----------------- | ------ |
| `main.py` | Math Expression                     |  Easy     | Always correct    |  100% |
| `pass.py` | Light Data Processing               |  Medium  | Mostly correct    |  100% |
| `fail.py` | Full Data Analysis + JSON structure |  Hard | Model struggles   |  0%   |

---

## How to Run Each File

```bash
python main.py
python pass.py
python fail.py
```


---

### Author

Project by **Nithin Sai Adupa**

---
