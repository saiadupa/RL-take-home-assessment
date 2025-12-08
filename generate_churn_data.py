import numpy as np
import pandas as pd


def main():
    np.random.seed(42)
    n = 600  # number of rows

    # Basic columns
    customer_id = np.arange(1000, 1000 + n)

    # Signup dates between 2021-01-01 and 2021-09-01
    base_date = np.datetime64("2021-01-01")
    signup_offset_days = np.random.randint(0, 244, size=n)  # up to ~8 months
    signup_date = base_date + signup_offset_days.astype("timedelta64[D]")

    country = np.random.choice(["US", "DE", "FR"], size=n, p=[0.4, 0.4, 0.2])
    plan_type = np.random.choice(
        ["basic", "pro", "enterprise"], size=n, p=[0.5, 0.35, 0.15]
    )
    device_count = np.random.randint(1, 6, size=n)

    # Usage features for months 1–6
    m1_usage = np.random.gamma(shape=2.0, scale=10.0, size=n)
    m2_usage = np.random.gamma(shape=2.0, scale=10.0, size=n)
    m3_usage = np.random.gamma(shape=2.0, scale=10.0, size=n)
    m4_usage = np.random.gamma(shape=2.0, scale=10.0, size=n)
    m5_usage = np.random.gamma(shape=2.0, scale=10.0, size=n)
    m6_usage = np.random.gamma(shape=2.0, scale=10.0, size=n)

    # Latent churn probability: depends on early usage, country, plan, device_count
    base_logit = -0.5
    logit = (
        base_logit
        - 0.03 * m1_usage
        - 0.02 * m2_usage
        - 0.01 * m3_usage
        + 0.2 * (country == "US")  # slightly more churn in US
        + 0.1 * (plan_type == "basic")
        + 0.05 * (device_count <= 2)
    )

    prob_churn = 1 / (1 + np.exp(-logit))
    churned = (np.random.rand(n) < prob_churn).astype(int)

    # Leaky columns:
    # future_revenue_3m: roughly 3 * monthly revenue but zero for churners
    base_revenue = np.random.uniform(20, 80, size=n)
    future_revenue_3m = np.where(
        churned == 1, 0.0, base_revenue * 3 + np.random.normal(0, 10, size=n)
    )

    # last_cancellation_reason: only defined if churned == 1
    reasons = ["too_expensive", "missing_features", "switched_competitor", "other"]
    last_cancellation_reason = np.array([""] * n, dtype=object)
    mask_churn = churned == 1
    last_cancellation_reason[mask_churn] = np.random.choice(
        reasons, size=mask_churn.sum()
    )

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "signup_date": signup_date,
            "country": country,
            "plan_type": plan_type,
            "device_count": device_count,
            "m1_usage": m1_usage,
            "m2_usage": m2_usage,
            "m3_usage": m3_usage,
            "m4_usage": m4_usage,
            "m5_usage": m5_usage,
            "m6_usage": m6_usage,
            "churned": churned,
            "future_revenue_3m": future_revenue_3m,
            "last_cancellation_reason": last_cancellation_reason,
        }
    )

    df.to_csv("churn_data.csv", index=False)
    print("Saved churn_data.csv with shape", df.shape)


if __name__ == "__main__":
    main()
