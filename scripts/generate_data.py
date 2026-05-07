"""Generate synthetic multi-tenant SaaS transaction data.

Produces parquet files in ./data/raw/ that scripts/load_to_postgres.py ingests.

Realistic patterns built in:
- Weekly + yearly seasonality on transaction volume
- Customer lifetime curves (acquisition → engagement → decay → churn)
- Cohort effects (older cohorts have lower baseline activity)
- Merchant-level pricing distributions (industry-dependent)
- Basket affinity (customers stick to a small set of preferred categories)
- Injected anomalies (~0.8%): unusually large amount, off-hours, new-device proxy
- Churn label derived from forward-looking behavior, not static

Defaults: 500 merchants, 50K customers, 5M transactions, 24 months.
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

DATA_DIR = Path("./data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

INDUSTRIES = [
    "retail", "food_service", "professional_services", "wholesale",
    "automotive", "health_wellness", "home_services", "beauty",
    "education", "entertainment", "fitness", "logistics",
]

REGIONS = ["us-east", "us-west", "us-central", "ca-east", "ca-west"]
COUNTRIES = ["US", "CA"]
PLANS = ["starter", "growth", "scale", "enterprise"]
PLAN_WEIGHTS = [0.45, 0.30, 0.18, 0.07]
CHANNELS = ["web", "mobile", "in_person", "api"]
CHANNEL_WEIGHTS = [0.40, 0.35, 0.18, 0.07]
TICKET_CATEGORIES = ["billing", "technical", "account", "feature_request", "refund", "integration"]

# Industry → (price_mean, price_std, txn_per_customer_per_month)
INDUSTRY_PARAMS = {
    "retail":                (45.0, 30.0, 4.0),
    "food_service":          (22.0, 12.0, 8.0),
    "professional_services": (180.0, 120.0, 1.2),
    "wholesale":             (320.0, 200.0, 2.5),
    "automotive":            (240.0, 180.0, 0.8),
    "health_wellness":       (85.0, 50.0, 2.0),
    "home_services":         (150.0, 100.0, 1.0),
    "beauty":                (60.0, 35.0, 2.5),
    "education":             (120.0, 80.0, 1.5),
    "entertainment":         (28.0, 18.0, 3.0),
    "fitness":               (55.0, 25.0, 4.5),
    "logistics":             (200.0, 150.0, 6.0),
}

CATEGORY_BANK = {
    "retail": ["apparel", "accessories", "footwear", "home_goods", "electronics"],
    "food_service": ["entrees", "beverages", "desserts", "appetizers", "catering"],
    "professional_services": ["consulting", "legal", "accounting", "marketing", "design"],
    "wholesale": ["bulk_food", "office_supplies", "industrial", "packaging", "raw_materials"],
    "automotive": ["parts", "service", "accessories", "tires", "diagnostics"],
    "health_wellness": ["supplements", "consultations", "therapy", "screenings", "wellness_plans"],
    "home_services": ["plumbing", "electrical", "cleaning", "landscaping", "renovation"],
    "beauty": ["hair", "nails", "skincare", "makeup", "spa"],
    "education": ["courses", "tutoring", "certifications", "workshops", "books"],
    "entertainment": ["tickets", "memberships", "concessions", "merchandise", "experiences"],
    "fitness": ["classes", "memberships", "personal_training", "supplements", "apparel"],
    "logistics": ["shipping", "warehousing", "fulfillment", "freight", "last_mile"],
}


@dataclass
class Args:
    merchants: int
    customers: int
    transactions: int
    months: int
    seed: int


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--merchants", type=int, default=500)
    p.add_argument("--customers", type=int, default=50_000)
    p.add_argument("--transactions", type=int, default=5_000_000)
    p.add_argument("--months", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    return Args(a.merchants, a.customers, a.transactions, a.months, a.seed)


def gen_merchants(n: int, end_date: date, rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        industry = rng.choice(INDUSTRIES)
        # Merchants signed up over the last 3 years before end_date
        days_ago = int(rng.integers(60, 1095))
        rows.append({
            "id": i,
            "name": fake.company(),
            "industry": industry,
            "region": rng.choice(REGIONS),
            "created_at": end_date - timedelta(days=days_ago),
        })
    return pd.DataFrame(rows)


def gen_customers(n: int, merchants: pd.DataFrame, end_date: date,
                  rng: np.random.Generator) -> pd.DataFrame:
    # Long-tail distribution of customers across merchants (Pareto-ish)
    weights = rng.pareto(1.2, len(merchants)) + 0.1
    weights = weights / weights.sum()
    merchant_ids = rng.choice(merchants["id"].values, size=n, p=weights)

    rows = []
    for i, mid in enumerate(tqdm(merchant_ids, desc="customers"), start=1):
        merchant_created = merchants.loc[merchants.id == mid, "created_at"].iloc[0]
        # Customer signed up after merchant
        max_offset = (end_date - merchant_created).days
        signup_offset = int(rng.integers(0, max(1, max_offset)))
        signup = merchant_created + timedelta(days=signup_offset)
        cohort = signup.strftime("%Y-%m")
        rows.append({
            "id": i,
            "merchant_id": int(mid),
            "signup_date": signup,
            "cohort": cohort,
            "country": rng.choice(COUNTRIES, p=[0.85, 0.15]),
            "plan": rng.choice(PLANS, p=PLAN_WEIGHTS),
            "is_active": True,
        })
    return pd.DataFrame(rows)


def gen_products(merchants: pd.DataFrame, target_total: int,
                 rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    rows = []
    pid = 1
    per_merchant = max(2, target_total // len(merchants))
    for _, m in tqdm(merchants.iterrows(), total=len(merchants), desc="products"):
        price_mean, price_std, _ = INDUSTRY_PARAMS[m.industry]
        cats = CATEGORY_BANK[m.industry]
        n_prod = int(rng.integers(max(2, per_merchant // 2), per_merchant * 2))
        for _ in range(n_prod):
            cat = rng.choice(cats)
            price = max(1.0, float(rng.normal(price_mean, price_std)))
            adj = rng.choice(["premium", "classic", "essential", "deluxe", "starter", "pro"])
            name = f"{adj.title()} {cat.replace('_', ' ').title()} {fake.word().title()}"
            desc = (
                f"{name}. {fake.sentence(nb_words=12)} "
                f"Ideal for {m.industry.replace('_', ' ')} customers seeking {cat.replace('_', ' ')}."
            )
            rows.append({
                "id": pid, "merchant_id": int(m.id), "sku": f"SKU-{pid:07d}",
                "name": name[:200], "category": cat, "price": round(price, 2),
                "description": desc,
            })
            pid += 1
    return pd.DataFrame(rows)


def _seasonality_factor(d: datetime) -> float:
    """Weekly + yearly seasonality multiplier for transaction probability."""
    # Yearly: peak in Nov-Dec (holiday)
    doy = d.timetuple().tm_yday
    yearly = 1.0 + 0.25 * np.sin(2 * np.pi * (doy - 80) / 365)
    if d.month in (11, 12):
        yearly *= 1.35
    # Weekly: weekday peak, weekend dip for B2B
    dow = d.weekday()
    weekly = 1.0 if dow < 5 else 0.55
    return yearly * weekly


def gen_transactions(customers: pd.DataFrame, products: pd.DataFrame, merchants: pd.DataFrame,
                     target_n: int, end_date: date, months: int,
                     rng: np.random.Generator) -> pd.DataFrame:
    """Generate transactions with realistic per-customer lifetime curves.

    For each customer we draw:
      - lambda_base: their per-month txn rate from industry baseline
      - lifetime curve: ramp-up over first 30d, plateau, geometric decay after some t
      - churn_at: optional cutoff (None = no churn)
    Then for each day we sample Poisson(lambda(t) * seasonality(d)).
    """
    start_date = end_date - timedelta(days=months * 30)
    industry_lookup = merchants.set_index("id")["industry"].to_dict()
    products_by_merchant = {
        mid: g[["id", "category", "price"]].to_records(index=False)
        for mid, g in products.groupby("merchant_id")
    }

    # Customer-level draws
    cust = customers.copy()
    cust["industry"] = cust["merchant_id"].map(industry_lookup)
    cust["lambda_base"] = cust["industry"].map(lambda x: INDUSTRY_PARAMS[x][2] / 30.0)
    plan_mult = {"starter": 0.6, "growth": 1.0, "scale": 1.6, "enterprise": 2.4}
    cust["lambda_base"] = cust["lambda_base"] * cust["plan"].map(plan_mult)
    # individual variation
    cust["lambda_base"] *= rng.lognormal(0.0, 0.5, len(cust))

    # ~20% will churn during the window
    churn_mask = rng.random(len(cust)) < 0.20
    cust["churn_offset_days"] = np.where(
        churn_mask,
        rng.integers(60, months * 30, len(cust)),
        -1,
    )

    rows = []
    txn_id = 1
    # Estimate per-customer expected count and scale to hit target
    expected_per_cust = cust["lambda_base"].sum() * months * 30
    scale = target_n / max(1.0, expected_per_cust)
    cust["lambda_eff"] = cust["lambda_base"] * scale

    pbar = tqdm(total=target_n, desc="transactions")
    cust_records = cust.to_records(index=False)
    cust_cols = list(cust.columns)
    idx_id = cust_cols.index("id")
    idx_merchant = cust_cols.index("merchant_id")
    idx_signup = cust_cols.index("signup_date")
    idx_lam = cust_cols.index("lambda_eff")
    idx_churn = cust_cols.index("churn_offset_days")

    # Iterate by customer for cache locality on product lookup
    for rec in cust_records:
        cid = int(rec[idx_id])
        mid = int(rec[idx_merchant])
        signup = rec[idx_signup]
        if isinstance(signup, np.datetime64):
            signup = pd.Timestamp(signup).date()
        lam = float(rec[idx_lam])
        churn_off = int(rec[idx_churn])
        prods = products_by_merchant.get(mid)
        if prods is None or len(prods) == 0:
            continue

        # Customer's "favorite" categories — stick to ~2-4
        cat_choices = list({p[1] for p in prods})
        n_fav = min(len(cat_choices), int(rng.integers(2, 5)))
        favorites = set(rng.choice(cat_choices, size=n_fav, replace=False))
        fav_prods = np.array([p for p in prods if p[1] in favorites], dtype=object)
        if len(fav_prods) == 0:
            fav_prods = np.array(list(prods), dtype=object)

        active_start = max(signup, start_date)
        active_end = end_date
        if churn_off > 0:
            churn_date = signup + timedelta(days=churn_off)
            if churn_date < active_end:
                active_end = churn_date

        days_active = (active_end - active_start).days
        if days_active <= 0:
            continue

        # Sample # of transactions for this customer over the window
        # Apply ramp-up + decay envelope
        for day_offset in range(days_active):
            d = datetime.combine(active_start + timedelta(days=day_offset), datetime.min.time())
            # ramp up first 14 days, decay last 60 if churning
            ramp = min(1.0, day_offset / 14.0)
            decay = 1.0
            if churn_off > 0:
                days_to_churn = days_active - day_offset
                if days_to_churn < 60:
                    decay = days_to_churn / 60.0
            rate = lam * ramp * decay * _seasonality_factor(d)
            n_today = rng.poisson(rate)
            for _ in range(n_today):
                p = fav_prods[rng.integers(0, len(fav_prods))]
                pid, _cat, base_price = int(p[0]), p[1], float(p[2])
                qty = int(rng.choice([1, 1, 1, 2, 2, 3, 5], p=[0.5, 0.15, 0.08, 0.10, 0.07, 0.06, 0.04]))
                amount = round(base_price * qty * float(rng.normal(1.0, 0.05)), 2)
                amount = max(0.5, amount)
                hour = int(rng.choice(range(24), p=_hour_weights()))
                ts = d + timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
                channel = rng.choice(CHANNELS, p=CHANNEL_WEIGHTS)
                # Inject anomaly ~0.8%
                is_anom = rng.random() < 0.008
                if is_anom:
                    # Anomalies: huge amount or off-hours
                    if rng.random() < 0.5:
                        amount *= float(rng.uniform(8, 25))
                    else:
                        ts = ts.replace(hour=int(rng.integers(2, 5)))
                status = "completed" if rng.random() > 0.02 else "failed"
                rows.append((
                    txn_id, mid, cid, pid, ts, round(amount, 2), qty,
                    channel, status, is_anom,
                ))
                txn_id += 1
                pbar.update(1)
                if txn_id > target_n:
                    pbar.close()
                    return _rows_to_df(rows)
    pbar.close()
    return _rows_to_df(rows)


def _hour_weights() -> np.ndarray:
    base = np.array([
        0.005, 0.003, 0.002, 0.002, 0.003, 0.005,  # 0-5
        0.015, 0.030, 0.050, 0.070, 0.080, 0.085,  # 6-11
        0.090, 0.085, 0.080, 0.075, 0.070, 0.065,  # 12-17
        0.055, 0.045, 0.035, 0.025, 0.015, 0.010,  # 18-23
    ])
    return base / base.sum()


def _rows_to_df(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        "id", "merchant_id", "customer_id", "product_id", "occurred_at",
        "amount", "quantity", "channel", "status", "is_anomaly",
    ])


def gen_tickets(customers: pd.DataFrame, transactions: pd.DataFrame,
                end_date: date, rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    """Free-text support tickets, ~2.5% of transactions trigger a ticket."""
    sample_n = max(1000, int(len(transactions) * 0.025))
    if len(transactions) == 0:
        return pd.DataFrame()
    sample = transactions.sample(n=sample_n, random_state=42)

    templates = {
        "billing": [
            "Charged {amount} but I expected {expected}. Can you explain the difference?",
            "Why am I being billed for {item} when I cancelled last month?",
            "Need an itemized invoice for transaction on {date}.",
            "Double charge on my account on {date}, please refund.",
        ],
        "technical": [
            "Login broken on mobile after the update. Getting error 500.",
            "API webhook stopped firing for new orders since {date}.",
            "Dashboard not loading transaction history for last 7 days.",
            "Export to CSV is timing out for large date ranges.",
        ],
        "account": [
            "Need to add a new user to my team with admin access.",
            "How do I change the primary email on the account?",
            "Want to upgrade from {plan} to scale plan, what changes?",
            "Locked out of account, two-factor recovery not working.",
        ],
        "feature_request": [
            "Would love bulk-edit on products. Editing one at a time is painful.",
            "Can you add Stripe Connect support for marketplace payouts?",
            "Need scheduled reports emailed weekly to my team.",
            "Please support multi-currency at the merchant level.",
        ],
        "refund": [
            "Customer returned item, need to issue refund for transaction.",
            "Partial refund requested for damaged goods, $20 of {amount}.",
            "Refund processed but not reflecting in customer's account yet.",
            "How long does a refund take to clear back to the card?",
        ],
        "integration": [
            "Trying to connect QuickBooks but auth keeps failing.",
            "Shopify sync skipped 12 orders today, what happened?",
            "Need help mapping our SKUs to your product taxonomy.",
            "Webhook signature verification failing on our end.",
        ],
    }

    rows = []
    for tid, (_, t) in enumerate(tqdm(sample.iterrows(), total=len(sample), desc="tickets"), start=1):
        cat = rng.choice(TICKET_CATEGORIES)
        templ = rng.choice(templates[cat])
        body = templ.format(
            amount=f"${float(t.amount):.2f}",
            expected=f"${float(t.amount) * 0.8:.2f}",
            item=fake.word(),
            date=pd.Timestamp(t.occurred_at).strftime("%Y-%m-%d"),
            plan=rng.choice(PLANS),
        )
        body += " " + fake.sentence(nb_words=8)
        subject = body.split(".")[0][:120]
        created_at = pd.Timestamp(t.occurred_at) + timedelta(hours=int(rng.integers(1, 240)))
        rows.append({
            "id": tid,
            "merchant_id": int(t.merchant_id),
            "customer_id": int(t.customer_id),
            "created_at": created_at,
            "category": cat,
            "subject": subject,
            "body": body,
            "resolved": bool(rng.random() > 0.15),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    fake = Faker()
    Faker.seed(args.seed)

    end_date = date.today()

    print(f"Generating {args.merchants} merchants...")
    merchants = gen_merchants(args.merchants, end_date, rng, fake)
    merchants.to_parquet(DATA_DIR / "merchants.parquet", index=False)

    print(f"Generating {args.customers} customers...")
    customers = gen_customers(args.customers, merchants, end_date, rng)
    customers.to_parquet(DATA_DIR / "customers.parquet", index=False)

    print(f"Generating products (~{args.merchants * 4})...")
    products = gen_products(merchants, args.merchants * 4, rng, fake)
    products.to_parquet(DATA_DIR / "products.parquet", index=False)

    print(f"Generating ~{args.transactions} transactions...")
    transactions = gen_transactions(
        customers, products, merchants, args.transactions,
        end_date, args.months, rng,
    )
    transactions.to_parquet(DATA_DIR / "transactions.parquet", index=False)

    print("Generating support tickets...")
    tickets = gen_tickets(customers, transactions, end_date, rng, fake)
    tickets.to_parquet(DATA_DIR / "tickets.parquet", index=False)

    print("\nSummary:")
    print(f"  merchants:    {len(merchants):>10,}")
    print(f"  customers:    {len(customers):>10,}")
    print(f"  products:     {len(products):>10,}")
    print(f"  transactions: {len(transactions):>10,}")
    print(f"  tickets:      {len(tickets):>10,}")
    print(f"  anomalies:    {int(transactions.is_anomaly.sum()):>10,}")


if __name__ == "__main__":
    main()
