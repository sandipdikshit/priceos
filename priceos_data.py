"""
PriceOS Synthetic Dataset
Realistic CPG transaction data for AME region (Africa, Middle East)
Deterministic seed — reproducible across deploys
"""

import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta

np.random.seed(42)

# ── BRAND / SKU CATALOGUE ────────────────────────────────────────────
SKU_CATALOGUE = [
    # (sku_id, brand, name, base_price, true_epsilon, margin_pct, base_volume, market)
    ("DTL-500-NA", "Dettol", "Dettol 500ml",       12.40, -2.41, 0.42, 4850, "Shoprite-NA"),
    ("DTL-250-NA", "Dettol", "Dettol 250ml",        7.20, -2.65, 0.38, 2100, "Shoprite-NA"),
    ("DTL-1L-NA",  "Dettol", "Dettol 1L",          21.50, -2.18, 0.44, 1920, "Shoprite-NA"),
    ("DTL-500-EG", "Dettol", "Dettol 500ml Egypt", 11.80, -2.55, 0.40, 3200, "Carrefour-EG"),
    ("HRP-500-NA", "Harpic", "Harpic 500ml",        6.80, -2.75, 0.36, 3200, "Shoprite-NA"),
    ("HRP-1L-NA",  "Harpic", "Harpic 1L",          19.80, -2.28, 0.45, 1450, "Shoprite-NA"),
    ("HRP-500-AE", "Harpic", "Harpic 500ml UAE",   14.50, -2.30, 0.43, 2800, "LuLu-AE"),
    ("FIN-TBL-NA", "Finish", "Finish Tabs 40ct",   14.20, -2.90, 0.48, 1800, "Shoprite-NA"),
    ("FIN-PWD-NA", "Finish", "Finish Powder 1kg",   9.80, -2.45, 0.41, 2200, "Shoprite-NA"),
    ("ARL-3KG-NA", "Ariel",  "Ariel 3kg",          22.80, -2.10, 0.46, 2900, "Shoprite-NA"),
    ("ARL-1KG-NA", "Ariel",  "Ariel 1kg",           9.50, -2.55, 0.40, 4100, "Shoprite-NA"),
    ("ARL-3KG-AE", "Ariel",  "Ariel 3kg UAE",      28.50, -1.95, 0.50, 1900, "LuLu-AE"),
]

# ── MARKET / COMPETITOR MAP ──────────────────────────────────────────
COMP_MAP = {
    "Shoprite-NA":  {"comp_brand": "Store-Brand-NA",  "comp_premium": -0.12},
    "Carrefour-EG": {"comp_brand": "Henkel-EG",       "comp_premium": +0.08},
    "LuLu-AE":      {"comp_brand": "Unilever-AE",     "comp_premium": +0.15},
}


def generate_weekly_data(sku: dict, n_weeks: int = 104) -> list:
    """
    Generate 2 years of weekly transaction data per SKU.
    Confounders injected deterministically per SKU seed.
    """
    np.random.seed(hash(sku["sku_id"]) % 10000)

    sku_id    = sku["sku_id"]
    base_px   = sku["base_price"]
    true_eps  = sku["true_epsilon"]
    mkt       = sku["market"]
    base_vol  = sku["base_volume"]
    comp_prem = COMP_MAP[mkt]["comp_premium"]

    records = []
    start = datetime(2024, 1, 1)

    for w in range(n_weeks):
        week_date = start + timedelta(weeks=w)

        # ── CONFOUNDERS ──────────────────────────────────────────────
        promo       = int(np.random.binomial(1, 0.22))
        season_idx  = float(1.0 + 0.28 * np.sin(2 * np.pi * w / 52) +
                            0.12 * np.sin(4 * np.pi * w / 52))
        comp_price  = float(base_px * (1 + comp_prem) + np.random.normal(0, 0.35))
        shelf_pos   = int(np.random.choice([1, 2, 3], p=[0.40, 0.40, 0.20]))
        dist_score  = float(np.clip(np.random.normal(0.85, 0.06), 0.65, 0.98))

        # ── PRICE (endogenous — confounders affect it) ────────────────
        price = float(base_px
                      - 1.75 * promo
                      - 0.28 * (comp_price - base_px * (1 + comp_prem))
                      + np.random.normal(0, 0.18))
        price = max(base_px * 0.70, min(base_px * 1.25, price))

        # ── VOLUME (true causal DGP) ──────────────────────────────────
        log_vol = (np.log(base_vol)
                   + true_eps * np.log(price / base_px)  # TRUE causal effect
                   + 0.42 * promo                         # promo display boost
                   + 0.28 * np.log(season_idx)           # seasonal demand
                   - 0.11 * shelf_pos                     # shelf penalty
                   + 0.58 * dist_score                    # distribution
                   + np.random.normal(0, 0.07))
        units_sold = int(max(10, round(np.exp(log_vol))))

        # ── REVENUE ───────────────────────────────────────────────────
        revenue = round(price * units_sold, 2)
        margin  = round(revenue * sku["margin_pct"], 2)

        records.append({
            "sku_id":       sku_id,
            "week":         week_date.strftime("%Y-%m-%d"),
            "price":        round(price, 2),
            "units_sold":   units_sold,
            "revenue":      revenue,
            "margin":       margin,
            "promo_week":   promo,
            "season_idx":   round(season_idx, 4),
            "comp_price":   round(comp_price, 2),
            "shelf_position": shelf_pos,
            "dist_score":   round(dist_score, 4),
        })

    return records


def init_db(db_path: str = "priceos.db") -> None:
    """Create SQLite schema and seed with synthetic data."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # ── SKU table ────────────────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS skus (
        sku_id TEXT PRIMARY KEY,
        brand TEXT, name TEXT, base_price REAL, true_epsilon REAL,
        margin_pct REAL, base_volume INTEGER, market TEXT
    )""")

    # ── Weekly transactions ──────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sku_id TEXT, week TEXT, price REAL, units_sold INTEGER,
        revenue REAL, margin REAL, promo_week INTEGER,
        season_idx REAL, comp_price REAL, shelf_position INTEGER, dist_score REAL
    )""")

    # ── Audit / decision log ─────────────────────────────────────────
    c.execute("""CREATE TABLE IF NOT EXISTS audit_log (
        decision_id TEXT PRIMARY KEY,
        sku_id TEXT, timestamp TEXT,
        current_price REAL, recommended_price REAL, price_change_pct REAL,
        epsilon REAL, confidence REAL,
        margin_uplift_weekly REAL, annual_uplift REAL,
        counterfactual TEXT, guardrails TEXT,
        status TEXT, approved_by TEXT
    )""")

    # Seed SKUs
    for s in SKU_CATALOGUE:
        c.execute("""INSERT OR IGNORE INTO skus VALUES (?,?,?,?,?,?,?,?)""", s)

    # Seed transactions (only if empty)
    c.execute("SELECT COUNT(*) FROM transactions")
    if c.fetchone()[0] == 0:
        for s in SKU_CATALOGUE:
            sku_dict = dict(zip(
                ["sku_id","brand","name","base_price","true_epsilon","margin_pct","base_volume","market"], s
            ))
            records = generate_weekly_data(sku_dict)
            c.executemany("""INSERT INTO transactions
                (sku_id,week,price,units_sold,revenue,margin,promo_week,
                 season_idx,comp_price,shelf_position,dist_score)
                VALUES (:sku_id,:week,:price,:units_sold,:revenue,:margin,:promo_week,
                        :season_idx,:comp_price,:shelf_position,:dist_score)""", records)

    conn.commit()
    conn.close()
    print(f"✓ DB initialised: {len(SKU_CATALOGUE)} SKUs × 104 weeks = {len(SKU_CATALOGUE)*104:,} records")


def get_sku_data(sku_id: str, db_path: str = "priceos.db") -> dict:
    """Fetch SKU metadata + transaction array."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    meta = c.execute("SELECT * FROM skus WHERE sku_id=?", (sku_id,)).fetchone()
    if not meta:
        conn.close()
        return {}

    rows = c.execute(
        "SELECT * FROM transactions WHERE sku_id=? ORDER BY week",
        (sku_id,)
    ).fetchall()
    conn.close()

    return {
        "meta": dict(meta),
        "transactions": [dict(r) for r in rows]
    }


def get_all_skus(db_path: str = "priceos.db") -> list:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM skus ORDER BY brand, name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_audit(decision: dict, db_path: str = "priceos.db") -> None:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO audit_log
        (decision_id,sku_id,timestamp,current_price,recommended_price,
         price_change_pct,epsilon,confidence,margin_uplift_weekly,annual_uplift,
         counterfactual,guardrails,status,approved_by)
        VALUES (:decision_id,:sku_id,:timestamp,:current_price,:recommended_price,
                :price_change_pct,:epsilon,:confidence,:margin_uplift_weekly,:annual_uplift,
                :counterfactual,:guardrails,:status,:approved_by)""", decision)
    conn.commit()
    conn.close()


def get_audit(decision_id: str, db_path: str = "priceos.db") -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM audit_log WHERE decision_id=?", (decision_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}
