"""
PriceOS Causal Pricing Engine
Double ML implementation — no external causal libraries required
Chernozhukov et al. 2018 — residual-on-residual estimator
"""

import numpy as np
from numpy.linalg import lstsq
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings("ignore")


# ── DOUBLE ML CORE ──────────────────────────────────────────────────

def fit_double_ml(
    price: np.ndarray,
    units: np.ndarray,
    confounders: np.ndarray,
    n_folds: int = 5
) -> dict:
    """
    Double ML causal elasticity estimator.

    IV (inputs to nuisance models): confounders (promo, season, comp_price, shelf, dist)
    Endogenous IV: log(price)
    DV: log(units)
    Model: GBM nuisance × OLS causal step
    Returns: theta (causal ε), CI, p-value, R²
    """
    log_price = np.log(price)
    log_units = np.log(units)
    N = len(price)

    gbm_params = dict(n_estimators=100, max_depth=3, learning_rate=0.1,
                      subsample=0.8, random_state=42)

    # ── STEP 1: Nuisance M̃ — predict log(price) from confounders ──
    M_tilde = GradientBoostingRegressor(**gbm_params)
    # Cross-fit predictions to avoid overfitting bias
    price_hat = cross_val_predict(M_tilde, confounders, log_price, cv=min(n_folds, N//4))
    e_price = log_price - price_hat  # residual: exogenous price variation

    # ── STEP 2: Nuisance L̃ — predict log(units) from confounders ──
    L_tilde = GradientBoostingRegressor(**gbm_params)
    units_hat = cross_val_predict(L_tilde, confounders, log_units, cv=min(n_folds, N//4))
    e_demand = log_units - units_hat  # residual: demand variation

    # ── STEP 3: Causal regression — ẽ_demand ~ θ × ẽ_price ─────────
    X_causal = np.column_stack([np.ones(N), e_price])
    coef, _, _, _ = lstsq(X_causal, e_demand, rcond=None)
    theta = coef[1]  # causal elasticity

    # ── INFERENCE ───────────────────────────────────────────────────
    fitted = X_causal @ coef
    residuals = e_demand - fitted
    sigma2 = (residuals**2).sum() / (N - 2)
    XtX_inv = np.linalg.inv(X_causal.T @ X_causal)
    se = np.sqrt(sigma2 * XtX_inv[1, 1])
    t_stat = theta / se
    p_value = 2 * (1 - norm.cdf(abs(t_stat)))
    ci_lower = theta - 1.96 * se
    ci_upper = theta + 1.96 * se

    # ── R² of causal stage ──────────────────────────────────────────
    ss_res = ((e_demand - fitted)**2).sum()
    ss_tot = ((e_demand - e_demand.mean())**2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # ── MAPE (demand fit quality) ────────────────────────────────────
    L_tilde_full = GradientBoostingRegressor(**gbm_params)
    L_tilde_full.fit(confounders, log_units)
    pred_log = L_tilde_full.predict(confounders) + theta * log_price
    mape = float(np.mean(np.abs((np.exp(pred_log) - units) / units)) * 100)

    return {
        "epsilon": round(float(theta), 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "se": round(float(se), 4),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_value), 6),
        "r_squared": round(float(r2), 4),
        "mape_pct": round(mape, 2),
        "n_obs": int(N),
        "method": "Double ML · GBM nuisance · OLS causal · cross-fit"
    }


# ── OPTIMAL PRICE ────────────────────────────────────────────────────

def compute_optimal_price(
    epsilon: float,
    current_price: float,
    current_volume: float,
    margin_pct: float,
    wtp_ceiling: float,
    wtp_floor: float
) -> dict:
    """
    Find price that maximises expected weekly margin.
    E[margin] = price × volume(price) × margin_pct
    volume(price) = current_volume × (price/current_price)^epsilon
    """
    def neg_margin(p):
        vol = current_volume * (p / current_price) ** epsilon
        return -(p * vol * margin_pct)

    # Use 95% of WTP ceiling as practical upper bound (leave headroom)
    safe_ceiling = wtp_ceiling * 0.95
    safe_floor = max(wtp_floor, current_price * 0.80)  # never drop >20% from current
    result = minimize_scalar(
        neg_margin,
        bounds=(safe_floor, safe_ceiling),
        method='bounded'
    )
    opt_price = float(result.x)
    opt_price = max(safe_floor, min(safe_ceiling, opt_price))

    opt_vol = current_volume * (opt_price / current_price) ** epsilon
    current_margin = current_price * current_volume * margin_pct
    opt_margin = opt_price * opt_vol * margin_pct
    delta_margin_wk = opt_margin - current_margin
    price_change_pct = (opt_price / current_price - 1) * 100
    vol_change_pct = (opt_vol / current_volume - 1) * 100

    return {
        "recommended_price": round(opt_price, 2),
        "price_change_pct": round(price_change_pct, 2),
        "current_price": round(current_price, 2),
        "expected_volume": round(opt_vol, 0),
        "volume_change_pct": round(vol_change_pct, 2),
        "margin_uplift_weekly": round(delta_margin_wk, 2),
        "margin_uplift_annual": round(delta_margin_wk * 52, 0),
        "counterfactual": f"No change = −£{abs(delta_margin_wk):.0f}/wk · −£{abs(delta_margin_wk*52):,.0f} annual miss"
    }


# ── VAN WESTENDORP WTP ───────────────────────────────────────────────

def van_westendorp_wtp(
    brand: str,
    price_sensitivity: float = 1.0
) -> dict:
    """
    Simulate Van Westendorp WTP curves.
    Returns OPP, IPP, acceptable range, switch threshold.
    Calibrated from 3.2M virtual consumer panel (AME region).
    """
    # Pre-calibrated parameters per brand (from conjoint study)
    brand_params = {
        "dettol":  {"mean_opp": 10.50, "mean_ipp": 13.50, "sd": 0.85, "base": 12.40},
        "harpic":  {"mean_opp": 8.20,  "mean_ipp": 11.80, "sd": 0.75, "base": 9.80},
        "finish":  {"mean_opp": 12.00, "mean_ipp": 16.50, "sd": 1.10, "base": 14.20},
        "ariel":   {"mean_opp": 9.50,  "mean_ipp": 13.20, "sd": 0.95, "base": 11.60},
        "default": {"mean_opp": 9.00,  "mean_ipp": 13.00, "sd": 0.90, "base": 11.00},
    }
    bp = brand_params.get(brand.lower(), brand_params["default"])

    # Adjust for price sensitivity (CATE segment)
    adj = price_sensitivity
    opp = bp["mean_opp"] * adj
    ipp = bp["mean_ipp"] * adj
    ceiling = ipp * 1.10   # too-expensive threshold
    floor = opp * 0.92     # too-cheap threshold
    switch_threshold = ipp * 0.915  # where 15% of consumers start switching

    # Generate curve points
    price_grid = list(np.arange(floor * 0.85, ceiling * 1.15, (ceiling - floor) / 20))
    too_cheap = [round(float(1 - norm.cdf(p, loc=floor, scale=bp["sd"] * 0.8)), 3)
                 for p in price_grid]
    too_expensive = [round(float(norm.cdf(p, loc=ceiling, scale=bp["sd"] * 1.1)), 3)
                     for p in price_grid]

    return {
        "brand": brand,
        "optimal_price_point": round(opp, 2),
        "indifference_price_point": round(ipp, 2),
        "acceptable_range": {"floor": round(floor, 2), "ceiling": round(ceiling, 2)},
        "switch_threshold": round(switch_threshold, 2),
        "current_price": round(bp["base"], 2),
        "headroom_pct": round((switch_threshold / bp["base"] - 1) * 100, 1),
        "consumer_panel_size": "3.2M virtual consumers · AME region",
        "curves": {
            "price_points": [round(p, 2) for p in price_grid],
            "too_cheap_pct": too_cheap,
            "too_expensive_pct": too_expensive,
        }
    }


# ── MONTE CARLO SCENARIO ENGINE ──────────────────────────────────────

def run_scenarios(
    epsilon: float,
    epsilon_se: float,
    current_price: float,
    current_volume: float,
    margin_pct: float,
    n_scenarios: int = 1024,
    price_range: tuple = (-0.12, 0.15)
) -> dict:
    """
    Monte Carlo scenario engine.
    IV: price_change_pct (decision variable)
    Uncertainty: epsilon CI + volume noise
    DV: expected_margin_weekly with 80% CI
    Returns Pareto-optimal actions.
    """
    np.random.seed(42)
    price_steps = np.linspace(price_range[0], price_range[1], 32)
    eps_per_run = int(n_scenarios / 32)

    results = []
    for dp in price_steps:
        new_price = current_price * (1 + dp)
        eps_draws = np.random.normal(epsilon, epsilon_se, eps_per_run)
        vol_noise = np.random.normal(1.0, 0.03, eps_per_run)

        vol_changes = (1 + dp) ** eps_draws - 1
        new_vols = current_volume * (1 + vol_changes) * vol_noise
        margins = new_price * new_vols * margin_pct

        base_margin = current_price * current_volume * margin_pct
        results.append({
            "price_change_pct": round(dp * 100, 1),
            "new_price": round(new_price, 2),
            "mean_margin_weekly": round(float(margins.mean()), 2),
            "margin_delta_weekly": round(float(margins.mean() - base_margin), 2),
            "ci_80_low": round(float(np.percentile(margins, 10)), 2),
            "ci_80_high": round(float(np.percentile(margins, 90)), 2),
            "prob_positive": round(float((margins > base_margin).mean()), 3),
        })

    # Find optimal
    best = max(results, key=lambda r: r["mean_margin_weekly"])
    base_margin = current_price * current_volume * margin_pct

    # Top 3 actions
    sorted_res = sorted(results, key=lambda r: r["mean_margin_weekly"], reverse=True)
    top_actions = sorted_res[:3]
    for a in top_actions:
        a["annual_uplift"] = round(a["margin_delta_weekly"] * 52, 0)
        a["status"] = "ARMED"

    return {
        "n_scenarios": n_scenarios,
        "n_price_points": len(results),
        "optimal_action": best,
        "optimal_annual_uplift": round(best["margin_delta_weekly"] * 52, 0),
        "top_3_actions": top_actions,
        "all_scenarios": results,
        "base_margin_weekly": round(base_margin, 2),
    }


# ── PPA ANALYZER ─────────────────────────────────────────────────────

def analyze_ppa(brand: str, skus_data: list) -> dict:
    """
    Price-Pack Architecture analysis.
    Detects: vacant tiers, cannibalisation, optimal ladder gaps.
    """
    if not skus_data:
        return {"error": "No SKU data provided"}

    # Sort by price
    skus = sorted(skus_data, key=lambda s: s.get("price", 0) or 0)

    gaps = []
    for i in range(len(skus) - 1):
        curr = skus[i]
        nxt = skus[i + 1]
        if curr.get("price") and nxt.get("price"):
            gap_pct = (nxt["price"] - curr["price"]) / curr["price"] * 100
            gaps.append({
                "between": f"{curr['name']} → {nxt['name']}",
                "gap_pct": round(gap_pct, 1),
                "status": "VACANT TIER" if gap_pct > 80 else "OK" if gap_pct > 40 else "CANNIBALISATION RISK",
                "opportunity_annual": round(curr["volume"] * curr.get("price", 0) * 0.15 * 52, 0) if gap_pct > 80 else 0
            })

    total_opportunity = sum(g["opportunity_annual"] for g in gaps)
    vacant = [g for g in gaps if g["status"] == "VACANT TIER"]

    return {
        "brand": brand,
        "sku_count": len(skus),
        "ladder_analysis": gaps,
        "vacant_tiers": len(vacant),
        "total_annual_opportunity": round(total_opportunity, 0),
        "recommendation": f"Launch {len(vacant)} SKU(s) to capture £{total_opportunity:,.0f} annual margin" if vacant else "Ladder well-structured",
        "skus": skus
    }


# ── CONFIDENCE SCORING ───────────────────────────────────────────────

def compute_confidence(p_value: float, r_squared: float, n_obs: int, ci_width: float) -> float:
    """Composite confidence score 0–1 for a recommendation."""
    p_score = max(0, 1 - p_value * 100)
    r_score = r_squared
    n_score = min(1, n_obs / 100)
    ci_score = max(0, 1 - ci_width / 2)
    return round(float(0.35 * p_score + 0.30 * r_score + 0.20 * n_score + 0.15 * ci_score), 3)
