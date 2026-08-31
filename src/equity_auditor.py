"""
equity_auditor.py

The "Who is being left behind?" feature -- built as a Justice40-style metric
(directly mirroring Paper 7, Pollack et al. 2025, in your literature review):
what SHARE of total investment (Rs.) flows to high-vulnerability zones,
compared against what share it "should" get if investment were proportional
to the population living in zones classified as high-vulnerability.

  target_share      = population of high-vuln zones / total population
  actual_share(method) = Rs. invested in high-vuln zones / total Rs. invested

A method "adequately" serves vulnerable zones when actual_share is close to
or above target_share. This is compared for:
  - Greedy (cost-effectiveness-only, the conventional baseline)
  - CarbonWise (equity-constrained NSGA-II recommendation)
"""
import pandas as pd

HIGH_VULN_THRESHOLD = 0.70


def investment_share_report(df: pd.DataFrame, wards: pd.DataFrame,
                             funded_ids: list[str]) -> dict:
    high_vuln_zone_ids = wards[wards["vulnerability_index"] >= HIGH_VULN_THRESHOLD]["zone_id"].tolist()

    funded_df = df[df["project_id"].isin(funded_ids)]
    total_invested = funded_df["cost_cr_inr"].sum()
    invested_in_high_vuln = funded_df[funded_df["zone_id"].isin(high_vuln_zone_ids)]["cost_cr_inr"].sum()

    actual_share = 100 * invested_in_high_vuln / total_invested if total_invested > 0 else 0

    total_pop = wards["approx_population_2023"].sum()
    high_vuln_pop = wards[wards["zone_id"].isin(high_vuln_zone_ids)]["approx_population_2023"].sum()
    target_share = 100 * high_vuln_pop / total_pop if total_pop > 0 else 0

    return {
        "total_invested_cr": total_invested,
        "invested_in_high_vuln_cr": invested_in_high_vuln,
        "actual_share_pct": actual_share,
        "target_share_pct": target_share,
        "gap_pct_points": actual_share - target_share,
    }


def audit_report(df: pd.DataFrame, wards: pd.DataFrame,
                  greedy_ids: list[str], carbonwise_ids: list[str]) -> dict:
    before = investment_share_report(df, wards, greedy_ids)
    after = investment_share_report(df, wards, carbonwise_ids)

    return {
        "before": before,
        "after": after,
        "target_share_pct": before["target_share_pct"],  # same for both
        "improvement_pct_points": after["actual_share_pct"] - before["actual_share_pct"],
    }


if __name__ == "__main__":
    from preprocessing import preprocess
    from equity_scoring import compute_equity_score
    from optimizer import run_nsga2, pareto_front_to_df, pick_balanced_solution
    from baselines import greedy_cost_effectiveness_allocation

    df = preprocess()
    df = compute_equity_score(df)
    wards = pd.read_csv("../data/raw/wards.csv")

    BUDGET_CR = 50.0
    greedy_mask = greedy_cost_effectiveness_allocation(df, BUDGET_CR)
    greedy_ids = df[greedy_mask]["project_id"].tolist()

    res, problem = run_nsga2(df, BUDGET_CR)
    front_df = pareto_front_to_df(res, problem, df)
    balanced = pick_balanced_solution(front_df)
    cw_ids = balanced["selected_projects"].split(";")

    report = audit_report(df, wards, greedy_ids, cw_ids)
    print(f"Target share (high-vuln zones' population share of Chennai): {report['target_share_pct']:.0f}%")
    print(f"\nBEFORE (Greedy, cost-effectiveness only):")
    print(f"  Investment share to high-vuln zones: {report['before']['actual_share_pct']:.0f}% "
          f"(gap: {report['before']['gap_pct_points']:+.0f} pts vs target)")
    print(f"\nAFTER (CarbonWise):")
    print(f"  Investment share to high-vuln zones: {report['after']['actual_share_pct']:.0f}% "
          f"(gap: {report['after']['gap_pct_points']:+.0f} pts vs target)")
    print(f"\nImprovement: {report['improvement_pct_points']:+.0f} percentage points")

