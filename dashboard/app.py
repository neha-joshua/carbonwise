"""
CarbonWise — Chennai Infrastructure Investment Decision Support
Run with: streamlit run app.py   (from inside the dashboard/ folder)
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates["carbonwise"] = pio.templates["plotly_white"]
pio.templates["carbonwise"].layout.font = dict(family="Times New Roman, Times, serif", size=13)
pio.templates.default = "carbonwise"
from io import StringIO
from datetime import date

from preprocessing import preprocess
from equity_scoring import compute_equity_score
from optimizer import run_nsga2, pareto_front_to_df, pick_balanced_solution
from baselines import (equal_split_allocation, greedy_cost_effectiveness_allocation,
                        summarize_portfolio)
from scenarios import build_scenarios
from agents import full_council_report
from equity_auditor import audit_report
from report import generate_report
from sensitivity import run_sensitivity
from llm import client as llm_client
from llm.explain import generate_explanation
from data_validation import validate as validate_data
from decision_rationale import build_rationale

# ---------------------------------------------------------------------------
# Page config + restrained, professional styling (no gradients, no emoji soup)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CarbonWise", layout="wide",
                    initial_sidebar_state="expanded", page_icon="◆")

INK = "#1A2B22"
FOREST = "#0F4C3A"
MUTED = "#6B7A72"
LINE = "#DDE5E0"
GOLD = "#B8860B"
DANGER = "#9C4141"
CARD_BG = "#FBFCFB"

st.markdown(f"""
<style>
    html, body, .stApp {{
        font-family: "Times New Roman", Times, serif;
    }}
    p, span, li, td, th, label, h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stDataFrame, .stButton button, .stSelectbox div, .stRadio label,
    .stTextInput input, .stNumberInput input, .stSlider label, .stCaption {{
        font-family: "Times New Roman", Times, serif;
    }}
    [data-testid="stIconMaterial"], [class*="Icon"], .material-icons,
    [data-testid="stExpanderToggleIcon"] {{
        font-family: "Material Symbols Rounded", "Material Icons" !important;
    }}
    .main {{ background-color: #FFFFFF; }}
    section[data-testid="stSidebar"] {{ background-color: #F7F9F8; border-right: 1px solid {LINE}; }}

    .cw-title {{ font-size: 26px; font-weight: 700; color: {INK}; margin-bottom: 2px; letter-spacing: -0.3px; }}
    .cw-subtitle {{ font-size: 14px; color: {MUTED}; margin-bottom: 4px; }}
    .cw-rule {{ border: none; border-top: 2px solid {FOREST}; width: 64px; margin: 10px 0 20px 0; }}

    .cw-tile {{
        background: {CARD_BG}; border: 1px solid {LINE}; border-radius: 6px;
        padding: 14px 18px; margin-bottom: 10px;
    }}
    .cw-tile-label {{ font-size: 11px; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px; }}
    .cw-tile-value {{ font-size: 28px; font-weight: 700; color: {INK}; line-height: 1.1; }}
    .cw-tile-context {{ font-size: 12px; color: {MUTED}; margin-top: 2px; }}

    .cw-finding {{
        border-left: 3px solid {FOREST}; background: {CARD_BG}; border-radius: 4px;
        padding: 10px 14px; margin-bottom: 8px; font-size: 14px; color: {INK};
    }}
    .cw-finding b {{ color: {FOREST}; }}

    .cw-callout {{
        background: #FBF7EC; border: 1px solid {GOLD}; border-radius: 6px;
        padding: 14px 18px; margin: 10px 0; font-size: 14px; color: {INK};
    }}
    .cw-source {{ font-size: 11px; color: {MUTED}; font-style: italic; }}

    div.stButton > button[kind="primary"] {{
        background-color: {FOREST}; border-color: {FOREST};
    }}
    div.stButton > button[kind="primary"]:hover {{
        background-color: {INK}; border-color: {INK};
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="cw-title">CarbonWise</div>
<div class="cw-subtitle">Cross-sector infrastructure investment platform — solar, transit, EV charging, drainage, green corridors, flood defenses & building retrofits — unified under equity-aware multi-objective optimisation. Chennai pilot.</div>
<hr class="cw-rule"/>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("**Budget & Priorities**")
budget_cr = st.sidebar.number_input("Available budget (₹ crore)", min_value=5.0,
                                     max_value=500.0, value=50.0, step=5.0)

st.sidebar.markdown("**Equity metric weights**")
w1 = st.sidebar.slider("Vulnerability", 0.0, 1.0, 0.33)
w2 = st.sidebar.slider("Population benefited", 0.0, 1.0, 0.33)
w3 = st.sidebar.slider("Infrastructure deprivation", 0.0, 1.0, 0.34)
total_w = w1 + w2 + w3
w1, w2, w3 = (w1/total_w, w2/total_w, w3/total_w) if total_w else (1/3, 1/3, 1/3)

run_button = st.sidebar.button("Run Optimizer", type="primary", use_container_width=True)
run_sensitivity_btn = st.sidebar.checkbox("Also run budget sensitivity analysis (~10s extra)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data sources**")
st.sidebar.caption(
    "Ward vulnerability: Nandhini, Das & Shinde (2025), IIT Kharagpur/NIUA — "
    "TOPSIS composite vulnerability index for Chennai's 15 GCC zones.\n\n"
    "Project cost/impact ranges: MNRE, CEA, CPHEEO, FAME-II published benchmarks."
)

# ---------------------------------------------------------------------------
# Load + score data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(w1, w2, w3):
    df = preprocess()
    df = compute_equity_score(df, w1=w1, w2=w2, w3=w3)
    wards = pd.read_csv("../data/raw/wards.csv")
    return df, wards

df, wards = load_data(w1, w2, w3)

# ---------------------------------------------------------------------------
# Infrastructure category overview -- makes the cross-sector scope of the
# platform immediately visible, rather than buried in a data table
# ---------------------------------------------------------------------------
st.markdown("**Infrastructure categories evaluated**")
_cat_summary = df.groupby("category").agg(
    n_projects=("project_id", "count"),
    total_cost=("cost_cr_inr", "sum"),
).reset_index().sort_values("total_cost", ascending=False)

_cat_cols = st.columns(4)
for i, (_, row) in enumerate(_cat_summary.iterrows()):
    with _cat_cols[i % 4]:
        st.markdown(f"""
        <div class="cw-tile" style="min-height: 92px;">
            <div class="cw-tile-label" style="font-size:10.5px; line-height:1.3;">{row['category']}</div>
            <div class="cw-tile-value" style="font-size:20px; margin-top:4px;">{int(row['n_projects'])} projects</div>
            <div class="cw-tile-context">\u20b9{row['total_cost']:.0f} cr candidate cost</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

tab_map, tab_data, tab_results = st.tabs(["Zone Profile", "Candidate Projects", "Investment Analysis"])

# ---------------------------------------------------------------------------
# Data validation report (sidebar) -- shown before any optimization runs,
# so the panel sees the system checks its inputs before trusting them
# ---------------------------------------------------------------------------
_validation = validate_data(df, wards)
with st.sidebar.expander(
    f"Data quality: {'Ready' if _validation['status'] == 'READY' else 'Issues found'}",
    expanded=False
):
    st.markdown(f"""
    - Projects loaded: **{_validation['n_projects']}**
    - Chennai zones: **{_validation['n_zones']}**
    - Project categories: **{_validation['n_categories']}**
    - Total candidate cost: **₹{_validation['total_candidate_cost_cr']} cr**
    - Missing values: **{_validation['n_missing_values']}**
    """)
    if _validation["issues"]:
        for issue in _validation["issues"]:
            st.warning(issue)


# ---------------------------------------------------------------------------
# TAB 1: Zone map / profile
# ---------------------------------------------------------------------------
with tab_map:
    col_map, col_detail = st.columns([2, 1])

    with col_map:
        fig_map = px.scatter(
            wards, x="lon", y="lat", size="approx_population_2023",
            color="vulnerability_index", color_continuous_scale="YlOrRd",
            hover_name="zone_name",
            hover_data={"lon": False, "lat": False, "vulnerability_index": ":.2f",
                        "infrastructure_deprivation_index": ":.2f",
                        "approx_population_2023": True},
            size_max=48, height=520,
        )
        fig_map.update_traces(marker=dict(line=dict(width=1.5, color="white")))
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="West \u2192 East", yaxis_title="South \u2192 North",
            plot_bgcolor="#F2F5F3", paper_bgcolor="white",
            coloraxis_colorbar=dict(title="Vulnerability"),
        )
        fig_map.update_yaxes(scaleanchor="x", scaleratio=1)
        for _, row in wards.iterrows():
            fig_map.add_annotation(x=row["lon"], y=row["lat"], text=row["zone_name"],
                                    showarrow=False, yshift=15, font=dict(size=9, color=INK))
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption("Zones positioned by real relative latitude/longitude. Bubble size = population, "
                   "color = vulnerability index.")


    with col_detail:
        st.markdown("**Zone inspector**")
        selected_zone = st.selectbox("Select a zone", wards["zone_name"].tolist(), label_visibility="collapsed")
        zrow = wards[wards["zone_name"] == selected_zone].iloc[0]

        st.markdown(f"""
        <div class="cw-tile"><div class="cw-tile-label">Vulnerability Index</div>
            <div class="cw-tile-value">{zrow['vulnerability_index']:.2f}</div></div>
        <div class="cw-tile"><div class="cw-tile-label">Infrastructure Deprivation</div>
            <div class="cw-tile-value">{zrow['infrastructure_deprivation_index']:.2f}</div></div>
        <div class="cw-tile"><div class="cw-tile-label">Population (2023 est.)</div>
            <div class="cw-tile-value">{zrow['approx_population_2023']:,}</div></div>
        """, unsafe_allow_html=True)

        n_candidate_projects = len(df[df["zone_id"] == zrow["zone_id"]])
        st.markdown(f"**Candidate projects in zone:** {n_candidate_projects}")
        st.markdown(f'<div class="cw-source">Source: {zrow["data_source"]}<br>{zrow["notes"]}</div>',
                     unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 2: Candidate projects table
# ---------------------------------------------------------------------------
with tab_data:
    fig_cat = go.Figure(go.Bar(
        x=_cat_summary["total_cost"], y=_cat_summary["category"],
        orientation="h", marker_color=FOREST,
        text=_cat_summary["n_projects"].astype(str) + " projects",
        textposition="outside",
    ))
    fig_cat.update_layout(height=280, margin=dict(l=0, r=150, t=10, b=0),
                           xaxis_title="Total candidate cost (\u20b9 crore)",
                           plot_bgcolor="white")
    st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("**Filter by category**")
    selected_categories = st.multiselect(
        "Category", options=sorted(df["category"].unique()),
        default=sorted(df["category"].unique()), label_visibility="collapsed"
    )

    st.markdown("**Candidate infrastructure projects**")
    filtered_df = df[df["category"].isin(selected_categories)] if selected_categories else df
    st.dataframe(
        filtered_df[["project_id", "project_name", "category", "zone_name",
            "cost_cr_inr", "carbon_score_raw", "resilience_score_raw", "equity_score"]]
        .rename(columns={"cost_cr_inr": "cost_₹cr", "carbon_score_raw": "carbon_score",
                          "resilience_score_raw": "resilience_score"})
        .round(1),
        height=400, use_container_width=True, hide_index=True
    )

# ---------------------------------------------------------------------------
# TAB 3: Investment analysis (optimizer, scenarios, findings, equity audit, sensitivity)
# ---------------------------------------------------------------------------
with tab_results:
    if not run_button:
        st.info("Set your budget and priority weights in the sidebar, then click **Run Optimizer**.")
    else:
        with st.spinner("Solving multi-objective portfolio selection (NSGA-II)..."):
            res, problem = run_nsga2(df, budget_cr)
            front_df = pareto_front_to_df(res, problem, df)
            balanced = pick_balanced_solution(front_df)

            eq_mask = equal_split_allocation(df, budget_cr)
            greedy_mask = greedy_cost_effectiveness_allocation(df, budget_cr)
            cw_ids = balanced["selected_projects"].split(";")
            cw_mask = df["project_id"].isin(cw_ids)

            comparison = pd.DataFrame([
                summarize_portfolio(df, eq_mask, "Equal Split"),
                summarize_portfolio(df, greedy_mask, "Greedy Cost-Effectiveness"),
                summarize_portfolio(df, cw_mask, "CarbonWise"),
            ])
            scenarios_df = build_scenarios(front_df)
            council = full_council_report(comparison, budget_cr)
            greedy_ids = df[greedy_mask]["project_id"].tolist()
            audit = audit_report(df, wards, greedy_ids, cw_ids)

        # --- KPI tiles ---
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, ctx in [
            (c1, "Carbon Score", f"{balanced['carbon_score']:.0f}", "sum across funded projects"),
            (c2, "Resilience Score", f"{balanced['resilience_score']:.0f}", "sum across funded projects"),
            (c3, "Equity Score", f"{balanced['equity_score']:.0f}", "sum across funded projects"),
            (c4, "Budget Utilised", f"{balanced['total_cost_cr']/budget_cr*100:.0f}%",
             f"₹{balanced['total_cost_cr']:.1f} cr of ₹{budget_cr:.0f} cr"),
        ]:
            col.markdown(f'<div class="cw-tile"><div class="cw-tile-label">{label}</div>'
                          f'<div class="cw-tile-value">{val}</div>'
                          f'<div class="cw-tile-context">{ctx}</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # --- Why this portfolio: structured decision rationale ---
        st.markdown("#### Why this portfolio?")
        rationale = build_rationale(front_df, balanced, budget_cr, audit)
        for i, point in enumerate(rationale, 1):
            st.markdown(f'<div class="cw-finding"><b>{i}. {point["title"]}.</b> {point["detail"]}</div>',
                         unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("#### Comparison against conventional allocation methods")
        fig_comp = go.Figure()
        for metric, color in zip(["carbon_score", "resilience_score", "equity_score"],
                                   [FOREST, "#3D6B8C", GOLD]):
            fig_comp.add_trace(go.Bar(name=metric.replace("_score", "").title(),
                                       x=comparison["method"], y=comparison[metric], marker_color=color))
        fig_comp.update_layout(barmode="group", height=360, margin=dict(l=0, r=0, t=10, b=0),
                                plot_bgcolor="white", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")

        # --- Scenario simulator ---
        st.markdown("#### Scenario comparison")
        st.caption("Four trade-off points drawn from the same Pareto front — not independently re-optimized runs.")
        fig_scen = go.Figure()
        for metric, color in zip(["carbon_score", "resilience_score", "equity_score"],
                                   [FOREST, "#3D6B8C", GOLD]):
            fig_scen.add_trace(go.Bar(name=metric.replace("_score", "").title(),
                                       x=scenarios_df["scenario"], y=scenarios_df[metric], marker_color=color))
        fig_scen.update_layout(barmode="group", height=360, margin=dict(l=0, r=0, t=10, b=0),
                                plot_bgcolor="white", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_scen, use_container_width=True)
        st.dataframe(scenarios_df[["scenario", "n_projects", "total_cost_cr",
                                     "carbon_score", "resilience_score", "equity_score"]].round(1),
                     use_container_width=True, hide_index=True)

        st.markdown("---")

        # --- Findings + Pareto front ---
        col_findings, col_pareto = st.columns([1, 1])
        with col_findings:
            st.markdown("#### Key findings")
            ai_status = "AI explanation: enabled" if llm_client.is_available() else "AI explanation: not configured (using rule-based analysis)"
            st.caption(ai_status)

            context = {
                "budget_cr": budget_cr,
                "carbon_score": round(balanced["carbon_score"], 0),
                "resilience_score": round(balanced["resilience_score"], 0),
                "equity_score": round(balanced["equity_score"], 0),
                "projects_funded": int(balanced["n_projects"]),
                "budget_utilised_pct": round(balanced["total_cost_cr"] / budget_cr * 100, 0),
                "greedy_carbon_score": round(comparison[comparison["method"].str.contains("Greedy")]["carbon_score"].iloc[0], 0),
                "high_vulnerability_investment_share_pct": round(audit["after"]["actual_share_pct"], 0),
            }
            explanation = generate_explanation(context, comparison, budget_cr)
            st.markdown(f'<div class="cw-finding">{explanation["text"]}</div>', unsafe_allow_html=True)
            if explanation["source"] == "llm":
                st.caption("Generated by LLM from verified optimizer output.")
            elif explanation["error"]:
                st.caption(f"Note: {explanation['error']}")

            with st.expander("Per-agent breakdown"):
                for entry in council:
                    label = entry["agent"]
                    st.markdown(f'<div class="cw-finding"><b>{label}.</b> {entry["statement"]}</div>',
                                 unsafe_allow_html=True)

        with col_pareto:
            st.markdown("#### Pareto-optimal trade-off frontier")
            fig_pareto = px.scatter(front_df, x="carbon_score", y="resilience_score",
                                     color="equity_score", color_continuous_scale="Viridis",
                                     hover_data={"n_projects": True, "total_cost_cr": ":.1f",
                                                 "carbon_score": ":.0f", "resilience_score": ":.0f",
                                                 "equity_score": ":.0f"},
                                     height=380)
            fig_pareto.add_trace(go.Scatter(
                x=[balanced["carbon_score"]], y=[balanced["resilience_score"]],
                mode="markers", marker=dict(symbol="star", size=20, color=DANGER,
                                             line=dict(color=INK, width=1)),
                name="Recommended",
                hovertext=f"RECOMMENDED<br>Projects: {int(balanced['n_projects'])}<br>"
                          f"Cost: \u20b9{balanced['total_cost_cr']:.1f} cr<br>"
                          f"Carbon: {balanced['carbon_score']:.0f} | "
                          f"Resilience: {balanced['resilience_score']:.0f} | "
                          f"Equity: {balanced['equity_score']:.0f}",
                hoverinfo="text"))
            fig_pareto.update_layout(margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white")
            st.plotly_chart(fig_pareto, use_container_width=True)
            st.caption(f"{len(front_df)} non-dominated solutions found across {len(df)} candidate projects.")

        st.markdown("---")

        # --- Equity Audit ---
        st.markdown("#### Distributive equity audit")
        st.caption("Share of total investment reaching Chennai's high-vulnerability zones, "
                   "benchmarked against a population-proportional target based on Chennai "
                   "residents living in zones classified as high-vulnerability (Justice40-style metric).")
        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown(f"""
            <div class="cw-tile"><div class="cw-tile-label">Greedy (cost-effectiveness only)</div>
                <div class="cw-tile-value" style="color:{DANGER}">{audit['before']['actual_share_pct']:.0f}%</div>
                <div class="cw-tile-context">of investment to high-vulnerability zones
                (target: {audit['target_share_pct']:.0f}%, gap {audit['before']['gap_pct_points']:.0f} pts)</div>
            </div>""", unsafe_allow_html=True)
        with col_after:
            st.markdown(f"""
            <div class="cw-tile"><div class="cw-tile-label">CarbonWise</div>
                <div class="cw-tile-value" style="color:{FOREST}">{audit['after']['actual_share_pct']:.0f}%</div>
                <div class="cw-tile-context">of investment to high-vulnerability zones
                (target: {audit['target_share_pct']:.0f}%, gap {audit['after']['gap_pct_points']:+.0f} pts)</div>
            </div>""", unsafe_allow_html=True)

        if audit["before"]["gap_pct_points"] < -5:
            st.markdown(f"""
            <div class="cw-callout">
                <b>Distributive shortfall detected.</b> Cost-effectiveness-only allocation directs
                {audit['before']['actual_share_pct']:.0f}% of investment to zones holding
                {audit['target_share_pct']:.0f}% of Chennai's high-vulnerability population — a
                {abs(audit['before']['gap_pct_points']):.0f}-point shortfall, consistent with the
                underinvestment pattern documented empirically in Nandhini et al. (2025).
                CarbonWise's recommended portfolio corrects this to
                {audit['after']['actual_share_pct']:.0f}% ({audit['improvement_pct_points']:+.0f} points)
                by encoding equity as an explicit optimisation objective.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="cw-callout">
                <b>Honest reading of this result.</b> At this budget, both methods currently meet
                or exceed the population-proportional target ({audit['target_share_pct']:.0f}%).
                CarbonWise does not claim to always outperform Greedy on this specific
                investment-share metric — its measured advantage here is in resilience and the
                weighted equity score (above). CarbonWise's contribution is that equity is encoded
                as an explicit, auditable optimisation objective and independently audited on every
                run, rather than assumed.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --- Sensitivity analysis (optional, slower) ---
        if run_sensitivity_btn:
            st.markdown("#### Budget sensitivity")
            st.caption("How outcomes and marginal returns change as available budget scales — "
                       "useful for a finance department deciding whether additional budget is worth requesting.")
            with st.spinner("Running optimizer across 5 budget levels (~10-15s)..."):
                levels = sorted(set([max(10, budget_cr * f) for f in [0.4, 0.7, 1.0, 1.5, 2.0]]))
                sens_df = run_sensitivity(df, levels, pop_size=80, n_gen=100)

            fig_sens = go.Figure()
            for metric, color in zip(["carbon_score", "resilience_score", "equity_score"],
                                       [FOREST, "#3D6B8C", GOLD]):
                fig_sens.add_trace(go.Scatter(x=sens_df["budget_cr"], y=sens_df[metric],
                                                mode="lines+markers", name=metric.replace("_score","").title(),
                                                line=dict(color=color, width=2)))
            fig_sens.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white",
                                    xaxis_title="Budget (₹ crore)", yaxis_title="Score",
                                    legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_sens, use_container_width=True)

            sens_df["marginal_carbon_per_cr"] = sens_df["carbon_score"].diff() / sens_df["budget_cr"].diff()
            st.dataframe(sens_df.round(2), use_container_width=True, hide_index=True)
            st.caption("Marginal carbon score gained per additional ₹1 crore — a declining trend indicates "
                       "diminishing returns on further budget increases.")
            st.markdown("---")

        # --- Recommended portfolio + export ---
        st.markdown("#### Recommended portfolio")
        recommended = df[df["project_id"].isin(cw_ids)][
            ["project_id", "project_name", "category", "zone_name", "cost_cr_inr",
             "carbon_score_raw", "resilience_score_raw", "equity_score"]
        ].round(1)
        st.dataframe(recommended, use_container_width=True, hide_index=True)

        csv_buffer = StringIO()
        recommended.to_csv(csv_buffer, index=False)
        report_header = (
            f"# CarbonWise Investment Brief\n"
            f"# Generated: {date.today().isoformat()}\n"
            f"# Budget: Rs. {budget_cr:.1f} crore | Utilised: Rs. {balanced['total_cost_cr']:.1f} crore "
            f"({balanced['total_cost_cr']/budget_cr*100:.0f}%)\n"
            f"# Carbon score: {balanced['carbon_score']:.0f} | Resilience score: {balanced['resilience_score']:.0f} "
            f"| Equity score: {balanced['equity_score']:.0f}\n"
            f"# Equity weights used: vulnerability={w1:.2f}, population_benefited={w2:.2f}, "
            f"infrastructure_deprivation={w3:.2f}\n\n"
        )

        full_report_text = generate_report(df, wards, budget_cr, balanced, front_df,
                                            comparison, audit, rationale, (w1, w2, w3))

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Download full investment decision report (TXT)",
                data=full_report_text,
                file_name=f"carbonwise_decision_report_{date.today().isoformat()}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                "Download portfolio only (CSV)",
                data=report_header + csv_buffer.getvalue(),
                file_name=f"carbonwise_portfolio_{date.today().isoformat()}.csv",
                mime="text/csv",
                use_container_width=True,
            )
