# CarbonWise
**A Multi-Agent Decision-Support System for Equity-Aware Carbon Reduction in Infrastructure Investment Planning**

Chennai pilot implementation for BCSE497J Project-I, Review 2.

---

## What this is (and isn't) — read this before your review

This is a **benchmark-grounded, city-real prototype**, not a system trained on
a single downloaded "municipal capex CSV" (that dataset doesn't exist in the
form people imagine, for any city, publicly). Here's exactly what's real and
what's a first-pass estimate, so you can represent it honestly to your panel:

| Element | Status |
|---|---|
| Chennai's 15-zone administrative structure, zone names | **Real** (Greater Chennai Corporation) |
| Zone-level vulnerability_index values | **Grounded in a real, cited academic study**: Nandhini, Das & Shinde (2025, IIT Kharagpur / National Institute of Urban Affairs), *"Vulnerability and Flood Disaster Governance in Chennai, India"* — a TOPSIS-based composite physical + socio-economic vulnerability index built from Landsat 9 satellite data, 2011 Census, and RAY slum-database records. North Chennai and the western periphery (Ambattur, Valasaravakkam) are their documented highest-vulnerability zones; central/coastal zones are documented lowest. South-zone values (Alandur, Adyar, Perungudi, Sholinganallur) are grounded in separate, independently documented hydrological flood evidence (Pallikaranai marsh, Adyar river, OMR corridor flooding) since the cited study doesn't zone the deep south explicitly — see `data_source` column in `wards.csv` for the exact justification per row. |
| **Real, citable empirical motivation for equity constraints**: the same study found a Spearman correlation of **-0.49** between ward vulnerability and actual GCC flood-infrastructure capital expenditure (R²=0.32, p=0.023) — i.e., **Chennai's real infrastructure spending is currently NOT aligned with need.** Cite this directly in your Problem Statement / Research Gap. | **Real, published finding** |
| Project categories (solar, retrofit, drainage, trees, transit, flood barriers, cool roofs) | **Real intervention types** used in Indian municipal climate-action planning |
| Cost / carbon-reduction / resilience-benefit ranges per category | **Benchmark-grounded order-of-magnitude ranges** (see source list in `data/raw/generate_projects.py` docstring: MNRE, CEA, BEE, CPHEEO, FAME-II). Individual project rows are randomly sampled within these ranges. **Upgrade opportunity before Final Review:** GCC's published capital-expenditure-by-zone data (referenced in the Nandhini et al. study and on the Smart Cities Chennai portal) could replace the drainage category's synthetic costs with real line-item figures, since that's exactly what that study analyzed. |
| NSGA-II optimization, Pareto front, baseline comparisons | **Fully real and working** — this is your actual contribution |

**How to say this to your panel:** *"Ward-level vulnerability is grounded in a
peer-reviewed IIT Kharagpur/NIUA study (Nandhini et al., 2025) that built a
TOPSIS composite vulnerability index for Chennai using satellite and census
data — and that same study found a statistically significant negative
correlation between vulnerability and actual flood-infrastructure spending
in Chennai, meaning the most vulnerable wards currently receive the least
investment. That finding is direct empirical motivation for why CarbonWise
encodes equity as a constraint rather than an afterthought. Project cost and
carbon/resilience impact ranges are grounded in published Indian
infrastructure benchmarks (MNRE, CEA, CPHEEO); individual project rows are
benchmark-sampled because no single public dataset contains all four metrics
jointly for any city — refining these with GCC's real capital-expenditure
line items is explicitly scoped as near-term follow-up work."*

---

## Project structure

```
carbonwise/
├── data/
│   ├── raw/
│   │   ├── wards.csv              # 15 Chennai GCC zones + vulnerability data
│   │   ├── projects.csv           # 56 candidate infrastructure projects (8 categories)
│   │   └── generate_projects.py   # regenerate projects.csv (documents all benchmark sources)
│   └── processed/
│       └── projects_processed.csv # output of preprocessing.py
├── src/
│   ├── preprocessing.py           # cleans + joins projects with ward data
│   ├── equity_scoring.py          # the formalized equity metric
│   ├── optimizer.py                # NSGA-II via pymoo — the core engine
│   ├── baselines.py                # Equal-Split & Greedy comparison methods
│   └── visualize.py                # generates the two figures below
├── dashboard/
│   └── app.py                      # Streamlit dashboard
├── results/
│   ├── figures/
│   │   ├── pareto_front.png
│   │   └── baseline_comparison.png
│   └── portfolios/
│       └── baseline_comparison.csv
└── README.md
```

## How to run everything

```bash
pip install -r requirements.txt

# 1. Regenerate the dataset (optional — it's already generated)
cd data/raw && python3 generate_projects.py && cd ../..

# 2. Run the full pipeline end-to-end (prints results, no dashboard)
cd src
python3 preprocessing.py       # cleans + joins data
python3 equity_scoring.py      # top-10 highest-equity projects
python3 optimizer.py           # runs NSGA-II, prints Pareto front + recommended portfolio
python3 baselines.py           # the full 3-way comparison table
python3 visualize.py           # generates results/figures/*.png

# 3. Launch the interactive dashboard
cd ../dashboard
streamlit run app.py
# then open the Local URL it prints (usually http://localhost:8501)
```

## The equity metric

```
Equity_Score(project) = w1 · Vulnerability_Index(ward)
                       + w2 · Population_Benefited_Ratio(project)
                       + w3 · Infrastructure_Deprivation(ward)
```

Weights default to equal thirds; the dashboard lets you adjust them live via
sliders. This is explicitly framed (per your Review-1 Research Gap) as a
first-pass **operational, auditable** metric — not a claimed-final formula.

## Sample result (Budget = ₹50 crore, 56 candidate projects)

| Method | Carbon | Resilience | Equity | Budget used |
|---|---:|---:|---:|---:|
| Equal Split | 455 | 1233 | 1685 | ₹26.4 cr (inefficient — leaves budget unspent) |
| Greedy (cost-effectiveness) | **752** | 1340 | 1867 | ₹48.0 cr |
| **CarbonWise (NSGA-II)** | 672 | **1629** | **2013** | ₹49.8 cr |

Greedy wins on carbon alone but sacrifices resilience and equity. CarbonWise
trades some carbon reduction for substantially stronger resilience and
equity — the exact trade-off no existing single-objective tool exposes,
per your Research Gap (Papers 1, 2, 4, 6, 8).

## What's NOT yet built 

- Full multi-agent LLM orchestration (Carbon/Resilience/Equity/Budget/Coordinator
  agents as separate reasoning agents, per Papers 5 & 6) — the dashboard's
  "coordinator summary" is currently a templated sentence, not agentic reasoning
- SHAP-based explainability (per Paper 2's methodology)
- GIS/map visualization of selected projects across zones
- Refinement of exact benchmark figures against primary sources

## Dashboard v2 — Command Center upgrade

The dashboard (`dashboard/app.py`) now includes, on top of the original
budget/optimizer basics:

- **Chennai Command Map** (Plotly): click through all 15 zones, see real
  vulnerability/population/infrastructure-deprivation data per zone
- **What-If Scenario Simulator**: Max Carbon / Max Equity / Max Resilience /
  Balanced — all four drawn from the SAME Pareto front (`src/scenarios.py`)
- **Multi-Agent Council**: five agents (Carbon, Resilience, Equity, Budget,
  Coordinator) each generate a real, data-grounded statement from the actual
  optimizer output (`src/agents.py`) — templated/rule-based, not an LLM call,
  which is the right choice for a live panel demo (no API risk, no
  hallucinated numbers)
- **Equity Impact Auditor**: a Justice40-style metric (`src/equity_auditor.py`)
  showing the real % of investment flowing to high-vulnerability zones,
  before (Greedy) vs after (CarbonWise) — directly mirrors Paper 7
  (Pollack et al., 2025) from your literature review

Run it with `cd dashboard && streamlit run app.py` (or `python3 -m streamlit
run app.py` if `streamlit` isn't on your PATH).

**Not built** (explicitly scoped as Final Review / stretch work): "Ask
CarbonWise" free-text LLM chat interface. Live LLM calls in front of a panel
carry real hallucination/latency risk for uncertain payoff — flag this
honestly as planned future work rather than attempting it under time
pressure.

## Dashboard v3 — professional redesign + real decision-support features

The dashboard was redesigned to move away from a demo/prototype look toward
something closer to an actual planning tool, and to add analysis that
produces a genuinely usable output rather than only a visualization:

- **Visual language**: removed heavy emoji/gradient styling in favour of a
  restrained tile-and-finding layout (`src/agents.py` output is now rendered
  as labeled "Key findings" rather than chat bubbles)
- **Budget sensitivity analysis** (`src/sensitivity.py`, optional checkbox
  in the sidebar): re-runs the optimizer across a range of budget levels and
  reports the marginal carbon gained per additional ₹1 crore — this
  reveals diminishing returns as budget scales, a genuinely useful
  finding for a finance-department conversation, not just a chart
- **Downloadable investment brief (CSV)**: the recommended portfolio, with
  budget/weight metadata in a header, exportable directly from the dashboard
  — turning CarbonWise's output into something a planner could actually take
  into a real budget meeting
- Offline-safe zone chart remains the default (see Dashboard v2 notes below)
  for demo reliability

## Optional AI explanation layer (`src/llm/`)

CarbonWise's "Key findings" section can optionally be upgraded from
rule-based text to an LLM-generated explanation — **without changing how
the system makes decisions**. This is a deliberate design boundary:

```
NSGA-II optimizer  →  produces the actual recommended portfolio (verified math)
Equity auditor     →  produces the actual equity metrics (verified math)
                              ↓
              LLM (optional) explains these numbers in prose
```

The LLM is never given raw project data and is never allowed to pick a
different portfolio or invent a number — it only explains numbers the
optimizer already computed. If no API key is configured, or the API call
fails for any reason (no internet, invalid key, rate limit, timeout), the
system **silently and immediately falls back** to the tested rule-based
explanation in `agents.py` — the dashboard never breaks or shows an error to
the end user.

### To enable the AI layer (entirely optional)

1. Get an OpenAI API key from platform.openai.com (requires billing enabled;
   cost is a fraction of a cent per explanation with the default model)
2. `cp .env.example .env`
3. Edit `.env` and paste your key after `OPENAI_API_KEY=`
4. Run the dashboard as normal — you'll see "AI explanation: enabled" instead
   of "AI explanation: not configured" in the Investment Analysis tab

**If you don't do this, nothing breaks** — CarbonWise runs its full pipeline
exactly as before, using the rule-based explanation engine.

### What was deliberately NOT built (and why)

A much larger vision exists for this direction: a free-text "Ask CarbonWise"
chat interface, conversational control of the optimizer ("give more weight
to equity" → re-runs NSGA-II automatically), RAG retrieval over the project's
literature review, and multi-agent "debate" visualization. These were
scoped OUT of the pre-Review-2 build deliberately:

- A live chat interface depending on an external API is a real live-demo
  risk (latency, cost, rate limits, network dependency) for a feature whose
  payoff is uncertain in front of a panel
- Conversational tool-calling (LLM deciding when to re-run the optimizer)
  needs careful guardrails to avoid the LLM silently misinterpreting a
  request and running an incorrect scenario
- RAG over the literature review is valuable but is a genuinely separate,
  multi-day subsystem (chunking, embedding, vector store, retrieval testing)

These are honestly scoped as **Final Review (October) stretch goals** — say
so explicitly if asked, rather than attempting them under time pressure.

## Phase-2 additions: data validation, decision rationale, diagrams

Three further additions, all deterministic (no LLM/API dependency),
verified end-to-end:

- **`src/data_validation.py`**: a real integrity check (missing values,
  negative costs, orphaned zone references, duplicate IDs) that runs before
  any optimization. Shown in the dashboard sidebar as "Data quality: Ready".
  Lets you tell the panel: *"CarbonWise validates all candidate project
  attributes before optimisation runs."*
- **`src/decision_rationale.py`**: the structured 5-point "Why this
  portfolio?" explanation (budget constraint, carbon standing within the
  Pareto front, equity impact, explicit trade-off, leading alternative) —
  all computed directly from optimizer output, no LLM required.
- **`src/generate_diagrams.py`**: generates `results/figures/
  architecture_diagram.png` and `methodology_flowchart.png` for direct use
  in the Review-2 PPT. Every box in the architecture diagram corresponds to
  an actual file in this codebase — not an aspirational diagram.

Run `python3 generate_diagrams.py` from inside `src/` any time you want to
regenerate these (e.g. after further code changes).

## Cross-sector positioning + Investment Decision Report

Two further additions:

- **EV Charging Hub** added as an 8th project category (56 projects total,
  7 per category). Combined with the existing Solar, Public Transit, and
  Urban Tree Canopy (green corridor) categories, CarbonWise now genuinely
  spans the "four infrastructure domains" framing (EV, solar, transit,
  green space) without being restructured into separate systems — exactly
  as intended: one unified multi-objective optimiser across all categories.
- **`src/report.py`**: generates a full 10-section Investment Decision
  Report (Executive Summary, Recommended Portfolio, Budget Allocation,
  Impact Summary, Trade-off Analysis, Baseline Comparison, Equity Audit,
  Alternative Pareto Solutions, Zone-Level Breakdown, Full Project List) —
  entirely from verified computed data, no LLM. Downloadable from the
  dashboard as a `.txt` file alongside the plain CSV export.
