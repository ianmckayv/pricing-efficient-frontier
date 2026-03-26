import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ============================================================
# CA_05c_frontier_app.py
#
# Streamlit visualization for the California mortgage
# pricing efficient frontier.
#
# Run with: streamlit run CA_05c_frontier_app.py
# ============================================================

st.set_page_config(
    page_title="CA Mortgage Pricing | Efficient Frontier",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #6b7280;
        font-family: 'IBM Plex Mono', monospace;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 600;
        color: #f0f2f5;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-delta {
        font-size: 12px;
        color: #10b981;
        font-family: 'IBM Plex Mono', monospace;
        margin-top: 2px;
    }
    .metric-delta.negative { color: #ef4444; }

    .section-header {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #4b5563;
        font-family: 'IBM Plex Mono', monospace;
        padding: 8px 0 4px 0;
        border-bottom: 1px solid #1e2130;
        margin-bottom: 16px;
    }
    .scenario-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .tag-sq { background: #1e3a2f; color: #34d399; border: 1px solid #065f46; }
    .tag-a  { background: #1e2d4a; color: #60a5fa; border: 1px solid #1d4ed8; }
    .tag-b  { background: #3b1f2e; color: #f472b6; border: 1px solid #9d174d; }

    .stSelectbox label { color: #9ca3af !important; font-size: 12px !important; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: #6b7280;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #f0f2f5; }
    h1 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 300 !important; }
    h2, h3 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 400 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_resource
def load_data():
    try:
        with open("CA_frontier_results.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading CA_frontier_results.pkl: {e}")
        return None

data = load_data()

if data is None:
    st.error("Could not load CA_frontier_results.pkl. Run CA_05b_frontier_optimize.py first.")
    st.stop()

# Handle both old format (frontier_df) and new format (frontier_records)
if 'frontier_records' in data:
    frontier_df = pd.DataFrame(data['frontier_records'])
elif 'frontier_df' in data:
    frontier_df = data['frontier_df'].copy()
    if isinstance(frontier_df, list):
        frontier_df = pd.DataFrame(frontier_df)
else:
    st.error("Unrecognized PKL format. Rerun CA_05b_frontier_optimize.py.")
    st.stop()

frontier_solutions = data['frontier_solutions']
status_quo = data['status_quo']
R_max = data['R_max']
V_max_revenue = data['V_max_revenue']

# Normalize column name
if 'avg_rate' in frontier_df.columns and 'wtd_avg_rate' not in frontier_df.columns:
    frontier_df['wtd_avg_rate'] = frontier_df['avg_rate']

SEGMENTS = sorted([s for s in frontier_df.columns
                   if s not in frontier_df.columns]
                  + ['S1','S2','S3','S4','S5','S6','S7','S8'])

def get_solution(run_id):
    """Extract solution dataframe for a given run."""
    run_data = frontier_solutions.get(run_id)
    if run_data is None:
        return None
    if isinstance(run_data, dict):
        return run_data.get('solution', run_data.get('seg_profile'))
    return run_data

def weighted_avg(series, weights):
    valid = series.notna() & weights.notna() & (weights > 0)
    if valid.sum() == 0:
        return series.mean()
    return (series[valid] * weights[valid]).sum() / weights[valid].sum()

def get_scenario_stats(solution_df):
    if solution_df is None or len(solution_df) == 0:
        return {}
    rev = solution_df['expected_revenue'].sum()
    vol = solution_df['expected_volume'].sum()
    wtd_rate = weighted_avg(solution_df['test_rate'], solution_df['expected_volume'])
    wtd_spread = weighted_avg(
        solution_df.get('relative_spread_test', solution_df.get('relative_spread_test', pd.Series([np.nan]*len(solution_df)))),
        solution_df['expected_volume']
    )
    return {
        'total_revenue': rev,
        'total_volume': vol,
        'wtd_avg_rate': wtd_rate,
        'wtd_avg_spread': wtd_spread,
        'n_clients': len(solution_df),
    }

# ============================================================
# HEADER
# ============================================================
col_title, col_info = st.columns([3, 1])
with col_title:
    st.markdown("# Pricing Efficient Frontier: The Trade-Off Between Contribution Margin and Volume")
    st.markdown(
        '<p style="color:#6b7280;font-size:13px;margin-top:-12px;">'
        'Ian McKay &nbsp;·&nbsp; Data: HMDA 2023 Public LAR &nbsp;·&nbsp; '
        'Elasticity model adjusted to enhance price sensitivities for illustration</p>',
        unsafe_allow_html=True
    )
with col_info:
    st.markdown(
        f'<div style="text-align:right;color:#4b5563;font-size:11px;'
        f'font-family:IBM Plex Mono,monospace;padding-top:20px;">'
        f'{len(frontier_df)} frontier points<br>'
        f'2,000 simulated clients<br>'
        f'8 market segments</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ============================================================
# INTRODUCTION — placeholder
# ============================================================
with st.expander("INTRODUCTION", expanded=False):
    st.markdown(
        """
Business leaders know they need to balance margin and volume. What they usually lack is a framework to understand the trade-off, and what pricing can and can't do.

In B2C lending and B2B SaaS, you rarely have a single price per segment. You have a list price, a range the sales rep can negotiate, and a threshold where deal desk takes over. In these cases, a target average price with defined guardrails works better than a fixed number. The efficient frontier approach simulates individual clients, captures variation in willingness to pay, and allows pricing flexibility at the segment level.

This tool shows that trade-off at a macro level, to allow business leaders to decide on a scenario that fits their strategy. Each point on the curve is a feasible pricing scenario across 8 segments. Moving along the curve is a choice: grow volume by pricing more competitively, or protect margin by pricing tighter. The curve tells you exactly what each choice costs.

A few things worth noting before you explore the scenarios. The optimization works at the segment level, not the individual client level. Within each segment, willingness to pay varies. The output is a distribution of prices, not a single number. In this example, the 40th percentile of that distribution is the deal desk threshold. The 95th percentile is the list price. Everything in between is negotiation space for the sales team.

This matters because a single optimal price per segment is not deployable in practice. Clients have different profiles, brokers have different relationships, sales reps have different incentives. Ranges with guardrails are how pricing actually gets implemented.

The underlying methodology mirrors what commercial pricing vendors deploy at large financial institutions, and what large banks build in-house. This demo runs on publicly available HMDA 2023 data from the CFPB. The value is in the sequencing, the design choices, and knowing what to build when. And sometimes in knowing when the data runs out: gaps in data are real, and experience and empirical judgment are legitimate inputs when the model can't get you all the way there.
        """
    )

# ============================================================
# SEGMENT TREE
# ============================================================
with st.expander("MARKET SEGMENTS — How the 8 segments are defined", expanded=False):
    st.markdown(
        '<p style="color:#6b7280;font-size:11px;font-family:IBM Plex Mono;">'
        'Segments defined by a regression tree on originated loans predicting rate spread. '
        'LTV is the primary split.'
        '</p>', unsafe_allow_html=True
    )
    tree_html = """
    <div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#9ca3af;line-height:2;padding:12px 0;">
      <div style="color:#f0f2f5;font-size:12px;margin-bottom:8px;">ROOT: Combined LTV ≤ 80%?</div>
      <div style="display:flex;gap:40px;">
        <div style="border-left:2px solid #1d4ed8;padding-left:16px;flex:1;">
          <div style="color:#60a5fa;margin-bottom:4px;">YES — Low LTV (&lt;80%)</div>
          <div style="border-left:2px solid #374151;padding-left:16px;">
            Property Value ≤ $440k?
            <div style="display:flex;gap:24px;margin-top:4px;">
              <div style="border-left:2px solid #374151;padding-left:12px;">
                YES — Lower value<br>
                Income ≤ $116.5k?<br>
                <span style="color:#34d399;">YES → <b>S4</b> avg +0.15pp<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Low LTV · Low value · Low income</span><br>
                <span style="color:#f59e0b;">NO &nbsp;→ <b>S7</b> avg +0.45pp<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Low LTV · Low value · High income</span>
              </div>
              <div style="border-left:2px solid #374151;padding-left:12px;">
                NO — Higher value<br>
                Broker channel?<br>
                <span style="color:#34d399;">NO &nbsp;→ <b>S1</b> avg -0.05pp ⭐ cheapest<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Low LTV · High value · Direct</span><br>
                <span style="color:#f59e0b;">YES → <b>S2</b> avg +0.07pp<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Low LTV · High value · Broker</span>
              </div>
            </div>
          </div>
        </div>
        <div style="border-left:2px solid #9d174d;padding-left:16px;flex:1;">
          <div style="color:#f472b6;margin-bottom:4px;">NO — High LTV (&gt;80%)</div>
          <div style="border-left:2px solid #374151;padding-left:16px;">
            LTV ≤ 90%?
            <div style="display:flex;gap:24px;margin-top:4px;">
              <div style="border-left:2px solid #374151;padding-left:12px;">
                YES — Moderate LTV<br>
                Broker channel?<br>
                <span style="color:#34d399;">NO &nbsp;→ <b>S3</b> avg +0.12pp<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mid LTV · Direct channel</span><br>
                <span style="color:#f59e0b;">YES → <b>S5</b> avg +0.31pp<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mid LTV · Broker channel</span>
              </div>
              <div style="border-left:2px solid #374151;padding-left:12px;">
                NO — High LTV (&gt;90%)<br>
                Income ≤ $176.5k?<br>
                <span style="color:#34d399;">YES → <b>S6</b> avg +0.32pp<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;High LTV · Moderate income</span><br>
                <span style="color:#f59e0b;">NO &nbsp;→ <b>S8</b> avg +0.49pp 🔴 priciest<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;High LTV · High income</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div style="margin-top:16px;color:#4b5563;font-size:10px;">
        Rate spread shown as pp above APOR. Broker = loan originated through broker channel.
        S1: high-value homes, low LTV, direct channel — most competitive market segment.
        S8: high LTV, high income, broker — least price sensitive segment.
      </div>
    </div>
    """
    st.markdown(tree_html, unsafe_allow_html=True)

# ============================================================
# PRICE DEFINITION
# ============================================================
with st.expander("PRICE DEFINITION — How price sensitivity is measured", expanded=False):
    st.markdown(
        """
        Several measures of price were evaluated during model development.
        Absolute interest rates were ruled out as a price variable because they
        are driven largely by macroeconomic conditions — APOR moves with the Fed,
        and a borrower's decision to accept or reject a loan offer depends not on
        the absolute rate but on how competitive that rate is relative to what
        the market offers for similar loans.

        **The price variable used in this model is the APR spread above APOR,
        measured relative to the segment average.**

        For each borrower, the price signal is:

        > *relative spread = client rate spread − average rate spread for their segment*

        A negative value means the borrower is being offered a below-market rate for
        their segment. A positive value means they are being priced above market.
        This definition controls for segment-level pricing differences and isolates
        the pure price sensitivity of each individual borrower.

        The elasticity model estimates the probability of acceptance as a function
        of this relative spread. The further above the segment average a borrower
        is priced, the lower their estimated probability of accepting the offer.

        *Note: win probabilities have been recalibrated from the post-approval
        observed data to represent full-funnel pre-application conversion rates,
        consistent with a realistic commercial deployment of this methodology.*
        """
    )

# ============================================================
# SIDEBAR — SCENARIO SELECTION
# ============================================================
with st.sidebar:
    st.markdown('<div class="section-header">SCENARIO SELECTION</div>',
                unsafe_allow_html=True)

    SQ_rev = status_quo.get('total_revenue', 0) if isinstance(status_quo, dict) else 0
    SQ_vol = status_quo.get('total_volume', 0) if isinstance(status_quo, dict) else 0

    run_options = {}
    for _, row in frontier_df.iterrows():
        rev_pct = ((row['total_revenue'] - SQ_rev) / SQ_rev * 100) if SQ_rev > 0 else 0
        vol_pct = ((row['total_volume'] - SQ_vol) / SQ_vol * 100) if SQ_vol > 0 else 0
        label = (f"Run {int(row['run'])} — "
                 f"Margin {rev_pct:+.1f}% vs SQ | "
                 f"Vol {vol_pct:+.1f}% vs SQ | "
                 f"Price {row['wtd_avg_rate']:.2f}%")
        run_options[label] = int(row['run'])

    st.markdown('<span class="scenario-tag tag-a">SCENARIO A</span>',
                unsafe_allow_html=True)
    scenario_a_label = st.selectbox(
        "Select Scenario A", options=list(run_options.keys()),
        index=0, key="scenario_a"
    )
    scenario_a_run = run_options[scenario_a_label]

    st.markdown('<span class="scenario-tag tag-b">SCENARIO B</span>',
                unsafe_allow_html=True)
    scenario_b_label = st.selectbox(
        "Select Scenario B", options=list(run_options.keys()),
        index=min(7, len(run_options)-1), key="scenario_b"
    )
    scenario_b_run = run_options[scenario_b_label]

# Get solutions
sol_sq = status_quo['solution'] if isinstance(status_quo, dict) else None
sol_a = get_solution(scenario_a_run)
sol_b = get_solution(scenario_b_run)

stats_sq = {
    'total_revenue': status_quo.get('total_revenue', 0) if isinstance(status_quo, dict) else 0,
    'total_volume': status_quo.get('total_volume', 0) if isinstance(status_quo, dict) else 0,
    'wtd_avg_rate': status_quo.get('wtd_avg_rate', 0) if isinstance(status_quo, dict) else 0,
}
stats_a = get_scenario_stats(sol_a)
stats_b = get_scenario_stats(sol_b)

# ============================================================
# EFFICIENT FRONTIER CHART
# ============================================================
st.markdown('<div class="section-header">EFFICIENT FRONTIER</div>',
            unsafe_allow_html=True)

fig_frontier = go.Figure()

# Frontier curve — Volume on X, Contribution Margin on Y
fig_frontier.add_trace(go.Scatter(
    x=frontier_df['total_volume'],
    y=frontier_df['total_revenue'],
    mode='lines+markers',
    line=dict(color='#374151', width=2),
    marker=dict(size=8, color='#4b5563', line=dict(color='#6b7280', width=1)),
    name='Frontier',
    hovertemplate=(
        '<b>Run %{customdata[0]}</b><br>'
        'Volume: $%{x:,.0f}<br>'
        'Contribution Margin: $%{y:,.0f}<br>'
        'Wtd Avg Price: %{customdata[1]:.2f}%<br>'
        'Margin vs SQ: %{customdata[2]:+.1f}%<br>'
        'Volume vs SQ: %{customdata[3]:+.1f}%<br>'
        '<extra></extra>'
    ),
    customdata=np.stack([
        frontier_df['run'].values,
        frontier_df['wtd_avg_rate'].values,
        np.round((frontier_df['total_revenue'] - stats_sq['total_revenue']) / stats_sq['total_revenue'] * 100, 1)
        if stats_sq['total_revenue'] > 0 else np.zeros(len(frontier_df)),
        np.round((frontier_df['total_volume'] - stats_sq['total_volume']) / stats_sq['total_volume'] * 100, 1)
        if stats_sq['total_volume'] > 0 else np.zeros(len(frontier_df)),
    ], axis=1)
))

# Highlight selected scenarios
for run_id, color, name, symbol in [
    (scenario_a_run, '#60a5fa', 'Scenario A', 'circle'),
    (scenario_b_run, '#f472b6', 'Scenario B', 'circle'),
]:
    row = frontier_df[frontier_df['run'] == run_id]
    if len(row) > 0:
        fig_frontier.add_trace(go.Scatter(
            x=row['total_volume'],
            y=row['total_revenue'],
            mode='markers',
            marker=dict(size=14, color=color, symbol=symbol,
                       line=dict(color='white', width=2)),
            name=name,
            hovertemplate=(
                f'<b>{name}</b><br>'
                'Volume: $%{x:,.0f}<br>'
                'Contribution Margin: $%{y:,.0f}<br>'
                '<extra></extra>'
            )
        ))

# Status quo point
if stats_sq['total_revenue'] > 0:
    fig_frontier.add_trace(go.Scatter(
        x=[stats_sq['total_volume']],
        y=[stats_sq['total_revenue']],
        mode='markers',
        marker=dict(size=14, color='#34d399', symbol='x',
                   line=dict(color='white', width=2)),
        name='Status Quo',
        hovertemplate=(
            '<b>Status Quo</b><br>'
            'Volume: $%{x:,.0f}<br>'
            'Contribution Margin: $%{y:,.0f}<br>'
            '<extra></extra>'
        )
    ))

fig_frontier.update_layout(
    paper_bgcolor='#0f1117',
    plot_bgcolor='#0f1117',
    font=dict(family='IBM Plex Mono', color='#9ca3af', size=11),
    xaxis=dict(
        title='Total Expected Volume ($)',
        gridcolor='#1e2130', zeroline=False,
        tickformat='$,.0f', title_font=dict(size=11)
    ),
    yaxis=dict(
        title='Total Contribution Margin ($)',
        gridcolor='#1e2130', zeroline=False,
        tickformat='$,.0f', title_font=dict(size=11)
    ),
    legend=dict(
        bgcolor='#1a1d27', bordercolor='#2a2d3a', borderwidth=1,
        font=dict(size=11)
    ),
    margin=dict(l=60, r=20, t=20, b=60),
    height=380,
)

st.plotly_chart(fig_frontier, width='stretch')

# ============================================================
# COMPARISON METRICS
# ============================================================
st.markdown('<div class="section-header">SCENARIO COMPARISON</div>',
            unsafe_allow_html=True)

col_sq, col_a, col_b = st.columns(3)

def metric_card(label, value, delta=None, delta_positive=True):
    delta_html = ""
    if delta is not None:
        cls = "metric-delta" if delta_positive else "metric-delta negative"
        delta_html = f'<div class="{cls}">{delta}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

with col_sq:
    st.markdown('<span class="scenario-tag tag-sq">STATUS QUO</span>',
                unsafe_allow_html=True)
    st.markdown(metric_card("Total Contribution Margin",
                f"${stats_sq['total_revenue']/1e6:.2f}M"), unsafe_allow_html=True)
    st.markdown(metric_card("Total Volume",
                f"${stats_sq['total_volume']/1e6:.0f}M"), unsafe_allow_html=True)
    st.markdown(metric_card("Wtd Avg Price",
                f"{stats_sq['wtd_avg_rate']:.2f}%"), unsafe_allow_html=True)

with col_a:
    st.markdown('<span class="scenario-tag tag-a">SCENARIO A</span>',
                unsafe_allow_html=True)
    if stats_a:
        rev_delta_a = (stats_a['total_revenue'] - stats_sq['total_revenue'])
        vol_delta_a = (stats_a['total_volume'] - stats_sq['total_volume'])
        st.markdown(metric_card("Total Contribution Margin",
            f"${stats_a['total_revenue']/1e6:.2f}M",
            f"{'+' if rev_delta_a >= 0 else ''}{rev_delta_a/1e3:,.0f}K vs SQ",
            rev_delta_a >= 0), unsafe_allow_html=True)
        st.markdown(metric_card("Total Volume",
            f"${stats_a['total_volume']/1e6:.0f}M",
            f"{'+' if vol_delta_a >= 0 else ''}{vol_delta_a/1e3:,.0f}K vs SQ",
            vol_delta_a >= 0), unsafe_allow_html=True)
        st.markdown(metric_card("Wtd Avg Price",
            f"{stats_a['wtd_avg_rate']:.2f}%"), unsafe_allow_html=True)

with col_b:
    st.markdown('<span class="scenario-tag tag-b">SCENARIO B</span>',
                unsafe_allow_html=True)
    if stats_b:
        rev_delta_b = (stats_b['total_revenue'] - stats_sq['total_revenue'])
        vol_delta_b = (stats_b['total_volume'] - stats_sq['total_volume'])
        st.markdown(metric_card("Total Contribution Margin",
            f"${stats_b['total_revenue']/1e6:.2f}M",
            f"{'+' if rev_delta_b >= 0 else ''}{rev_delta_b/1e3:,.0f}K vs SQ",
            rev_delta_b >= 0), unsafe_allow_html=True)
        st.markdown(metric_card("Total Volume",
            f"${stats_b['total_volume']/1e6:.0f}M",
            f"{'+' if vol_delta_b >= 0 else ''}{vol_delta_b/1e3:,.0f}K vs SQ",
            vol_delta_b >= 0), unsafe_allow_html=True)
        st.markdown(metric_card("Wtd Avg Price",
            f"{stats_b['wtd_avg_rate']:.2f}%"), unsafe_allow_html=True)

# ============================================================
# SEGMENT BREAKDOWN
# ============================================================
st.markdown("---")
st.markdown('<div class="section-header">SEGMENT BREAKDOWN</div>',
            unsafe_allow_html=True)

def get_segment_profile(solution_df):
    if solution_df is None or len(solution_df) == 0:
        return pd.DataFrame()
    total_vol = solution_df['expected_volume'].sum()
    total_rev = solution_df['expected_revenue'].sum()
    seg_col = 'segment' if 'segment' in solution_df.columns else 'segment_label_x'
    if seg_col not in solution_df.columns:
        return pd.DataFrame()
    profile = (
        solution_df.groupby(seg_col)
        .apply(lambda g: pd.Series({
            'n': len(g),
            'pct_volume': g['expected_volume'].sum() / total_vol * 100,
            'pct_revenue': g['expected_revenue'].sum() / total_rev * 100,
            'wtd_avg_rate': weighted_avg(g['test_rate'], g['expected_volume']),
            'p40_rate': g['test_rate'].quantile(0.40),
            'p95_rate': g['test_rate'].quantile(0.95),
            'total_volume': g['expected_volume'].sum(),
            'total_revenue': g['expected_revenue'].sum(),
            'rates': g['test_rate'].tolist(),
        }))
        .reset_index()
        .rename(columns={seg_col: 'segment'})
        .sort_values('segment')
    )
    return profile

profile_sq = get_segment_profile(sol_sq)
profile_a = get_segment_profile(sol_a)
profile_b = get_segment_profile(sol_b)

# Get all segments present
all_segs = sorted(set(
    list(profile_a['segment'].unique() if len(profile_a) > 0 else []) +
    list(profile_b['segment'].unique() if len(profile_b) > 0 else [])
))

if all_segs:
    tabs = st.tabs([f"  {s}  " for s in all_segs])

    for tab, seg in zip(tabs, all_segs):
        with tab:
            col_left, col_right = st.columns([1, 2])

            with col_left:
                st.markdown(f"**Segment {seg}**")

                for label, profile, color in [
                    ("Status Quo", profile_sq, "#34d399"),
                    ("Scenario A", profile_a, "#60a5fa"),
                    ("Scenario B", profile_b, "#f472b6"),
                ]:
                    if len(profile) == 0:
                        continue
                    row = profile[profile['segment'] == seg]
                    if len(row) == 0:
                        continue
                    row = row.iloc[0]
                    st.markdown(
                        f'<div style="border-left:3px solid {color};'
                        f'padding:8px 12px;margin:6px 0;background:#1a1d27;'
                        f'border-radius:0 6px 6px 0;">'
                        f'<div style="color:{color};font-size:10px;'
                        f'font-family:IBM Plex Mono;letter-spacing:0.1em;">'
                        f'{label.upper()}</div>'
                        f'<div style="color:#f0f2f5;font-size:13px;'
                        f'font-family:IBM Plex Mono;margin-top:4px;">'
                        f'Wtd Price: {row["wtd_avg_rate"]:.2f}%</div>'
                        f'<div style="color:#9ca3af;font-size:11px;">'
                        f'P40: {row["p40_rate"]:.2f}% (deal desk threshold)</div>'
                        f'<div style="color:#9ca3af;font-size:11px;">'
                        f'P95: {row["p95_rate"]:.2f}% (suggested list price)</div>'
                        f'<div style="color:#9ca3af;font-size:11px;">'
                        f'Vol: {row["pct_volume"]:.1f}% &nbsp;|&nbsp; '
                        f'Margin: {row["pct_revenue"]:.1f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            with col_right:
                # Price distribution histogram
                fig_hist = go.Figure()

                for label, profile, color in [
                    ("Scenario A", profile_a, "#60a5fa"),
                    ("Scenario B", profile_b, "#f472b6"),
                ]:
                    if len(profile) == 0:
                        continue
                    row = profile[profile['segment'] == seg]
                    if len(row) == 0:
                        continue
                    rates = row.iloc[0]['rates']
                    if not rates:
                        continue

                    fig_hist.add_trace(go.Histogram(
                        x=rates,
                        name=label,
                        nbinsx=10,
                        marker_color=color,
                        opacity=0.6,
                        histnorm='probability',
                    ))

                    # P40 and P95 lines
                    p40 = row.iloc[0]['p40_rate']
                    p95 = row.iloc[0]['p95_rate']

                    fig_hist.add_vline(
                        x=p40, line_dash='dash',
                        line_color=color, opacity=0.8,
                        annotation_text=f"P40 {p40:.2f}%",
                        annotation_font=dict(size=10, color=color),
                        annotation_position="top right"
                    )
                    fig_hist.add_vline(
                        x=p95, line_dash='dot',
                        line_color=color, opacity=0.8,
                        annotation_text=f"P95 {p95:.2f}%",
                        annotation_font=dict(size=10, color=color),
                        annotation_position="top left"
                    )

                fig_hist.update_layout(
                    paper_bgcolor='#0f1117',
                    plot_bgcolor='#0f1117',
                    font=dict(family='IBM Plex Mono', color='#9ca3af', size=10),
                    xaxis=dict(
                        title='Interest Rate (%)',
                        gridcolor='#1e2130', zeroline=False,
                        ticksuffix='%'
                    ),
                    yaxis=dict(
                        title='Probability',
                        gridcolor='#1e2130', zeroline=False
                    ),
                    legend=dict(
                        bgcolor='#1a1d27', bordercolor='#2a2d3a',
                        borderwidth=1, font=dict(size=10)
                    ),
                    barmode='overlay',
                    margin=dict(l=40, r=20, t=30, b=40),
                    height=260,
                    title=dict(
                        text=f'Price Distribution — Segment {seg}',
                        font=dict(size=11, color='#6b7280'),
                        x=0
                    )
                )
                st.plotly_chart(fig_hist, width='stretch')

                # Corner solution warning
                for label, profile in [("Scenario A", profile_a),
                                        ("Scenario B", profile_b)]:
                    if len(profile) == 0:
                        continue
                    row = profile[profile['segment'] == seg]
                    if len(row) == 0:
                        continue
                    rates = row.iloc[0]['rates']
                    if len(rates) > 0:
                        unique_rates = len(set(round(r, 2) for r in rates))
                        if unique_rates <= 2:
                            st.markdown(
                                f'<p style="color:#f59e0b;font-size:10px;'
                                f'font-family:IBM Plex Mono;margin-top:-4px;">'
                                f'⚠ {label}: corner solution detected — '
                                f'all clients priced at {unique_rates} point(s). '
                                f'Consider widening price range for this segment.</p>',
                                unsafe_allow_html=True
                            )

                st.markdown(
                    '<p style="color:#4b5563;font-size:10px;font-family:'
                    'IBM Plex Mono;margin-top:-8px;">'
                    'Dashed = P40 deal desk threshold &nbsp;|&nbsp; '
                    'Dotted = P95 list price</p>',
                    unsafe_allow_html=True
                )

# ============================================================
# DATASET
# ============================================================
st.markdown("---")
with st.expander("DATASET — Source data and filters applied", expanded=False):
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("""
**Source**
[HMDA 2023 Public Loan Application Register (LAR)](https://ffiec.cfpb.gov/data-publication/2023/nationwide)
Consumer Financial Protection Bureau (CFPB) / FFIEC
Published annually under the Home Mortgage Disclosure Act

**Geographic filter**
- State: California only

**Loan filters**
- Action taken: Loan originated (1) or Approved, not accepted (2)
- Loan purpose: Home purchase only
- Loan type: Conventional (not FHA/VA/USDA)
- Conforming loan limit: Loan amount ≤ conforming limit
- Lien status: First lien only
- Occupancy: Principal residence
        """)
    with col_d2:
        st.markdown("""
**Rate filters**
- Interest rate: 4.99% – 9.70%
- Rate spread: -1.50 to +2.87 pp above APOR
- Loan amount: $65,000 – $725,000 (winsorized)
- Intro rate period: Null (fixed rate only)

**Variables excluded**
- Race, ethnicity, sex, age (fair lending compliance)
- Tract-to-MSA income ratio (redlining proxy risk)
- FFIEC median family income (neighborhood income)

**Sample size**
- Raw CA conforming purchase loans: ~174,000
- After filters: ~119,000 originated + approved-not-accepted
- Simulated client universe for optimization: 2,000 (stratified random sample)

**Note:** This is public regulatory data. Any institution
can replicate this analysis on their own internal data,
which would yield sharper elasticity estimates.
        """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    '<p style="color:#374151;font-size:10px;font-family:IBM Plex Mono;'
    'text-align:center;">'
    'HMDA 2023 Public LAR · California Conventional Purchase Mortgages · '
    'Win probabilities calibrated to pre-application conversion rates · '
    'For methodology demonstration purposes</p>',
    unsafe_allow_html=True
)