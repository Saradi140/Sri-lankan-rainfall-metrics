import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Sri Lanka Rainfall: Ensemble Learning Dashboard", page_icon="🌧️", layout="wide")
ART = "artifacts"
FAMCOL = {"Baseline": "#9aa0a6", "Bagging": "#2a9d8f", "Boosting": "#e76f51", "Stacking": "#6a4c93",
          "Hybrid": "#f4a261", "Time series": "#457b9d", "Neural net": "#8d99ae"}
SEASONS = {0: "Inter-monsoon", 1: "SW monsoon (May-Sep)", 2: "NE monsoon (Dec-Feb)"}


@st.cache_data
def load():
    meta = json.load(open(f"{ART}/meta.json"))
    return dict(
        meta=meta,
        metrics=pd.read_csv(f"{ART}/metrics.csv"),
        ci=pd.read_csv(f"{ART}/bootstrap_ci.csv", index_col=0),
        pred=pd.read_csv(f"{ART}/predictions.csv", parse_dates=["date"]),
        fc=pd.read_csv(f"{ART}/latest_forecast.csv", parse_dates=["date"]),
        hist=pd.read_csv(f"{ART}/history.csv", parse_dates=["date"]),
        clim=pd.read_csv(f"{ART}/climatology.csv"),
        state=pd.read_csv(f"{ART}/latest_state.csv"),
        imp=pd.read_csv(f"{ART}/feature_importance.csv"),
    )


@st.cache_resource
def load_models(files):
    m = {n: joblib.load(p) for n, p in files.items()}
    return m, joblib.load("models/stack_meta.joblib"), joblib.load("models/rf_ann_residual.joblib")


D = load()
meta, metrics, pred, fc, hist = D["meta"], D["metrics"], D["pred"], D["fc"], D["hist"]
name2code = {v: k for k, v in meta["districts"].items()}
ranked = metrics[metrics.Family != "Baseline"]
best = ranked.iloc[0]

st.sidebar.title("🌧️ Rainfall Ensembles")
st.sidebar.caption("Sri Lanka · WFP/CHIRPS dekadal rainfall · 25 districts")
page = st.sidebar.radio("Explore", ["Overview", "Model arena", "Backtest viewer", "Next-dekad outlook", "Live what-if predictor", "Insights"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Data to {meta['last_observed']} · hold-out from {meta['test_start']}")

# ------------------------------------------------------------------ Overview
if page == "Overview":
    st.title("Forecasting Sri Lankan rainfall with ensemble learning")
    st.write("Can bagging, boosting, stacking and hybrid models forecast the next 10-day (dekadal) rainfall total for each district, "
             "and do they beat the long-term average? All models are scored on a **time-based hold-out** they never saw during training.")
    c = st.columns(4)
    c[0].metric("Districts", 25)
    c[1].metric("Dekads of data", int(hist.groupby("PCODE").size().iloc[0]))
    c[2].metric("Best model (RMSE)", best.Model, f"{best.RMSE:.1f} mm")
    c[3].metric("Skill vs climatology", f"{best['Skill vs climatology']:.1%}")
    left, right = st.columns([3, 2])
    with left:
        d = st.selectbox("District", sorted(name2code), index=sorted(name2code).index("Colombo"))
        h = hist[hist.PCODE == name2code[d]]
        fig = go.Figure()
        fig.add_scatter(x=h.date, y=h.rfh, name="Observed", line=dict(color="#2a9d8f"))
        fig.add_scatter(x=h.date, y=h.rfh_avg, name="Long-term average", line=dict(color="#555", dash="dash"))
        fig.update_layout(height=340, margin=dict(t=30, b=10), yaxis_title="mm per dekad", title=f"{d}: dekadal rainfall")
        st.plotly_chart(fig, width='stretch')
    with right:
        ann = (hist.groupby("district")["rfh"].mean() * 36).sort_values()
        fig = px.bar(ann, orientation="h", color=ann, color_continuous_scale="Teal", labels={"value": "mm / year", "district": ""})
        fig.update_layout(height=340, margin=dict(t=30, b=10), coloraxis_showscale=False, title="Approx. mean annual rainfall")
        st.plotly_chart(fig, width='stretch')
    st.info("**Why Sri Lanka is interesting:** two monsoons (SW May-Sep, NE Dec-Feb) and two inter-monsoon periods create a strong seasonal cycle, "
            "and a sharp wet-zone (south-west) vs dry-zone (north, east, south-east) contrast.")

# ------------------------------------------------------------------ Model arena
elif page == "Model arena":
    st.title("Model arena: bagging vs boosting vs stacking vs hybrids")
    fams = st.multiselect("Families", list(FAMCOL), default=list(FAMCOL))
    m = metrics[metrics.Family.isin(fams)].sort_values("RMSE", ascending=False)
    clim_rmse = metrics.loc[metrics.Model.str.startswith("Climatology"), "RMSE"].iloc[0]
    fig = px.bar(m, x="RMSE", y="Model", color="Family", color_discrete_map=FAMCOL, orientation="h", height=520)
    fig.add_vline(x=clim_rmse, line_dash="dash", annotation_text="climatology")
    fig.update_layout(margin=dict(t=20), xaxis_title="RMSE on hold-out (mm per dekad), lower is better")
    st.plotly_chart(fig, width='stretch')
    st.dataframe(metrics.drop(columns=["Train time (s)"]).style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R2": "{:.3f}", "Skill vs climatology": "{:+.1%}"}),
                 width='stretch', hide_index=True)
    st.subheader("Best of each family")
    st.dataframe(ranked.groupby("Family").first().sort_values("RMSE")[["Model", "RMSE", "MAE", "R2", "Skill vs climatology"]].round(3), width='stretch')
    st.subheader("Are the differences real? Bootstrap 95% intervals on RMSE")
    st.dataframe(D["ci"].round(2), width='stretch')
    st.caption("The intervals overlap heavily: with about 49 hold-out dekads, the top models are statistically hard to separate. "
               "Skill vs climatology = 1 - RMSE / RMSE of the long-term average (positive is better).")

# ------------------------------------------------------------------ Backtest
elif page == "Backtest viewer":
    st.title("Backtest viewer: forecast vs reality on unseen data")
    c1, c2 = st.columns([1, 3])
    d = c1.selectbox("District", ["All districts (mean)"] + sorted(name2code))
    cols = [c for c in pred.columns if c not in ("date", "PCODE", "district", "Actual")]
    pick = c2.multiselect("Models", cols, default=["Climatology (long-term avg)", best.Model, "Stacking (Ridge meta)", "Hybrid: SARIMAX + RF"])
    p = pred if d.startswith("All") else pred[pred.PCODE == name2code[d]]
    p = p.groupby("date")[["Actual"] + pick].mean().reset_index()
    fig = go.Figure(go.Scatter(x=p.date, y=p.Actual, name="Actual", line=dict(color="black", width=3)))
    for n in pick: fig.add_scatter(x=p.date, y=p[n], name=n, mode="lines")
    fig.update_layout(height=430, yaxis_title="mm per dekad", margin=dict(t=20), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width='stretch')
    pp = pred if d.startswith("All") else pred[pred.PCODE == name2code[d]]
    rows = [{"Model": n, "RMSE": np.sqrt(((pp.Actual - pp[n]) ** 2).mean()), "MAE": (pp.Actual - pp[n]).abs().mean()} for n in pick]
    st.dataframe(pd.DataFrame(rows).sort_values("RMSE").round(2), hide_index=True)

# ------------------------------------------------------------------ Next-dekad outlook
elif page == "Next-dekad outlook":
    st.title(f"Outlook for the dekad starting {meta['next_dekad']}")
    cols = [c for c in fc.columns if c not in ("date", "PCODE", "district")]
    pick = st.selectbox("Model", cols, index=cols.index(best.Model))
    o = fc[["district", pick, "Climatology (long-term avg)"]].copy()
    o["Difference vs normal (%)"] = (o[pick] / o["Climatology (long-term avg)"].replace(0, np.nan) - 1) * 100
    fig = px.bar(o.sort_values(pick), x=pick, y="district", orientation="h", height=650, labels={pick: "Forecast (mm in 10 days)", "district": ""},
                 color="Difference vs normal (%)", color_continuous_scale="RdBu", color_continuous_midpoint=0)
    st.plotly_chart(fig, width='stretch')
    st.caption("Experimental model output from a 4.8-year dataset. It is not an official forecast. For decisions use the Department of Meteorology.")

# ------------------------------------------------------------------ Live predictor
elif page == "Live what-if predictor":
    st.title("What-if predictor")
    st.write("Models re-fitted on all data. Pick a district and a target dekad, adjust recent rainfall, and compare every model's forecast.")
    models, stack_meta, ann_res = load_models(meta["model_files"])
    c = st.columns(4)
    d = c[0].selectbox("District", sorted(name2code), index=sorted(name2code).index("Kandy"))
    code = name2code[d]
    nd = pd.Timestamp(meta["next_dekad"])
    month = c[1].selectbox("Target month", list(range(1, 13)), index=nd.month - 1, format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"))
    dk = c[2].selectbox("Dekad of month", [0, 1, 2], index=min(nd.day // 10, 2), format_func=lambda i: ["1st-10th", "11th-20th", "21st-end"][i])
    dekad = (month - 1) * 3 + dk
    h = hist[hist.PCODE == code].rfh.values
    st_ = D["state"][D["state"].PCODE == code].iloc[0]
    lag_def = [float(h[-1]), float(h[-2]), float(h[-3]), float(h[-4])]
    st.markdown("**Rainfall in the previous four dekads (mm)**, prefilled from the latest data")
    lc = st.columns(4)
    lags = [lc[i].number_input(f"lag{i+1} ({'last dekad' if i == 0 else f'{i+1} dekads ago'})", 0.0, 600.0, round(lag_def[i], 1), 1.0) for i in range(4)]
    clim_t = float(D["clim"][(D["clim"].PCODE == code) & (D["clim"].dekad == dekad)].rfh_avg.iloc[0])
    ang = 2 * np.pi * dekad / 36
    mon = 1 if month in (5, 6, 7, 8, 9) else 2 if month in (12, 1, 2) else 0
    row = dict(lag1=lags[0], lag2=lags[1], lag3=lags[2], lag4=lags[3], roll3=np.mean(lags[:3]), roll6=np.mean(lags + [h[-5], h[-6]]),
               r1h_l1=st_.r1h, r3h_l1=st_.r3h, rfq_l1=st_.rfq, r1q_l1=st_.r1q, r3q_l1=st_.r3q, clim_t=clim_t, clim_mean=st_.clim_mean,
               sin1=np.sin(ang), cos1=np.cos(ang), sin2=np.sin(2 * ang), cos2=np.cos(2 * ang), monsoon=mon)
    X = pd.DataFrame([row])[meta["base_feats"]]
    P = {n: float(m.predict(X)[0]) for n, m in models.items()}
    P["Stacking (Ridge meta)"] = float(stack_meta.predict(np.array([[P[n] for n in meta["stack_bases"]]]))[0])
    P["Hybrid: RF + ANN"] = P["Random Forest"] + float(ann_res.predict(np.column_stack([X.values, [[P["Random Forest"]]]]))[0])
    P = {k: max(v, 0) for k, v in P.items() if k != "Decision Tree"}
    df = pd.DataFrame({"Model": list(P), "Forecast (mm)": list(P.values())})
    df["Family"] = df.Model.map(meta["family"]).replace({"Bagging family": "Bagging"})
    fig = px.bar(df.sort_values("Forecast (mm)"), x="Forecast (mm)", y="Model", color="Family", color_discrete_map=FAMCOL, orientation="h", height=400)
    fig.add_vline(x=clim_t, line_dash="dash", annotation_text=f"long-term avg {clim_t:.0f} mm")
    st.plotly_chart(fig, width='stretch')
    st.caption("SARIMAX-based hybrids need the full time series, so they appear in the Backtest and Next-dekad pages instead.")

# ------------------------------------------------------------------ Insights
else:
    st.title("Insights: what drives the forecasts, and when do ensembles help?")
    a, b = st.columns(2)
    imp = D["imp"].sort_values("mean").tail(12)
    fig = px.bar(imp, x="mean", y="feature", orientation="h", height=420, labels={"mean": "Mean normalised importance (4 tree models)", "feature": ""})
    fig.update_traces(marker_color="#2a9d8f")
    a.plotly_chart(fig, width='stretch')
    b.image(f"{ART}/shap_summary.png", caption="SHAP summary for LightGBM (hold-out)")
    st.subheader("Accuracy by season")
    p = pred.copy(); p["Season"] = p.date.dt.month.map(lambda m: SEASONS[1 if m in (5, 6, 7, 8, 9) else 2 if m in (12, 1, 2) else 0])
    ms = ["Climatology (long-term avg)", best.Model, "Random Forest", "Stacking (Ridge meta)", "Hybrid: SARIMAX + RF"]
    tab = pd.DataFrame({n: p.groupby("Season").apply(lambda g: np.sqrt(((g.Actual - g[n]) ** 2).mean())) for n in dict.fromkeys(ms)})
    st.dataframe(tab.round(1))
    st.markdown("""
**Take-aways**
* **The long-term average is a strong opponent.** Predicting a 10-day rainfall total is intrinsically hard; the best ensembles beat climatology by only a few percent overall.
* **Ensembles add value where rain is irregular:** in the *inter-monsoon* periods (Mar-Apr, Oct-Nov: convective storms, cyclone-related rain) the tree ensembles cut error by roughly 15% versus the average. In the monsoon seasons the calendar already explains most of the rain, and the ML models do not improve on it.
* **The most important feature is the long-term average for that dekad** (`clim_t`), followed by last dekad's rainfall and the seasonal cycle. Models are largely learning *the calendar plus a little persistence*.
* **Bagging and boosting finished close together**; differences are within bootstrap noise. **Stacking did not beat its best member**, and the **hybrids did not beat the simple ensembles** on this small dataset. Extra model layers add variance when only ~3,000 training rows exist.
""")
