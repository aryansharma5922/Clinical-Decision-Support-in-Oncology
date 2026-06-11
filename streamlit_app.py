from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import src.predict as predict_api
from src.evaluate import load_metrics
from src.explainability import (
    approximate_original_feature_importance,
    create_shap_summary_plot,
    create_shap_waterfall_plot,
    pca_component_importance,
)
from src.predict import append_prediction_history, load_prediction_history, predict_tumor
from src.preprocess import FEATURE_NAMES, MODEL_DIR, clean_dataset, download_wdbc_dataset, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"

st.set_page_config(page_title="Oncology Decision Support", page_icon="+", layout="wide")

THEMES = {
    False: {
        "bg": "#f8fafc",
        "sidebar": "#eaf2f8",
        "panel": "#ffffff",
        "panel_alt": "#f1f5f9",
        "text": "#0f172a",
        "muted": "#475569",
        "border": "#dbe3ea",
        "input": "#ffffff",
        "primary": "#1d4ed8",
        "primary_text": "#ffffff",
        "focus": "#2563eb",
        "grid": "#e2e8f0",
        "template": "plotly_white",
        "graph_fill": "#eef6ff",
        "graph_border": "#7aa7c7",
    },
    True: {
        "bg": "#0f172a",
        "sidebar": "#020617",
        "panel": "#111827",
        "panel_alt": "#1e293b",
        "text": "#e5e7eb",
        "muted": "#cbd5e1",
        "border": "#334155",
        "input": "#0b1220",
        "primary": "#60a5fa",
        "primary_text": "#020617",
        "focus": "#93c5fd",
        "grid": "#334155",
        "template": "plotly_dark",
        "graph_fill": "#1e293b",
        "graph_border": "#60a5fa",
    },
}


def ensure_artifacts() -> None:
    required = [MODEL_DIR / "scaler.pkl", MODEL_DIR / "pca.pkl", MODEL_DIR / "voting_model.pkl"]
    if not all(path.exists() for path in required):
        with st.spinner("Preparing clinical decision support models..."):
            from src.train import train_models

            train_models()


@st.cache_data(show_spinner=False)
def cached_dataset():
    return clean_dataset(load_dataset() if (PROJECT_ROOT / "data" / "breast_cancer.csv").exists() else download_wdbc_dataset())


@st.cache_data(show_spinner=False)
def cached_metrics():
    return load_metrics()


def load_artifacts() -> dict:
    path = MODEL_DIR / "training_artifacts.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def predict_batch(df: pd.DataFrame, model_name: str | None = None) -> pd.DataFrame:
    if hasattr(predict_api, "predict_batch"):
        return predict_api.predict_batch(df, model_name)
    missing = [feature for feature in FEATURE_NAMES if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    X = df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
    invalid = X.columns[X.isna().any()].tolist()
    if invalid:
        raise ValueError(f"These feature columns contain non-numeric or blank values: {', '.join(invalid)}")
    scaler, pca, model, selected_model = predict_api.load_prediction_stack(model_name)
    probabilities = model.predict_proba(pca.transform(scaler.transform(X)))[:, 1]
    predictions = probabilities >= 0.5
    results = df.copy()
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results["model"] = selected_model
    results["prediction"] = ["Malignant" if value else "Benign" for value in predictions]
    results["malignant_probability"] = probabilities
    results["confidence"] = [prob if pred else 1 - prob for prob, pred in zip(probabilities, predictions)]
    results["risk_level"] = ["High" if prob >= 0.70 else "Medium" if prob >= 0.40 else "Low" for prob in probabilities]
    result_columns = ["timestamp", "model", "prediction", "malignant_probability", "confidence", "risk_level"]
    return results[result_columns + [column for column in results.columns if column not in result_columns]]


def append_batch_prediction_history(results: pd.DataFrame) -> None:
    if hasattr(predict_api, "append_batch_prediction_history"):
        predict_api.append_batch_prediction_history(results)
        return
    history_columns = ["timestamp", "model", "prediction", "confidence", "malignant_probability", "risk_level", *FEATURE_NAMES]
    rows = results[history_columns].copy()
    history_path = REPORT_DIR / "prediction_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        rows = pd.concat([pd.read_csv(history_path), rows], ignore_index=True)
    rows.to_csv(history_path, index=False)


def inject_css(dark_mode: bool) -> None:
    theme = THEMES[dark_mode]
    px.defaults.template = theme["template"]
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {theme["bg"]};
            --panel-bg: {theme["panel"]};
            --panel-alt: {theme["panel_alt"]};
            --input-bg: {theme["input"]};
            --text-main: {theme["text"]};
            --text-muted: {theme["muted"]};
            --border-soft: {theme["border"]};
            --accent: {theme["primary"]};
            --accent-text: {theme["primary_text"]};
            --focus-ring: {theme["focus"]};
        }}
        html,
        body {{
            background: var(--app-bg) !important;
            color: var(--text-main) !important;
        }}
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {{
            background: var(--app-bg) !important;
            color: var(--text-main) !important;
        }}
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {{
            background: {theme["sidebar"]} !important;
            color: var(--text-main) !important;
        }}
        [data-testid="stSidebar"] * {{ color: var(--text-main) !important; }}
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4,
        [data-testid="stMain"] h5,
        [data-testid="stMain"] h6,
        [data-testid="stMain"] p,
        [data-testid="stMain"] label,
        [data-testid="stMain"] span,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"] {{
            color: var(--text-main) !important;
        }}
        [data-testid="stCaptionContainer"],
        [data-testid="stMetricDelta"],
        small {{
            color: var(--text-muted) !important;
        }}
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        header {{
            background: transparent !important;
        }}
        button,
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-primary"],
        [data-testid="stDownloadButton"] button {{
            color: var(--text-main) !important;
            border-color: var(--border-soft) !important;
            background: var(--panel-bg) !important;
        }}
        [data-testid="baseButton-primary"] {{
            background: var(--accent) !important;
            color: var(--accent-text) !important;
            border-color: var(--accent) !important;
        }}
        [data-testid="baseButton-primary"] * {{
            color: var(--accent-text) !important;
        }}
        button:hover,
        [data-testid="baseButton-secondary"]:hover,
        [data-testid="stDownloadButton"] button:hover {{
            border-color: var(--focus-ring) !important;
            color: var(--text-main) !important;
        }}
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        div[data-baseweb="textarea"],
        div[data-baseweb="textarea"] textarea,
        textarea {{
            background: var(--input-bg) !important;
            color: var(--text-main) !important;
            border-color: var(--border-soft) !important;
        }}
        input,
        textarea,
        [contenteditable="true"] {{
            caret-color: var(--text-main) !important;
        }}
        input::placeholder,
        textarea::placeholder {{
            color: var(--text-muted) !important;
            opacity: 1 !important;
        }}
        div[data-baseweb="input"] input,
        div[data-baseweb="base-input"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] svg,
        div[data-baseweb="popover"],
        div[role="listbox"],
        div[role="option"] {{
            background: var(--input-bg) !important;
            color: var(--text-main) !important;
            fill: var(--text-main) !important;
        }}
        div[role="option"]:hover,
        div[aria-selected="true"] {{
            background: var(--panel-alt) !important;
            color: var(--text-main) !important;
        }}
        div[data-baseweb="slider"] * {{
            color: var(--text-main) !important;
        }}
        div[data-baseweb="slider"] div[role="slider"] {{
            background: var(--accent) !important;
            border-color: var(--accent) !important;
        }}
        [data-testid="stExpander"],
        [data-testid="stExpander"] details,
        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        [data-testid="stTabs"] button {{
            background: var(--panel-bg) !important;
            border-color: var(--border-soft) !important;
            color: var(--text-main) !important;
        }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {{
            color: var(--text-main) !important;
        }}
        [data-testid="stDataFrame"] *,
        [data-testid="stTable"] * {{
            color: var(--text-main) !important;
        }}
        [data-testid="stDataFrame"] canvas {{
            background: var(--panel-bg) !important;
        }}
        [data-testid="stTabs"] button[aria-selected="true"] {{
            border-bottom-color: var(--accent) !important;
            color: var(--accent) !important;
        }}
        [data-testid="stAlert"] {{
            color: #0f172a !important;
        }}
        [data-testid="stAlert"] * {{
            color: inherit !important;
        }}
        hr {{
            border-color: var(--border-soft) !important;
        }}
        .metric-card {{ padding: 1rem; border: 1px solid var(--border-soft); border-radius: 8px; background: var(--panel-bg) !important; color: var(--text-main) !important; }}
        .risk-low {{ background:#dcfce7; color:#14532d; padding:1rem; border-radius:8px; border:1px solid #86efac; }}
        .risk-medium {{ background:#fef9c3; color:#713f12; padding:1rem; border-radius:8px; border:1px solid #fde047; }}
        .risk-high {{ background:#fee2e2; color:#7f1d1d; padding:1rem; border-radius:8px; border:1px solid #fca5a5; }}
        [data-testid="stMain"] .risk-low,
        [data-testid="stMain"] .risk-low * {{ color: #14532d !important; }}
        [data-testid="stMain"] .risk-medium,
        [data-testid="stMain"] .risk-medium * {{ color: #713f12 !important; }}
        [data-testid="stMain"] .risk-high,
        [data-testid="stMain"] .risk-high * {{ color: #7f1d1d !important; }}
        .js-plotly-plot .xtick text,
        .js-plotly-plot .ytick text,
        .js-plotly-plot .xaxislayer-above text,
        .js-plotly-plot .yaxislayer-above text,
        .js-plotly-plot .gtitle,
        .js-plotly-plot .xtitle,
        .js-plotly-plot .ytitle,
        .js-plotly-plot .legendtext,
        .js-plotly-plot .colorbar text {{
            fill: var(--text-main) !important;
            color: var(--text-main) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def theme_plot(fig):
    theme = THEMES[st.session_state.get("dark_mode", False)]
    fig.update_layout(
        template=theme["template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=theme["panel"],
        font_color=theme["text"],
        legend=dict(font=dict(color=theme["text"])),
        coloraxis_colorbar=dict(tickfont=dict(color=theme["text"]), title=dict(font=dict(color=theme["text"]))),
    )
    fig.update_xaxes(gridcolor=theme["grid"], zerolinecolor=theme["grid"], color=theme["text"], title_font_color=theme["text"])
    fig.update_yaxes(gridcolor=theme["grid"], zerolinecolor=theme["grid"], color=theme["text"], title_font_color=theme["text"])
    return fig


def themed_plotly_chart(fig, **kwargs) -> None:
    st.plotly_chart(theme_plot(fig), **kwargs)


def workflow_diagram() -> None:
    theme = THEMES[st.session_state.get("dark_mode", False)]
    st.graphviz_chart(
        f"""
        digraph {{
          rankdir=LR;
          graph [bgcolor="transparent"];
          node [shape=box, style="rounded,filled", fontcolor="{theme["text"]}", fillcolor="{theme["graph_fill"]}", color="{theme["graph_border"]}"];
          edge [color="{theme["graph_border"]}", fontcolor="{theme["text"]}"];
          A [label="WDBC Dataset"]; B [label="Cleaning and Encoding"]; C [label="StandardScaler"];
          D [label="PCA 95% Variance"]; E [label="Random Forest"]; F [label="SVM"];
          G [label="Soft Voting Ensemble"]; H [label="Prediction + SHAP"];
          A -> B -> C -> D; D -> E -> G; D -> F -> G -> H;
        }}
        """
    )


def home_page(df) -> None:
    st.title("Clinical Decision Support in Oncology")
    st.subheader("Early Detection of Malignant Tumours from Diagnostic Imaging Features")
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", f"{df.shape[0]}")
    c2.metric("Diagnostic Features", "30")
    c3.metric("Target Classes", "Malignant / Benign")
    st.write(
        "This B.Tech final year project predicts tumor malignancy using the Breast Cancer Wisconsin Diagnostic dataset, PCA dimensionality reduction, Random Forest, SVM, and a soft voting ensemble."
    )
    workflow_diagram()


def dataset_page(df) -> None:
    st.header("Dataset Overview")
    st.dataframe(df.head(25), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Dataset Statistics")
        st.dataframe(df[FEATURE_NAMES].describe().T, use_container_width=True)
    with c2:
        st.subheader("Missing Value Analysis")
        missing = df.isna().sum().reset_index()
        missing.columns = ["column", "missing_values"]
        st.dataframe(missing, use_container_width=True)
        dist = df["diagnosis"].map({"M": "Malignant", "B": "Benign"}).value_counts().reset_index()
        dist.columns = ["Diagnosis", "Count"]
        themed_plotly_chart(px.pie(dist, names="Diagnosis", values="Count", hole=0.45), use_container_width=True)


def eda_page(df) -> None:
    st.header("EDA Dashboard")
    labeled = df.assign(Diagnosis=df["diagnosis"].map({"M": "Malignant", "B": "Benign"}))
    tabs = st.tabs(["Distribution", "Correlation", "Histograms", "Box Plots", "Pair Plot", "Feature Importance"])
    with tabs[0]:
        counts = labeled["Diagnosis"].value_counts().reset_index()
        counts.columns = ["Diagnosis", "Count"]
        themed_plotly_chart(px.bar(counts, x="Diagnosis", y="Count", color="Diagnosis", text="Count"), use_container_width=True)
    with tabs[1]:
        themed_plotly_chart(px.imshow(df[FEATURE_NAMES].corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=760), use_container_width=True)
    with tabs[2]:
        feature = st.selectbox("Feature", FEATURE_NAMES, key="hist_feature")
        themed_plotly_chart(px.histogram(labeled, x=feature, color="Diagnosis", marginal="box", nbins=40), use_container_width=True)
    with tabs[3]:
        selected = st.multiselect("Features", FEATURE_NAMES, default=FEATURE_NAMES[:5], key="box_features")
        if selected:
            melted = labeled.melt(id_vars="Diagnosis", value_vars=selected, var_name="Feature", value_name="Value")
            themed_plotly_chart(px.box(melted, x="Feature", y="Value", color="Diagnosis"), use_container_width=True)
    with tabs[4]:
        pair_features = st.multiselect("Pair plot features", FEATURE_NAMES, default=["radius_mean", "texture_mean", "perimeter_mean", "area_mean"])
        if len(pair_features) >= 2:
            themed_plotly_chart(px.scatter_matrix(labeled, dimensions=pair_features, color="Diagnosis", height=780), use_container_width=True)
    with tabs[5]:
        importance = approximate_original_feature_importance().head(15)
        themed_plotly_chart(px.bar(importance, x="importance", y="feature", orientation="h", title="PCA-weighted Random Forest Importance"), use_container_width=True)


def pca_page() -> None:
    st.header("PCA Analysis")
    artifacts = load_artifacts()
    evr = artifacts.get("explained_variance_ratio", [])
    cumulative = artifacts.get("cumulative_explained_variance", [])
    if not evr:
        st.warning("PCA artifacts are not available yet.")
        return
    c1, c2 = st.columns([1, 2])
    c1.metric("Selected Components", artifacts.get("pca_components", len(evr)))
    c1.metric("Variance Retained", f"{cumulative[-1] * 100:.2f}%")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, len(evr) + 1)), y=evr, name="Individual"))
    fig.add_trace(go.Scatter(x=list(range(1, len(cumulative) + 1)), y=cumulative, mode="lines+markers", name="Cumulative"))
    fig.update_layout(xaxis_title="Principal Component", yaxis_title="Explained Variance Ratio")
    c2.plotly_chart(theme_plot(fig), use_container_width=True)
    themed_plotly_chart(px.bar(pca_component_importance(), x="component", y="importance", title="PCA Component Importance"), use_container_width=True)


def model_page() -> None:
    st.header("Model Performance")
    metrics = cached_metrics()
    if metrics.empty:
        st.warning("Model metrics are not available yet.")
        return
    st.dataframe(metrics.style.format({c: "{:.4f}" for c in metrics.columns if c != "model"}), use_container_width=True)
    chart_df = metrics.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"], var_name="Metric", value_name="Score")
    themed_plotly_chart(px.bar(chart_df, x="model", y="Score", color="Metric", barmode="group"), use_container_width=True)
    c1, c2 = st.columns(2)
    if (REPORT_DIR / "roc_curves.png").exists():
        c1.image(str(REPORT_DIR / "roc_curves.png"), caption="ROC Curves")
    if (REPORT_DIR / "precision_recall_curves.png").exists():
        c2.image(str(REPORT_DIR / "precision_recall_curves.png"), caption="Precision-Recall Curves")


def prediction_form() -> dict:
    artifacts = load_artifacts()
    medians = artifacts.get("feature_medians", {})
    mins = artifacts.get("feature_min", {})
    maxs = artifacts.get("feature_max", {})
    values = {}
    groups = {"Mean Features": FEATURE_NAMES[:10], "Standard Error Features": FEATURE_NAMES[10:20], "Worst Features": FEATURE_NAMES[20:]}
    for group, features in groups.items():
        with st.expander(group, expanded=(group == "Mean Features")):
            cols = st.columns(2)
            for idx, feature in enumerate(features):
                low = float(mins.get(feature, 0.0))
                high = float(maxs.get(feature, max(low + 1.0, 1.0)))
                default = float(medians.get(feature, (low + high) / 2))
                values[feature] = cols[idx % 2].number_input(
                    feature, min_value=max(0.0, low * 0.5), max_value=high * 1.5,
                    value=default, step=max((high - low) / 100, 0.001), format="%.5f",
                )
    return values


def prediction_page() -> None:
    st.header("Tumor Prediction")
    selected_model = st.selectbox("Model", ["Voting Ensemble", "Random Forest", "SVM"])
    values = prediction_form()
    if st.button("Predict", type="primary"):
        try:
            result = predict_tumor(values, selected_model)
            append_prediction_history(result)
            st.session_state["last_prediction"] = result
            css = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}[result["risk_level"]]
            st.markdown(
                f"""<div class="{css}"><h3>Prediction: {result['prediction']}</h3>
                <p><b>Confidence Score:</b> {result['confidence'] * 100:.2f}%</p>
                <p><b>Risk Category:</b> {result['risk_level']}</p>
                <p><b>Malignant Probability:</b> {result['malignant_probability'] * 100:.2f}%</p></div>""",
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")


def batch_prediction_page() -> None:
    st.header("Batch Prediction")
    selected_model = st.selectbox("Model", ["Voting Ensemble", "Random Forest", "SVM"], key="batch_model")
    artifacts = load_artifacts()
    medians = artifacts.get("feature_medians", {})
    template = pd.DataFrame([{feature: float(medians.get(feature, 0.0)) for feature in FEATURE_NAMES}])
    st.download_button("Download CSV Template", template.to_csv(index=False), "batch_prediction_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV with diagnostic feature columns", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV that contains the 30 diagnostic feature columns used by the model.")
        return
    if st.session_state.get("last_batch_source") != uploaded.name:
        st.session_state.pop("last_batch_prediction", None)
        st.session_state["last_batch_source"] = uploaded.name

    try:
        batch_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"CSV could not be read: {exc}")
        return
    if batch_df.empty:
        st.error("The uploaded CSV does not contain any records.")
        return

    st.subheader("Uploaded Records")
    st.dataframe(batch_df.head(50), use_container_width=True)
    missing = [feature for feature in FEATURE_NAMES if feature not in batch_df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    save_history = st.checkbox("Save batch results to prediction history", value=True)
    if st.button("Run Batch Prediction", type="primary"):
        try:
            results = predict_batch(batch_df, selected_model)
            if save_history:
                append_batch_prediction_history(results)
            st.session_state["last_batch_prediction"] = results
            st.success(f"Completed predictions for {len(results)} records.")
        except Exception as exc:
            st.error(f"Batch prediction could not be completed: {exc}")

    results = st.session_state.get("last_batch_prediction")
    if results is not None and not results.empty:
        st.subheader("Batch Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", f"{len(results)}")
        c2.metric("Malignant", f"{int(results['prediction'].eq('Malignant').sum())}")
        c3.metric("High Risk", f"{int(results['risk_level'].eq('High').sum())}")
        st.dataframe(results, use_container_width=True)
        summary = results["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).reset_index()
        summary.columns = ["Risk Level", "Count"]
        themed_plotly_chart(px.bar(summary, x="Risk Level", y="Count", color="Risk Level", text="Count"), use_container_width=True)
        st.download_button(
            "Download Batch Results CSV",
            results.to_csv(index=False),
            f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
        )


def explainable_ai_page() -> None:
    st.header("Explainable AI")
    themed_plotly_chart(px.bar(approximate_original_feature_importance().head(20), x="importance", y="feature", orientation="h", title="Global Feature Contribution"), use_container_width=True)
    if st.button("Generate SHAP Summary Plot"):
        path = create_shap_summary_plot()
        st.image(str(path), caption="SHAP Summary Plot") if path and path.exists() else st.warning("Install SHAP to generate this plot.")
    last = st.session_state.get("last_prediction")
    if last and st.button("Generate Waterfall Plot for Latest Prediction"):
        path = create_shap_waterfall_plot(last["inputs"])
        st.image(str(path), caption="SHAP Waterfall Plot") if path and path.exists() else st.warning("Install SHAP to generate this plot.")
    elif not last:
        st.info("Create a prediction first to view an individual SHAP explanation.")


def history_page() -> None:
    st.header("Prediction History")
    history = load_prediction_history()
    if history.empty:
        st.info("No predictions have been recorded yet.")
        return
    st.dataframe(history.sort_values("timestamp", ascending=False), use_container_width=True)
    st.download_button("Download History CSV", history.to_csv(index=False), "prediction_history.csv", "text/csv")


def build_pdf_report(result: dict) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Oncology Decision Support Prediction Report", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    for label in ["model", "prediction", "risk_level"]:
        pdf.cell(0, 8, f"{label.replace('_', ' ').title()}: {result[label]}", ln=True)
    pdf.cell(0, 8, f"Confidence Score: {result['confidence'] * 100:.2f}%", ln=True)
    pdf.cell(0, 8, f"Malignant Probability: {result['malignant_probability'] * 100:.2f}%", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Patient Imaging Feature Inputs", ln=True)
    pdf.set_font("Arial", size=9)
    for feature, value in result["inputs"].items():
        pdf.cell(0, 6, f"{feature}: {value:.5f}", ln=True)
    output = pdf.output(dest="S")
    return output.encode("latin1") if isinstance(output, str) else bytes(output)


def download_reports_page() -> None:
    st.header("Download Reports")
    last = st.session_state.get("last_prediction")
    if last:
        st.download_button("Download Latest Prediction PDF", build_pdf_report(last), "oncology_prediction_report.pdf", "application/pdf")
    else:
        st.info("Create a prediction to generate a patient-level PDF report.")
    metrics = cached_metrics()
    if not metrics.empty:
        st.download_button("Download Model Metrics CSV", metrics.to_csv(index=False), "model_metrics.csv", "text/csv")
    history = load_prediction_history()
    if not history.empty:
        st.download_button("Download Prediction History CSV", history.to_csv(index=False), "prediction_history.csv", "text/csv")
    batch = st.session_state.get("last_batch_prediction")
    if batch is not None and not batch.empty:
        st.download_button("Download Latest Batch Results CSV", batch.to_csv(index=False), "latest_batch_predictions.csv", "text/csv")


def main() -> None:
    ensure_artifacts()
    df = cached_dataset()
    st.sidebar.title("Oncology CDSS")
    dark_mode = st.sidebar.toggle("Dark Mode", value=False)
    st.session_state["dark_mode"] = dark_mode
    inject_css(dark_mode)
    page = st.sidebar.radio(
        "Menu",
        ["Home", "Dataset Overview", "EDA Dashboard", "PCA Analysis", "Model Performance", "Tumor Prediction", "Batch Prediction", "Explainable AI", "Prediction History", "Download Reports"],
    )
    {
        "Home": home_page,
        "Dataset Overview": dataset_page,
        "EDA Dashboard": eda_page,
        "PCA Analysis": lambda _: pca_page(),
        "Model Performance": lambda _: model_page(),
        "Tumor Prediction": lambda _: prediction_page(),
        "Batch Prediction": lambda _: batch_prediction_page(),
        "Explainable AI": lambda _: explainable_ai_page(),
        "Prediction History": lambda _: history_page(),
        "Download Reports": lambda _: download_reports_page(),
    }[page](df)


if __name__ == "__main__":
    main()
