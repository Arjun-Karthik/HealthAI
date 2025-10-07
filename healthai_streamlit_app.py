# healthai_streamlit_app.py
# Streamlit app for HealthAI Suite — USES SAVED MODELS and shows LABEL NAMES
import os, io, json, pickle, glob, re, types, zipfile, sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple, Any

# Optional libs (logged gracefully)
try:
    import shap
except Exception:
    shap = None
try:
    import mlflow
except Exception:
    mlflow = None

# ML/AI toolkits
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, silhouette_score,
    calinski_harabasz_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, precision_recall_curve
)
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import label_binarize

# For association rules
try:
    from mlxtend.frequent_patterns import apriori, association_rules
except Exception:
    apriori = None
    association_rules = None

APP_TITLE = "HealthAI Suite Dashboard"
MODELS_DIR = Path("Models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🩺")
st.title("🩺 HealthAI Suite Dashboard")

# Practically "no limit" uploads (bounded by host)
try:
    st.set_option("server.maxUploadSize", 10_000_000)  # ~10 TB in MB
except Exception:
    pass

# ---------- Utilities: models ----------
def extract_models_zip():
    """Auto-extract Models.zip to ./Models if present."""
    z = Path("Models.zip")
    if z.exists():
        try:
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(MODELS_DIR)
            st.success("Extracted Models.zip into ./Models")
        except Exception as e:
            st.warning(f"Could not extract Models.zip: {e}")

extract_models_zip()

def list_model_files():
    pats = [
        str(MODELS_DIR / "**" / "*.pkl"),
        str(MODELS_DIR / "**" / "*.joblib"),
    ]
    files = []
    for pat in pats:
        files.extend(glob.glob(pat, recursive=True))
    return sorted(set(files))

# ---------- Robust unpickling (handles custom ToStr etc.) ----------
class ToStr(BaseEstimator, TransformerMixin):
    """Picklable categorical-to-string transformer used in training pipelines."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.fillna("NA").astype(str).to_numpy()
        return pd.DataFrame(X).fillna("NA").astype(str).to_numpy()

def _ensure_module(mod_name: str) -> types.ModuleType:
    m = sys.modules.get(mod_name)
    if m is None:
        m = types.ModuleType(mod_name)
        sys.modules[mod_name] = m
    return m

# Register ToStr where old pickles might expect it
for alias in ("main", "__main__"):
    _ensure_module(alias)
    setattr(sys.modules[alias], "ToStr", ToStr)

def _joblib_or_pickle_load(path: str):
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def resilient_load_model(path: str, manual_stub_names=None):
    """
    Load pickle/joblib model. If a custom class/function is missing (e.g., 'ToStr'),
    inject a stub and retry.
    """
    try:
        return _joblib_or_pickle_load(path)
    except Exception as e1:
        msg = str(e1)
        m_attr = re.search(r"Can't get attribute '([^']+)'", msg)
        m_mod  = re.search(r"on <module '([^']+)'", msg)
        missing = m_attr.group(1) if m_attr else None
        modname = m_mod.group(1) if m_mod else None

        names_to_stub = []
        if missing: names_to_stub.append(missing)
        if manual_stub_names:
            names_to_stub += [s for s in manual_stub_names if s]

        class _IdTrans(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None): return self
            def transform(self, X): return X

        for nm in names_to_stub:
            for target_mod in [modname or "main", "main", "__main__", __name__]:
                m = _ensure_module(target_mod)
                if nm and nm.lower() in {"tostr", "to_string", "stringify"}:
                    setattr(m, nm, ToStr)
                else:
                    setattr(m, nm, _IdTrans)

        return _joblib_or_pickle_load(path)

# ---------- Schema inference from saved pipeline ----------
def infer_schema_from_pipeline(model) -> dict:
    """
    Returns {'number': [...], 'category': [...], 'other': [...]}
    by looking for a ColumnTransformer (commonly in step named 'pre').
    """
    number, category, other = [], [], []
    pre = None

    if isinstance(model, Pipeline):
        for name, step in model.steps:
            if isinstance(step, ColumnTransformer) or name.lower() in ("pre", "preprocess", "preprocessor"):
                if isinstance(step, ColumnTransformer):
                    pre = step
                elif isinstance(step, Pipeline):
                    for n2, s2 in step.steps:
                        if isinstance(s2, ColumnTransformer):
                            pre = s2
                            break
                break

    if isinstance(pre, ColumnTransformer):
        for _, trans, cols in pre.transformers_:
            cols = list(cols) if isinstance(cols, (list, tuple, pd.Index)) else []
            if isinstance(trans, Pipeline):
                inner = [s for _, s in trans.steps]
                if any(isinstance(s, OneHotEncoder) for s in inner):
                    category.extend(cols)
                elif any(isinstance(s, StandardScaler) for s in inner):
                    number.extend(cols)
                else:
                    other.extend(cols)
            elif isinstance(trans, OneHotEncoder):
                category.extend(cols)
            elif isinstance(trans, StandardScaler):
                number.extend(cols)
            else:
                other.extend(cols)
    else:
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            number = list(names)

    def dedup(seq):
        seen = set(); out=[]
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    return {"number": dedup(number), "category": dedup(category), "other": dedup(other)}

# ---------- Label map helpers ----------
def parse_label_map(text: str):
    """
    Parse "0=Normal,1=Pneumonia" or "0: Normal\n1: Pneumonia" into a dict
    mapping keys as both int and str -> label name.
    """
    m = {}
    if not text or not text.strip():
        return m
    parts = re.split(r"[,;\n]+", text.strip())
    for p in parts:
        if not p.strip():
            continue
        if ":" in p:
            k, v = p.split(":", 1)
        elif "=" in p:
            k, v = p.split("=", 1)
        else:
            continue
        k = k.strip(); v = v.strip()
        m[k] = v
        try:
            m[int(k)] = v
        except Exception:
            pass
    return m

def load_label_map_ui(classes, key_prefix="clf"):
    st.markdown("##### Label names")

    lm_file = st.file_uploader(
        "Upload label map (JSON {'0':'Normal',...} or CSV with columns 'code','label')",
        type=["json","csv"], key=f"{key_prefix}_labelmap_file"
    )
    label_map = {}

    if lm_file is not None:
        try:
            if lm_file.name.lower().endswith(".json"):
                raw = json.load(io.StringIO(lm_file.getvalue().decode("utf-8")))
                for k, v in raw.items():
                    label_map[k] = v
                    try: label_map[int(k)] = v
                    except: pass
            else:  # CSV
                dfm = pd.read_csv(lm_file)
                if {"code","label"}.issubset(dfm.columns):
                    for _, row in dfm.iterrows():
                        k = row["code"]; v = row["label"]
                        label_map[str(k)] = str(v)
                        try: label_map[int(k)] = str(v)
                        except: pass
        except Exception as e:
            st.warning(f"Could not read label map: {e}")

    # Auto if still empty
    if not label_map and classes is not None and len(classes) > 0:
        if not np.issubdtype(np.array(classes).dtype, np.number):
            for c in classes:
                label_map[c] = str(c)
        else:
            for c in classes:
                label_map[c] = f"class_{c}"
                label_map[str(c)] = f"class_{c}"

    return label_map

def name_of(x, label_map):
    """Return friendly label string for a predicted/true value using label_map."""
    return label_map.get(x, label_map.get(str(x), str(x)))

# ---------- Importance helpers (keep only important columns) ----------
def get_preprocessor_and_estimator(model):
    pre, est, pipe = None, model, None
    if isinstance(model, Pipeline):
        pipe = model
        est = pipe.steps[-1][1]
        for name, step in pipe.steps:
            if isinstance(step, ColumnTransformer) or name.lower() in ("pre", "preprocess", "preprocessor"):
                if isinstance(step, ColumnTransformer):
                    pre = step
                elif isinstance(step, Pipeline):
                    for _, s2 in step.steps:
                        if isinstance(s2, ColumnTransformer):
                            pre = s2
                            break
    return pre, est, pipe or model

def _safe_get_feature_names(pre: ColumnTransformer, X_trans):
    names = None
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = None
    if X_trans is not None and hasattr(X_trans, "shape"):
        k = X_trans.shape[1]
        if names is None or len(names) != k:
            names = [f"f{i}" for i in range(k)]
    return names

def collapse_onehot_names(feature_names, schema):
    cat_cols = set(schema.get("category", []))
    collapsed = []
    for nm in feature_names:
        base = re.split(r"\.__|__", nm)[-1]
        best = base
        for c in cat_cols:
            if base.startswith(c + "_") or base == c:
                best = c
                break
        collapsed.append(best)
    return collapsed

def try_native_importance(estimator):
    if hasattr(estimator, "feature_importances_"):
        imp = np.asarray(estimator.feature_importances_, dtype=float)
        s = imp.sum()
        return imp / s if s > 0 else imp
    if hasattr(estimator, "coef_"):
        co = np.asarray(estimator.coef_, dtype=float)
        if co.ndim == 2:
            co = np.mean(np.abs(co), axis=0)
        else:
            co = np.abs(co)
        s = co.sum()
        return co / s if s > 0 else co
    return None

def important_original_features(model, X_df, top_k=15):
    schema = infer_schema_from_pipeline(model)
    pre, est, pipe = get_preprocessor_and_estimator(model)

    X_trans = None
    feat_names = None
    if pre is not None:
        try:
            X_trans = pre.transform(X_df)
        except Exception:
            X_trans = None
        feat_names = _safe_get_feature_names(pre, X_trans)
    else:
        feat_names = list(X_df.columns)

    native = try_native_importance(est)
    if native is not None:
        expected_len = X_trans.shape[1] if X_trans is not None else len(feat_names)
        if len(native) == expected_len:
            collapsed = collapse_onehot_names(feat_names, schema)
            df_imp = pd.DataFrame({"orig": collapsed, "imp": native})
            agg = (df_imp.groupby("orig", as_index=False)["imp"].sum()
                         .sort_values("imp", ascending=False))
            cols = [c for c in agg["orig"].tolist() if c in X_df.columns]
            return cols[:top_k]

    # Fallback: numeric variance
    try:
        v = pd.DataFrame(X_df).select_dtypes(include=[np.number]).var()
        v = v.replace([np.inf, -np.inf], 0).fillna(0).sort_values(ascending=False)
        cols = [c for c in v.index.tolist() if c in X_df.columns]
        if cols:
            return cols[:top_k]
    except Exception:
        pass

    return list(X_df.columns)[:top_k]

# ---------- Plots ----------
def plot_conf_mat(y_true_labels, y_pred_labels, title="Confusion Matrix"):
    labels = sorted(list(set(y_true_labels) | set(y_pred_labels)))
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)
    import plotly.express as px
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
                    color_continuous_scale="Blues", title=title)
    fig.update_layout(xaxis_title="Predicted", yaxis_title="True")
    fig.update_xaxes(side="top")
    st.plotly_chart(fig, use_container_width=True)

def plot_roc_pr_binary(y_true, scores):
    try:
        auc = roc_auc_score(y_true, scores)
        st.write(f"ROC-AUC: **{auc:.3f}**")
    except Exception:
        pass
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)

    import matplotlib.pyplot as plt
    fig1, ax1 = plt.subplots(figsize=(4.5,4))
    ax1.plot(fpr, tpr); ax1.plot([0,1],[0,1], "--")
    ax1.set_title("ROC"); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    st.pyplot(fig1)
    fig2, ax2 = plt.subplots(figsize=(4.5,4))
    ax2.plot(recall, precision)
    ax2.set_title("PR"); ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    st.pyplot(fig2)

def plot_roc_micro_multiclass(y_true, proba, classes):
    try:
        y_bin = label_binarize(y_true, classes=classes)
        auc = roc_auc_score(y_bin, proba, average="micro")
        st.write(f"ROC-AUC (micro): **{auc:.3f}**")
    except Exception:
        pass

# ---------------------- Navigation ----------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select module",
    [
        "🔬 1. Exploratory Data Analysis",
        "🩺 2. Disease Prediction",
        "🏥 3. LOS Prediction",
        "🧭 4. Clustering",
        "🧾 5. Association Rules",
        "📈 6. Time Series (ED Vitals)",
        "🩻 7. Imaging (CNN)",
        "💬 8. Sentiment & Chatbot",
        "⚙️ Settings & Logging"
    ]
)

# ---------------------- Helpers ----------------------
def load_csv_file(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return pd.DataFrame()

def find_paths(patterns):
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat, recursive=True))
    return hits

# Imaging model (from your CNN notebook)
class PneumoniaCNN(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.drop2 = nn.Dropout(0.1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.drop4 = nn.Dropout(0.2)
        self.bn4   = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.drop5 = nn.Dropout(0.2)
        self.bn5   = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.flat  = nn.Flatten()
        self.fc1   = nn.Linear(256*4*4, 128)
        self.dropf = nn.Dropout(0.2)
        self.head  = nn.Linear(128, 1)  # single logit
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.bn1(x); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.drop2(x); x = self.bn2(x); x = self.pool2(x)
        x = F.relu(self.conv3(x)); x = self.bn3(x); x = self.pool3(x)
        x = F.relu(self.conv4(x)); x = self.drop4(x); x = self.bn4(x); x = self.pool4(x)
        x = F.relu(self.conv5(x)); x = self.drop5(x); x = self.bn5(x); x = self.pool5(x)
        x = self.flat(x)
        x = F.relu(self.fc1(x)); x = self.dropf(x)
        logits = self.head(x)
        return logits

def preprocess_xray(img: Image.Image, img_size=150):
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return tfm(img).unsqueeze(0)

def try_load_imaging_model(best_path="Models/best_by_val_acc.pt", device="cpu"):
    model = PneumoniaCNN(in_ch=1).to(device)
    if best_path and os.path.exists(best_path):
        try:
            ckpt = torch.load(best_path, map_location=device)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state)
            model.eval()
            return model, ckpt.get("classes", ["NORMAL","PNEUMONIA"]), ckpt.get("img_size", 150)
        except Exception as e:
            st.warning(f"Could not load model from {best_path}: {e}")
    model.eval()
    return model, ["NORMAL","PNEUMONIA"], 150

# ---------------------- Pages ----------------------

if page.startswith("🔬 1"):
    st.subheader("Exploratory Data Analysis")
    up = st.file_uploader("Upload a CSV for quick EDA", type=["csv"])
    if up:
        def read_csv_robust(uploaded_file):
            raw = uploaded_file.getvalue()  # bytes
            # Try common encodings first
            for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
                try:
                    return pd.read_csv(io.BytesIO(raw), encoding=enc, engine="python", on_bad_lines="skip")
                except Exception:
                    pass
            # Fallback: decode to text (ignore bad bytes), normalize NBSP, auto-detect separator
            try:
                text = raw.decode("cp1252", errors="ignore").replace("\xa0", " ")
                return pd.read_csv(io.StringIO(text), engine="python", sep=None, on_bad_lines="skip")
            except Exception as e:
                st.error(f"Failed to parse CSV: {e}")
                return pd.DataFrame()

        df = read_csv_robust(up)
        if df.empty:
            st.stop()

        # Clean column names (trim & remove non-breaking spaces)
        df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]

        # --- Basic Info ---
        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))

        # Show dtypes with column names
        dtypes_df = pd.DataFrame(df.dtypes, columns=["Data type"]).reset_index()
        dtypes_df.columns = ["Column Name", "Data type"]
        st.write("### Column Data Types")
        dtypes_df.index = range(1, len(dtypes_df) + 1)
        st.dataframe(dtypes_df)

        # Missing values
        st.write("### Missing Values")
        missing_df = (
            df.isna().sum()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"index": "Column Name", 0: "Missing Count"})
        )
        missing_df.index = range(1, len(missing_df) + 1)
        st.dataframe(missing_df)

        # Preview
        st.write("### Preview")
        preview_df = df.head().reset_index(drop=True)
        preview_df.index = range(1, len(preview_df) + 1)
        st.dataframe(preview_df)

        # --- Numeric summary ---
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            st.markdown("### Numeric Summary")
            numeric_summary = df[num_cols].describe().T.reset_index()
            numeric_summary = numeric_summary.rename(columns={"index": "Column Name"})
            numeric_summary.index = range(1, len(numeric_summary) + 1)
            st.dataframe(numeric_summary, use_container_width=True)

            # Correlation heatmap
            st.markdown("#### Correlation Heatmap")
            try:
                import seaborn as sns, matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(df[num_cols].corr(numeric_only=True), annot=False, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            except Exception as e:
                st.info(f"(Skipped heatmap — {e})")

            # Outlier check using IQR
            st.markdown("#### Outlier Detection (IQR method)")
            outlier_report = {}
            for col in num_cols:
                try:
                    Q1, Q3 = df[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                    outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
                    outlier_report[col] = int(outlier_count)
                except Exception:
                    outlier_report[col] = 0

            outlier_df = (
                pd.DataFrame.from_dict(outlier_report, orient="index", columns=["Outlier Count"])
                .reset_index()
                .rename(columns={"index": "Column Name"})
            )
            outlier_df.index = range(1, len(outlier_df) + 1)
            st.dataframe(outlier_df)

            # Distribution plots
            st.markdown("#### Distributions of Numeric Columns")
            try:
                import seaborn as sns, matplotlib.pyplot as plt
                sel_num = st.multiselect("Pick numeric columns to plot", num_cols, default=num_cols[:2])
                if sel_num:
                    for c in sel_num:
                        fig, ax = plt.subplots()
                        sns.histplot(df[c].dropna(), kde=True, bins=30, ax=ax)
                        ax.set_title(f"Distribution of {c}")
                        st.pyplot(fig)
            except Exception as e:
                st.info(f"(Skipped distribution plots — {e})")

        # --- Categorical summary ---
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            st.markdown("### Categorical Summary")
            for c in cat_cols:
                st.write(f"**{c}** — {df[c].nunique()} unique values")
                st.write(df[c].value_counts(dropna=False).head(10))

        # --- Duplicate rows ---
        dup_count = df.duplicated().sum()
        st.write("Duplicate Rows:", int(dup_count))

        # --- Target column detection ---
        target_guess = next((c for c in ["label", "target", "y", "outcome", "disease"] if c in df.columns), None)
        if target_guess:
            st.success(f"Possible target column detected: `{target_guess}`")
            st.write(df[target_guess].value_counts(dropna=False))

elif page.startswith("🩺 2"):
    st.subheader("Disease Prediction")
    st.caption("Pick a saved classification pipeline (.pkl/.joblib), optionally map label codes to names, then score a CSV.")
    stub_txt = st.sidebar.text_input("If your pickle needs custom classes, list names (comma-separated)", value="")

    # === Specify exactly which columns you want to display from the uploaded CSV ===
    COLUMNS_TO_SHOW = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        'sex',
        'age_at_admit',
        'language',
        'label_disease'
    ]

    # Discover models
    found_models = list_model_files()
    picked = st.selectbox("Choose a saved classifier from ./Models", ["(Upload Model File)"] + found_models)

    model_path = None
    if picked != "(Upload Model File)":
        model_path = picked

    clf = None
    classes_for_map = None
    if model_path:
        try:
            manual = [s.strip() for s in stub_txt.split(",") if s.strip()]
            clf = resilient_load_model(model_path, manual_stub_names=manual)
            st.success(f"Loaded classifier: {model_path}")
            classes_for_map = getattr(clf, "classes_", None)
        except Exception as e:
            st.error(f"Failed to load classifier: {e}")

    if clf is None:
        st.info("Load a saved classifier to begin.")
    else:
        # Label map UI + resolver
        label_map = load_label_map_ui(classes_for_map, key_prefix="clf")
        def map_array(a):
            return [name_of(v, label_map) for v in a]

        st.markdown("#### Upload batch CSV")
        csv = st.file_uploader("CSV with feature columns", type=["csv"], key="clf_batch_csv")
        out_name = st.text_input("Output filename", value="Disease Predictions.csv")

        if csv is not None:
            df = pd.read_csv(csv, low_memory=False)
            X = df.copy()

            try:
                # Require predict_proba for probability filtering
                if not hasattr(clf, "predict_proba"):
                    st.error("This model does not support probability estimates (`predict_proba`).")
                else:
                    proba = clf.predict_proba(X)

                    # Determine top class and its probability for each row
                    top_idx = np.argmax(proba, axis=1)
                    top_prob = proba[np.arange(len(proba)), top_idx]

                    # Top-class label codes and mapped names
                    raw_classes = getattr(clf, "classes_", np.arange(proba.shape[1]))
                    top_class_codes = np.array(raw_classes)[top_idx]
                    top_class_names = np.array(map_array(top_class_codes))

                    # Compose a base output with prediction + probability
                    base_out = pd.DataFrame({
                        "prediction": top_class_names,
                        "accuracy": np.round(top_prob, 4)
                    }, index=X.index)

                    # Filter to 0.80–0.90 inclusive
                    mask = (top_prob >= 0.80) & (top_prob <= 0.97)

                    if not mask.any():
                        st.info("No rows fell into the probability.")
                    else:
                        # Keep only the specified columns that actually exist
                        selected_input_cols = [c for c in COLUMNS_TO_SHOW if c in X.columns]

                        # Build the final filtered view: specified columns + prediction + pred_prob
                        filtered_small = pd.concat(
                            [
                                X.loc[mask, selected_input_cols].reset_index(drop=True),
                                base_out.loc[mask, ["prediction", "accuracy"]].reset_index(drop=True),
                            ],
                            axis=1
                        )

                        st.success("Predicted Table")
                        filtered_small.index = range(1, len(filtered_small) + 1)
                        st.dataframe(filtered_small)

                        st.download_button(
                            "⬇️ Download Prediction CSV",
                            data=filtered_small.to_csv(index=False).encode("utf-8"),
                            file_name=out_name, mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

elif page.startswith("🏥 3"):
    st.subheader("Length of Stay Prediction")
    st.caption("Loads the production inference pipeline (`los_inference_pipeline.pkl`) and predicts LOS on a CSV. Displays prediction.")

    # === Specify exactly which columns you want to display from the uploaded CSV ===
    COLUMNS_TO_SHOW = [
        "subject_id", 
        "hadm_id", 
        "admittime",
        "sex", 
        "age_at_admit",
        "insurance",
        "admission_type", 
        "admission_location", 
        "is_emergency", 
        "dow", 
        "hour",
        "comor_count"
    ]

    # ---- Helper to read saved feature list ----
    def _load_los_features_json(models_dir: Path) -> list:
        fjson = models_dir / "LOS_Model/los_features.json"
        if fjson.exists():
            try:
                with open(fjson, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                feats = data.get("features", [])
                if isinstance(feats, list) and len(feats) > 0:
                    return feats
            except Exception as e:
                st.warning(f"Could not read los_features.json: {e}")
        return []

    # ---- Model picker: default to the production inference pipeline ----
    default_infer_path = MODELS_DIR / "LOS_Model/los_inference_pipeline.pkl"
    found_models = list_model_files()
    choices = [str(default_infer_path)] if default_infer_path.exists() else []
    # also allow picking any other .pkl/.joblib in ./Models
    choices += [p for p in found_models if p.endswith((".pkl", ".joblib")) and "los_inference_pipeline.pkl" not in p]
    choices = ["(Upload Model File)"] + (choices if choices else ["(no default found)"])

    picked = st.selectbox("Choose a saved regressor", choices, key="reg_pick_fixed")

    # ---- Resolve model path (save uploads to disk so joblib can read) ----
    model_path = None
    if picked not in ("(Upload Model File)", "(no default found)"):
        model_path = picked

    # ---- Load model (pure sklearn Pipeline) ----
    reg = None
    feats = []
    if model_path:
        try:
            import joblib
            reg = joblib.load(model_path)
            st.success(f"Loaded regressor: {model_path}")

            # Try to load saved feature list (used to align incoming CSV)
            feats = _load_los_features_json(MODELS_DIR)
            if not feats:
                # Fallback: feature_names_in_ if present
                feats = getattr(reg, "feature_names_in_", [])
                if feats is None:
                    feats = []
        except Exception as e:
            st.error(f"Failed to load regressor: {e}")

    if reg is None:
        st.info("Load a saved regressor to begin (ideally `Models/los_inference_pipeline.pkl`).")
    else:
        # Inputs mirror the classification block
        st.markdown("#### Upload batch CSV")
        csv = st.file_uploader("CSV with feature columns", type=["csv"], key="reg_batch_csv_fixed")
        out_name = st.text_input("Output filename", value="LOS Predictions.csv", key="reg_outname_fixed")

        if csv is not None:
            # Robust CSV read
            try:
                df = pd.read_csv(csv, low_memory=False)
            except Exception:
                csv.seek(0)
                df = pd.read_csv(io.StringIO(csv.getvalue().decode("utf-8", errors="ignore")))

            # ------ Align columns to the model's expected features ------
            if feats:
                X = df.copy()
                # Add missing expected columns so imputers can handle them
                missing = [c for c in feats if c not in X.columns]
                for c in missing:
                    X[c] = np.nan
                # Keep only expected columns, in saved order
                X = X.loc[:, feats]
            else:
                # No known feature list; pass everything
                X = df.copy()

            # Helper: convert float days → "X days Y hours"
            def days_hours_fmt(value: float) -> str:
                if pd.isna(value):
                    return "NaN"
                d = int(value)
                h = int(round((value - d) * 24))
                # Guard rounding edge case: 23.999 -> 24h
                if h == 24:
                    d += 1
                    h = 0
                return f"{d} days {h} hours"

            try:
                # Predict
                y_pred = reg.predict(X)

                # Keep only the specified input columns that actually exist
                selected_input_cols = [c for c in COLUMNS_TO_SHOW if c in df.columns]

                filtered_small = pd.concat(
                    [
                        df.loc[:, selected_input_cols].reset_index(drop=True),
                        pd.DataFrame({
                            "pred_los_days": [days_hours_fmt(v) for v in y_pred]
                        }).reset_index(drop=True),
                    ],
                    axis=1
                )

                st.markdown("<h5>Patient Stay Duration Forecast</h5>", unsafe_allow_html=True)
                unique_out = filtered_small.drop_duplicates(subset=["subject_id"])
                unique_out.index = range(1, len(unique_out) + 1)
                st.dataframe(unique_out)

                st.download_button(
                    "⬇️ Download Prediction CSV",
                    data=filtered_small.to_csv(index=False).encode("utf-8"),
                    file_name=out_name,
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Batch prediction failed: {e}")


elif page.startswith("🧭 4"):
    st.subheader("Clustering")
    up = st.file_uploader(
        "Upload CSV for clustering (pipeline will transform/select numeric as needed)",
        type=["csv"]
    )

    if up is None:
        st.info("Upload a CSV to preview and cluster.")
    else:
        # --- Robust CSV load (handles UploadedFile or raw bytes) ---
        try:
            cluster = pd.read_csv(up)
        except Exception:
            up.seek(0)
            cluster = pd.read_csv(io.StringIO(up.getvalue().decode("utf-8", errors="ignore")))

        # --- Define the important columns (pre-selected from your dataset analysis) ---
        COLUMN_RENAME_MAP = {
            "hdb_cluster": "Clusters",
            "n": "Cluster Size",
            "lab_24_hr_creatinine_mean_med": "24h Creatinine (avg)",
            "lab_glucose,_joint_fluid_mean_med": "Glucose (Joint Fluid)",
            "lab_creatinine_clearance_mean_med": "Creatinine Clearance",
            "lab_glucose,_ascites_mean_med": "Glucose (Ascitic Fluid)",
            "lab_hemoglobin_other_mean_med": "Hemoglobin",
            "lab_triglycerides,_ascites_mean_med": "Triglycerides (Ascitic Fluid)",
            "lab_glucose,_body_fluid_mean_med": "Glucose (Body Fluid)",
            "lab_cholesterol,_pleural_mean_med": "Cholesterol (Pleural Fluid)",
            "age_med": "Median Age",
            "lab_creatinine,_urine_mean_med": "Creatinine (Urine)",
            "lab_urine_creatinine_mean_med": "Urine Creatinine",
            "lab_triglycerides_mean_med": "Triglycerides",
            "lab_glucose_mean_med": "Glucose",
            "lab_cholesterol,_total_mean_med": "Total Cholesterol"
        }

        # Keep only important columns
        cols_to_show = [c for c in COLUMN_RENAME_MAP.keys() if c in cluster.columns]
        filtered_small = cluster.loc[:, cols_to_show].copy()

        # Rename to user-friendly names
        filtered_small = filtered_small.rename(columns=COLUMN_RENAME_MAP)

        # Reset index for clean display
        filtered_small.index = range(1, len(filtered_small) + 1)

        st.markdown("<h5>Patient Cluster Profiles</h5>", unsafe_allow_html=True)
        st.dataframe(filtered_small.head(50), use_container_width=True)

elif page.startswith("🧾 5"):
    st.subheader("Association Rules")

    # --- File uploader (give it a persistent key so we can reset it) ---
    assoc_up = st.file_uploader(
        "Upload CSV for association rules (one-hot or transactional format)",
        type=["csv"],
        key="assoc_uploader",
    )

    if assoc_up is None:
        st.info("Upload a CSV to preview for association rules.")
    else:
        # Only parse if we haven't already parsed this selection this run
        try:
            assoc = pd.read_csv(assoc_up)
        except Exception:
            assoc_up.seek(0)
            assoc = pd.read_csv(io.StringIO(assoc_up.getvalue().decode("utf-8", errors="ignore")))

        # Store in session (optional) to avoid duplicate work later in the page
        st.session_state["assoc_df"] = assoc

        # Reset index for clean display
        assoc = assoc.copy()
        assoc.index = range(1, len(assoc) + 1)

        st.markdown("<h5>Diagnostic & Treatment Co-occurrence Rules</h5>", unsafe_allow_html=True)
        st.dataframe(assoc, use_container_width=True)

elif page.startswith("📈 6"):
    st.subheader("Time Series — ED Vitals (upload your own LSTM forecaster)")
    st.caption(
        "Upload a TorchScript model (.pt) OR a state_dict checkpoint (.pt) "
        "+ a feature scaler (joblib/pkl) + meta.json, then predict next-step heart rate (bpm) "
        "from ED vitals resampled to 15-min."
    )

    import io, json, joblib, torch, os, hashlib
    from pathlib import Path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Upload artifacts ----------
    st.markdown("### Upload model + scaler + meta")
    model_up  = st.file_uploader("Model (.pt) — TorchScript **or** state_dict", type=["pt", "pth"], key="vitals_model_up")
    scaler_up = st.file_uploader("Scaler (joblib/pkl)", type=["joblib", "pkl"], key="vitals_scaler_up")
    meta_up   = st.file_uploader("Meta (meta.json)", type=["json"], key="vitals_meta_up")

    @st.cache_resource(show_spinner=False)
    def _load_user_artifacts(_model_bytes: bytes, _scaler_bytes: bytes, _meta_bytes: bytes):
        if _model_bytes is None or _scaler_bytes is None or _meta_bytes is None:
            raise ValueError("Please upload model (.pt), scaler (joblib/pkl), and meta.json.")

        tmp_dir = Path("./uploads/vitals_user"); tmp_dir.mkdir(parents=True, exist_ok=True)
        model_p  = tmp_dir / "model.pt"
        scaler_p = tmp_dir / "scaler.joblib"
        meta_p   = tmp_dir / "meta.json"
        model_p.write_bytes(_model_bytes)
        scaler_p.write_bytes(_scaler_bytes)
        meta_p.write_bytes(_meta_bytes)

        # -------- meta --------
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        SEQ_LEN   = int(meta.get("seq_len", 8))
        FEATURES_META = list(meta.get("features", []))
        TARGET    = str(meta.get("target", "heart_rate"))

        # target scaling info (either z-score or min-max)
        TARGET_SCALING = str(meta.get("target_scaling", "zscore")).lower()
        y_mean = float(meta.get("y_mean", 0.0))
        y_std  = float(meta.get("y_std", 1.0)) or 1.0
        y_min  = meta.get("y_min", None)
        y_max  = meta.get("y_max", None)
        y_min  = float(y_min) if y_min is not None else None
        y_max  = float(y_max) if y_max is not None else None

        scaler = joblib.load(scaler_p)

        # -------- model(s) --------
        model_js, model_sd = None, None
        try:
            # keep TorchScript on CPU to avoid hidden-state device mismatch
            model_js = torch.jit.load(str(model_p), map_location="cpu").eval()
        except Exception:
            payload = torch.load(str(model_p), map_location=device)
            input_dim = int(payload.get("input_dim", len(FEATURES_META)))
            hidden    = int(payload.get("hidden", 64))
            layers    = int(payload.get("layers", 1))
            dropout   = float(payload.get("dropout", 0.1))

            class LSTMReg(torch.nn.Module):
                def __init__(self, input_dim, hidden, layers, dropout):
                    super().__init__()
                    self.lstm = torch.nn.LSTM(
                        input_dim, hidden, num_layers=layers, batch_first=True,
                        dropout=dropout if layers > 1 else 0.0
                    )
                    self.head = torch.nn.Sequential(
                        torch.nn.Linear(hidden, 64),
                        torch.nn.ReLU(),
                        torch.nn.Linear(64, 1)
                    )
                def forward(self, x):
                    out, _ = self.lstm(x)
                    last = out[:, -1, :]
                    return self.head(last)

            model_sd = LSTMReg(input_dim, hidden, layers, dropout).to(device).eval()
            state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            model_sd.load_state_dict(state_dict)

        # -------- reconcile features with scaler --------
        if hasattr(scaler, "feature_names_in_") and scaler.feature_names_in_ is not None:
            FEATURES_USED = list(map(str, scaler.feature_names_in_))
            if FEATURES_META and set(FEATURES_USED) != set(FEATURES_META):
                st.warning("meta.json features differ from scaler features; using scaler.feature_names_in_.")
        else:
            n_expected = int(getattr(scaler, "n_features_in_", len(FEATURES_META)))
            base_pref = [f for f in ["sbp","dbp","resp_rate","spo2","temperature"] if f in FEATURES_META] or FEATURES_META[:]
            FEATURES_USED = base_pref[:n_expected] if len(base_pref) >= n_expected \
                            else base_pref + [f"__pad_{i}" for i in range(n_expected - len(base_pref))]

        scaler_means_named = {}
        if hasattr(scaler, "mean_") and hasattr(scaler, "n_features_in_"):
            if hasattr(scaler, "feature_names_in_") and scaler.feature_names_in_ is not None:
                scaler_means_named = {str(k): float(v) for k, v in zip(scaler.feature_names_in_, scaler.mean_)}

        return {
            "meta": meta,
            "scaler": scaler,
            "model_js": model_js,          # CPU
            "model_sd": model_sd,          # CUDA if available
            "seq_len": SEQ_LEN,
            "features_meta": FEATURES_META,
            "features_used": FEATURES_USED,
            "scaler_means_named": scaler_means_named,
            "target": TARGET,
            # target scaling bundle
            "target_scaling": TARGET_SCALING,
            "y_mean": y_mean, "y_std": y_std,
            "y_min": y_min,   "y_max": y_max,
        }

    arts = None
    if model_up and scaler_up and meta_up:
        try:
            arts = _load_user_artifacts(model_up.getbuffer(), scaler_up.getbuffer(), meta_up.getbuffer())
            st.success("Models uploaded and loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load artifacts: {e}")

    # ---------- Performance options ----------
    st.markdown("### ⚙️ Performance options")
    fast_mode = st.checkbox("Fast mode (sample a subset of stays)", value=True)
    max_stays = st.number_input("Max stays to process (fast mode)", min_value=10, max_value=5000, value=200, step=10)
    max_rows  = st.number_input("Hard cap on vitals rows to parse", min_value=50_000, max_value=5_000_000, value=500_000, step=50_000)
    max_windows = st.number_input("Hard cap on windows to predict", min_value=10_000, max_value=2_000_000, value=300_000, step=10_000)
    batch_size_pred = st.number_input("Prediction batch size", min_value=256, max_value=8192, value=2048, step=256)

    # ---------- Upload data ----------
    st.markdown("### Upload data")
    vitals_up = st.file_uploader(
        "Vitals CSV (required) — columns like: stay_id, charttime, heart_rate, sbp, dbp, resp_rate, spo2, temperature",
        type=["csv"], key="ts_vitals")

    if arts is None:
        st.info("Upload model, scaler, and meta first."); st.stop()
    if vitals_up is None:
        st.info("Upload a vitals CSV to start."); st.stop()

    import pandas as pd, numpy as np

    SEQ_LEN        = arts["seq_len"]
    FEATURES_USED  = arts["features_used"]
    TARGET         = arts["target"]
    scaler         = arts["scaler"]
    model_js       = arts["model_js"]
    model_sd       = arts["model_sd"]

    # --- target scaling params ---
    TARGET_SCALING = arts["target_scaling"]
    y_mean, y_std  = arts["y_mean"], (arts["y_std"] if arts["y_std"] != 0 else 1.0)
    y_min, y_max   = arts["y_min"], arts["y_max"]

    # inverse transform helper
    def inverse_target(y_norm: np.ndarray) -> np.ndarray:
        mode = (TARGET_SCALING or "zscore").lower()
        if mode in ("z", "zscore", "standard", "standardize"):
            return y_norm * y_std + y_mean
        if mode in ("minmax", "min-max", "mm"):
            if y_min is None or y_max is None or not np.isfinite(y_min) or not np.isfinite(y_max):
                st.warning("target_scaling=minmax but y_min/y_max not found; falling back to z-score (mean/std).")
                return y_norm * y_std + y_mean
            return y_norm * (y_max - y_min) + y_min
        # unknown mode → try to infer
        if (y_min is not None) and (y_max is not None):
            return y_norm * (y_max - y_min) + y_min
        return y_norm * y_std + y_mean

    vitals_bytes = vitals_up.getvalue()

    def _hash_bytes(b: bytes) -> str:
        return hashlib.md5(b).hexdigest() if b is not None else ""

    @st.cache_data(show_spinner=False)
    def _read_csv_from_bytes(_csv_bytes: bytes, _max_rows: int) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(_csv_bytes), low_memory=False, nrows=int(_max_rows))

    @st.cache_data(show_spinner=False)
    def _standardize_vital_cols_cached(_bytes_hash: str, _df: pd.DataFrame):
        vs = _df.copy()
        vs.columns = [c.lower() for c in vs.columns]
        canon_map = {
            "heart_rate":"heart_rate","heartrate":"heart_rate","hr":"heart_rate",
            "sbp":"sbp","systolic_bp":"sbp","systolic":"sbp",
            "dbp":"dbp","diastolic_bp":"dbp","diastolic":"dbp",
            "resp_rate":"resp_rate","respiratory_rate":"resp_rate","rr":"resp_rate",
            "spo2":"spo2","o2sat":"spo2","oxygen_saturation":"spo2",
            "temperature":"temperature","temp":"temperature","temp_c":"temperature","temp_f":"temp_f",
        }
        ren = {c: canon_map[c.lower()] for c in vs.columns if c.lower() in canon_map}
        vs = vs.rename(columns=ren)
        time_col = next((c for c in ["charttime","edcharttime","time","chart_time"] if c in vs.columns), None)
        stay_col = next((c for c in ["stay_id","edstay_id","ed_stay_id"] if c in vs.columns), None)
        if time_col is None or stay_col is None:
            raise RuntimeError("Could not detect time/stay columns in vitals CSV.")
        if "temp_f" in vs.columns and "temperature" not in vs.columns:
            vs["temperature"] = (pd.to_numeric(vs["temp_f"], errors="coerce") - 32.0) * (5.0/9.0)
            vs = vs.drop(columns=["temp_f"])
        for c in ["heart_rate","sbp","dbp","resp_rate","spo2","temperature"]:
            if c in vs.columns: vs[c] = pd.to_numeric(vs[c], errors="coerce")
        vs[time_col] = pd.to_datetime(vs[time_col], errors="coerce")
        vs = vs.dropna(subset=[time_col, stay_col])
        return vs, time_col, stay_col

    @st.cache_data(show_spinner=False)
    def _resample_15min_cached(_key: str, _vs: pd.DataFrame, _time_col: str, _stay_col: str) -> pd.DataFrame:
        want = [c for c in ["sbp","dbp","resp_rate","spo2","temperature","heart_rate"] if c in _vs.columns]
        out = []
        groups = list(_vs.groupby(_stay_col, sort=False))
        if fast_mode: groups = groups[:int(max_stays)]
        for sid, g in groups:
            if g.empty: continue
            g = g[[ _time_col ] + want].copy().set_index(_time_col).sort_index()
            r = g.resample("15min").mean().interpolate(method="time").ffill().bfill().reset_index().rename(columns={_time_col:"time"})
            r[_stay_col] = sid
            out.append(r)
        if not out: return pd.DataFrame(columns=["time", _stay_col] + want)
        return pd.concat(out, ignore_index=True)

    vit_hash = _hash_bytes(vitals_bytes)

    vs_raw = _read_csv_from_bytes(vitals_bytes, max_rows)
    try:
        vs_std, time_col, stay_col = _standardize_vital_cols_cached(vit_hash, vs_raw)
    except Exception as e:
        st.error(f"Preprocess error: {e}"); st.stop()

    res = _resample_15min_cached(vit_hash + f":{int(fast_mode)}:{int(max_stays)}", vs_std, time_col, stay_col)
    if res.empty:
        st.error("Resampling produced no rows. Check the input columns and data quality."); st.stop()

    # Ensure all scaler-required features exist; missing -> NaN (handled below)
    for f in FEATURES_USED:
        if f not in res.columns: res[f] = np.nan

    need_cols = ["time", stay_col] + FEATURES_USED + ([TARGET] if TARGET in res.columns else [])
    res = res[[c for c in need_cols if c in res.columns]].dropna(subset=["time"])

    # Build windows EXACTLY in scaler feature order/length
    def _make_windows(df: pd.DataFrame, features_used: list, seq_len: int):
        Xs, times, sids, ys = [], [], [], []
        df = df.sort_values([stay_col, "time"])
        for sid, g in df.groupby(stay_col, sort=False):
            if len(g) <= seq_len: continue
            M = np.column_stack([
                pd.to_numeric(g.get(f, pd.Series(np.nan, index=g.index)), errors="coerce").to_numpy(dtype=float)
                for f in features_used
            ])
            # fill per-column NaNs with median or 0
            for j in range(M.shape[1]):
                col = M[:, j]
                if np.isnan(col).any():
                    med = np.nanmedian(col)
                    M[:, j] = np.where(np.isfinite(col), col, med if np.isfinite(med) else 0.0)
            for i in range(len(M) - seq_len):
                Xs.append(M[i:i+seq_len, :])
                times.append(g.iloc[i+seq_len]["time"])
                sids.append(sid)
                ys.append(g.iloc[i+seq_len][TARGET] if TARGET in g.columns else np.nan)
        if not Xs:
            return np.empty((0, seq_len, len(features_used))), np.array([]), np.array([]), np.array([])
        return np.array(Xs), np.array(times), np.array(sids), np.array(ys, dtype=float)

    X_all, t_pred, sid_arr, y_true = _make_windows(res, FEATURES_USED, SEQ_LEN)
    if X_all.shape[0] == 0:
        st.error("No valid windows could be created (need at least SEQ_LEN rows per stay with required vitals)."); st.stop()

    n_expected = int(getattr(scaler, "n_features_in_", X_all.shape[-1]))
    if X_all.shape[-1] != n_expected:
        st.error(f"Feature count mismatch after alignment: windows have {X_all.shape[-1]} features but scaler expects {n_expected}. Double-check meta/scaler."); st.stop()

    X_scaled = scaler.transform(X_all.reshape(-1, X_all.shape[-1])).reshape(X_all.shape).astype("float32", copy=False)

    # ---------- Predict (batched) ----------
    def predict_batched(x_np: np.ndarray, batch: int) -> np.ndarray:
        out_list = []
        with torch.no_grad():
            if model_js is not None:
                for i in range(0, len(x_np), batch):
                    xb = torch.from_numpy(x_np[i:i+batch]).to("cpu")
                    yb = model_js(xb)
                    out_list.append(yb.detach().cpu().numpy())
            else:
                param_dev = next(model_sd.parameters()).device
                model_sd.eval()
                for i in range(0, len(x_np), batch):
                    xb = torch.from_numpy(x_np[i:i+batch]).to(param_dev)
                    yb = model_sd(xb)
                    out_list.append(yb.detach().cpu().numpy())
        return np.concatenate(out_list, axis=0).ravel()

    y_hat_norm = predict_batched(X_scaled, int(batch_size_pred))

    # --- Inverse-scaling from meta (z-score or min-max) ---
    mode   = str(arts.get("target_scaling", "zscore")).lower()
    y_mean = float(arts.get("y_mean", 0.0))
    y_std  = float(arts.get("y_std", 1.0)) or 1.0
    y_min  = arts.get("y_min", None)
    y_max  = arts.get("y_max", None)

    if mode in ("minmax", "min-max", "mm") and y_min is not None and y_max is not None:
        y_pred = y_hat_norm * (float(y_max) - float(y_min)) + float(y_min)
        st.caption("Using min–max inverse scaling from meta.json")
    else:
        y_pred = y_hat_norm * y_std + y_mean
        st.caption("Using z-score inverse scaling from meta.json")

    # ---------- Robust auto-calibration (only if truth exists & preds look tiny) ----------
    have_truth = TARGET in res.columns
    need_fix   = np.nanmedian(y_pred) < 20  # HR shouldn’t be single digits

    if have_truth and need_fix:
        # Align preds and truths on (stay_id, pred_time)
        tmp = pd.DataFrame({
            "stay_id": sid_arr,
            "pred_time": pd.to_datetime(t_pred),
            "_pred": y_pred
        })
        truth = res[["time", stay_col, TARGET]].rename(
            columns={"time": "pred_time", stay_col: "stay_id", TARGET: "_true"}
        )
        tmp = tmp.merge(truth, on=["stay_id", "pred_time"], how="inner").dropna(subset=["_true"])
        if len(tmp) >= 50:
            x = tmp["_pred"].to_numpy(float)
            y = tmp["_true"].to_numpy(float)

            pred_med, true_med = np.median(x), np.median(y)
            pred_iqr = np.subtract(*np.percentile(x, [75, 25]))
            true_iqr = np.subtract(*np.percentile(y, [75, 25]))
            if pred_iqr <= 1e-6:
                a = (true_med / max(pred_med, 1e-6))
            else:
                a = true_iqr / pred_iqr
            b = true_med - a * pred_med

            # refine with least-squares around robust init
            A = np.vstack([x, np.ones_like(x)]).T
            a_ls, b_ls = np.linalg.lstsq(A, y, rcond=None)[0]
            # blend (robust 30%, LS 70%)
            a = 0.3 * a + 0.7 * a_ls
            b = 0.3 * b + 0.7 * b_ls

            y_pred = a * y_pred + b
            st.info(f"Auto-calibrated predictions (a={a:.3f}, b={b:.3f}) using {len(tmp)} rows with truth.")
        else:
            st.warning("Not enough truth rows to auto-calibrate; showing inverse-scaled predictions only.")

    # ---------- Output ----------
    out = pd.DataFrame({
        "stay_id": sid_arr,
        "pred_time": pd.to_datetime(t_pred),
        "pred_heart_rate_bpm": np.round(y_pred, 2),
    })
    if TARGET in res.columns:
        truth = res[["time", stay_col, TARGET]].rename(
            columns={"time":"pred_time", stay_col:"stay_id", TARGET:"true_heart_rate_bpm"}
        )
        out = out.merge(truth, on=["stay_id","pred_time"], how="left")

    st.markdown("### Heart Rate Forecast from ED Vitals")
    uniq_stays = out["stay_id"].dropna().unique().tolist()
    pick_stay = st.selectbox("Filter by stay (optional)", ["(all)"] + [str(int(s)) for s in uniq_stays[:2000]], index=0)

    view = out if pick_stay == "(all)" else out[out["stay_id"] == int(pick_stay)]
    view = view.sort_values(["stay_id","pred_time"]).reset_index(drop=True)
    view.index = range(1, len(view)+1)
    st.dataframe(view.head(200), use_container_width=True)

    st.markdown("#### Forecast plot")
    plot_df = view.set_index("pred_time")[["pred_heart_rate_bpm"]]
    if "true_heart_rate_bpm" in view.columns:
        plot_df["true_heart_rate_bpm"] = view.set_index("pred_time")["true_heart_rate_bpm"]
    st.line_chart(plot_df)

    st.download_button(
        "⬇️ Download predictions (CSV)",
        data=out.sort_values(["stay_id","pred_time"]).to_csv(index=False).encode("utf-8"),
        file_name="ed_vitals_hr_forecast.csv",
        mime="text/csv"
    )

elif page.startswith("🩻 7"):
    st.subheader("Imaging — Chest X-ray CNN")

    import torch
    import numpy as np
    import pandas as pd
    from PIL import Image
    from pathlib import Path
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Allow numpy scalar for safe loading paths (PyTorch 2.6+)
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass

    # --- The exact model used in training (so we can load `model_state`)
    class PneumoniaCNN(nn.Module):
        def __init__(self, in_ch=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch,  32, kernel_size=3, stride=1, padding=1)  
            self.bn1   = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2, 2)  

            self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
            self.drop2 = nn.Dropout(0.1)
            self.bn2   = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2, 2)  

            self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
            self.bn3   = nn.BatchNorm2d(64)
            self.pool3 = nn.MaxPool2d(2, 2)  

            self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
            self.drop4 = nn.Dropout(0.2)
            self.bn4   = nn.BatchNorm2d(128)
            self.pool4 = nn.MaxPool2d(2, 2)  

            self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
            self.drop5 = nn.Dropout(0.2)
            self.bn5   = nn.BatchNorm2d(256)
            self.pool5 = nn.MaxPool2d(2, 2)  

            self.flat  = nn.Flatten()
            self.fc1   = nn.Linear(256*4*4, 128)
            self.dropf = nn.Dropout(0.2)
            self.head  = nn.Linear(128, 1)  

        def forward(self, x):
            x = F.relu(self.conv1(x)); x = self.bn1(x); x = self.pool1(x)
            x = F.relu(self.conv2(x)); x = self.drop2(x); x = self.bn2(x); x = self.pool2(x)
            x = F.relu(self.conv3(x)); x = self.bn3(x); x = self.pool3(x)
            x = F.relu(self.conv4(x)); x = self.drop4(x); x = self.bn4(x); x = self.pool4(x)
            x = F.relu(self.conv5(x)); x = self.drop5(x); x = self.bn5(x); x = self.pool5(x)
            x = self.flat(x)
            x = F.relu(self.fc1(x)); x = self.dropf(x)
            logits = self.head(x)
            return logits

    # --- Locate a checkpoint
    MODELS_DIR = Path("Models/CNN_Model")
    ckpt_path = None
    for cand in [MODELS_DIR / "best_by_val_acc.pt", MODELS_DIR / "last.pt"]:
        if cand.exists():
            ckpt_path = str(cand)
            break

    if ckpt_path is None:
        st.error(f"Could not find a checkpoint in {MODELS_DIR}. Expecting best_by_val_acc.pt or last.pt.")
        st.stop()

    # --- Loader that supports: TorchScript, pickled nn.Module, or dict with `model_state`
    def load_xray_model(best_path: str, device: str):
        """
        Returns (model.eval().to(device), classes, img_size)
        """
        # 1) TorchScript (arch-free)
        try:
            model_js = torch.jit.load(best_path, map_location=device)
            model_js.eval()
            classes = getattr(model_js, "classes", ["NORMAL", "PNEUMONIA"])
            img_size = int(getattr(model_js, "img_size", 224))
            return model_js, classes, img_size
        except Exception:
            pass

        # 2) Pickled nn.Module OR dict
        #    Explicitly set weights_only=False to allow full pickles (PyTorch 2.6+ changed default).
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)

        # 2a) Entire model pickled
        if isinstance(ckpt, torch.nn.Module):
            model = ckpt.to(device).eval()
            classes = getattr(model, "classes", ["NORMAL", "PNEUMONIA"])
            img_size = int(getattr(model, "img_size", 224))
            return model, classes, img_size

        # 2b) Dict-format checkpoints (your training script)
        if isinstance(ckpt, dict):
            classes = ckpt.get("classes", ["NORMAL", "PNEUMONIA"])
            img_size = int(ckpt.get("img_size", 150))  # your training default is 150

            # Preferred key from your training loop
            state_dict = ckpt.get("model_state", None)
            if state_dict is None:
                # some older code uses "state_dict"
                state_dict = ckpt.get("state_dict", None)

            if isinstance(state_dict, dict):
                model = PneumoniaCNN(in_ch=1)
                # Clean common prefixes if any (not needed for your script, but safe)
                cleaned = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        k = k[len("module."):]
                    elif k.startswith("model."):
                        k = k[len("model."):]
                    cleaned[k] = v
                missing, unexpected = model.load_state_dict(cleaned, strict=False)
                if missing or unexpected:
                    st.info(f"Loaded with strict=False. Missing keys: {len(missing)}, unexpected: {len(unexpected)}")
                model = model.to(device).eval()
                return model, classes, img_size

        # 2c) Anything else → error with guidance
        raise RuntimeError(
            "Unsupported checkpoint format. Expected TorchScript, a pickled nn.Module, "
            "or a dict containing 'model_state'/'state_dict'."
        )

    # ---- Load model
    try:
        model, classes, default_img_size = load_xray_model(ckpt_path, device=device)
        st.success(f"Loaded model from: {ckpt_path}")
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        st.stop()

    # ---- Inference UI
    up_img = st.file_uploader("Upload a chest X-ray (jpg/png)", type=["jpg", "jpeg", "png"])
    if up_img:
        image = Image.open(up_img).convert("RGB")
        st.image(image, caption="Uploaded X-ray", width=256)

        # Use your existing preprocessing util (must convert to 1x1xHxW normalized tensor)
        img_t = preprocess_xray(image, img_size=default_img_size).to(device)

        with torch.no_grad():
            logits = model(img_t).squeeze(0)

        # Binary head → sigmoid over single logit
        if not isinstance(logits, torch.Tensor) or logits.ndim == 0 or logits.numel() == 1:
            logits = torch.as_tensor(logits)
        if logits.numel() == 1:
            prob = torch.sigmoid(logits).item()
            st.write(f"**Pneumonia probability:** {prob:.3f}")
            st.progress(min(1.0, max(0.0, prob)))
            st.success("Prediction: NORMAL" if prob < 0.5 else "Prediction: PNEUMONIA")
        else:
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            if not classes or len(classes) != len(probs):
                classes = [f"class_{i}" for i in range(len(probs))]
            top_idx = int(probs.argmax())
            st.write("**Class probabilities:**")
            st.dataframe(
                pd.DataFrame({"class": classes, "prob": probs}).sort_values("prob", ascending=False),
                use_container_width=True
            )
            st.success(f"Prediction: {classes[top_idx]} ({probs[top_idx]:.3f})")

    # Show test metrics
    for cand in [MODELS_DIR / "test_metrics.csv", MODELS_DIR / "Models" / "test_metrics.csv"]:
        if cand.exists():
            st.caption("Model test metrics:")
            st.dataframe(pd.read_csv(cand))
            break

elif page.startswith("💬 8"):
    # -------------------------------------------------------------------------
    # Safe env guards (place before any transformers import path)
    # -------------------------------------------------------------------------
    import os
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # -------------------------------------------------------------------------
    # Patch minimal accelerate shims (prevents import errors in transformers)
    # -------------------------------------------------------------------------
    def _patch_accelerate_to_avoid_urllib3():
        import sys, types
        from importlib.machinery import ModuleSpec

        def _ensure_pkg(name: str) -> types.ModuleType:
            mod = sys.modules.get(name)
            if mod is None:
                mod = types.ModuleType(name)
                sys.modules[name] = mod
            if not hasattr(mod, "__path__"):
                mod.__path__ = []
            pkg = name.rpartition(".")[0]
            mod.__package__ = pkg if pkg else name
            if not getattr(mod, "__file__", None):
                mod.__file__ = f"<stub {name}>"
            if getattr(mod, "__spec__", None) is None:
                spec = ModuleSpec(name, loader=None)
                try:
                    spec.submodule_search_locations = []  # type: ignore[attr-defined]
                except Exception:
                    pass
                mod.__spec__ = spec
            return mod

        def _ensure_minimal_stubs(acc_mod):
            if not hasattr(acc_mod, "dispatch_model"):
                acc_mod.dispatch_model = lambda model, *a, **k: model

            class _IEW:
                def __enter__(self): return self
                def __exit__(self, *exc): return False
            if not hasattr(acc_mod, "init_empty_weights"):
                acc_mod.init_empty_weights = _IEW

            if not hasattr(acc_mod, "infer_auto_device_map"):
                acc_mod.infer_auto_device_map = lambda *a, **k: {}
            if not hasattr(acc_mod, "get_balanced_memory"):
                acc_mod.get_balanced_memory = lambda *a, **k: {}

            if not hasattr(acc_mod, "PartialState"):
                class PartialState:
                    def __init__(self, *args, **kwargs):
                        self.device = "cpu"
                        self.process_index = 0
                        self.local_process_index = 0
                        self.num_processes = 1
                        self.num_machines = 1
                    def print(self, *a, **k): pass
                    def wait_for_everyone(self): pass
                    def __enter__(self): return self
                    def __exit__(self, *exc): return False
                acc_mod.PartialState = PartialState

            needed = [
                "accelerate.hooks","accelerate.utils","accelerate.commands",
                "accelerate.commands.config","accelerate.commands.config.default",
                "accelerate.big_modeling","accelerate.utils.other","accelerate.utils.bnb",
                "accelerate.state","accelerate.logging",
            ]
            for name in ["accelerate"] + needed:
                _ensure_pkg(name)

            hooks = sys.modules["accelerate.hooks"]
            if not hasattr(hooks, "AlignDevicesHook"):
                class AlignDevicesHook:
                    def __init__(self, *args, **kwargs): pass
                hooks.AlignDevicesHook = AlignDevicesHook
            if not hasattr(hooks, "add_hook_to_module"):
                hooks.add_hook_to_module = lambda *a, **k: None
            if not hasattr(hooks, "remove_hook_from_module"):
                hooks.remove_hook_from_module = lambda *a, **k: None

            bm = sys.modules["accelerate.big_modeling"]
            bm.dispatch_model = acc_mod.dispatch_model
            bm.init_empty_weights = acc_mod.init_empty_weights

            utils = sys.modules["accelerate.utils"]
            defaults = {
                "get_balanced_memory": acc_mod.get_balanced_memory,
                "infer_auto_device_map": acc_mod.infer_auto_device_map,
                "is_xpu_available": lambda *a, **k: False,
                "is_tpu_available": lambda *a, **k: False,
                "is_mps_available": lambda *a, **k: False,
                "is_npu_available": lambda *a, **k: False,
                "extract_model_from_parallel": lambda *a, **k: a[0] if a else None,
                "send_to_device": lambda *a, **k: a[0] if a else None,
                "broadcast_object_list": lambda *a, **k: None,
                "pad_across_processes": lambda *a, **k: a[0] if a else None,
                "wait_for_everyone": lambda *a, **k: None,
                "reduce": lambda tensor, op="sum": tensor,
                "set_module_tensor_to_device": lambda *a, **k: a[0] if a else None,
                "check_tied_parameters_on_same_device": lambda *a, **k: True,
                "get_max_memory": lambda *a, **k: {"cpu": None},
                "find_tied_parameters": lambda *a, **k: {},
                "retie_parameters": lambda *a, **k: None,
                "named_module_tensors": lambda *a, **k: [],
                "find_executable_batch_size": lambda fn, *a, **k: fn,
                "release_memory": lambda *a, **k: None,
                "load_offloaded_weights": lambda *a, **k: None,
                "offload_weight": lambda *a, **k: None,
                "offload_state_dict": lambda state, *a, **k: state,
                "save_offload_index": lambda *a, **k: None,
                "load_offload_index": lambda *a, **k: {},
            }
            for name, fn in defaults.items():
                if not hasattr(utils, name):
                    setattr(utils, name, fn)

            other = sys.modules["accelerate.utils.other"]
            if not hasattr(other, "recursive_getattr"):
                def recursive_getattr(obj, attr, default=None):
                    for p in attr.split("."):
                        obj = getattr(obj, p, None)
                        if obj is None:
                            return default
                    return obj
                other.recursive_getattr = recursive_getattr

            cfg_def = sys.modules["accelerate.commands.config.default"]
            if not hasattr(cfg_def, "write_basic_config"):
                cfg_def.write_basic_config = lambda *a, **k: None

            bnb = sys.modules["accelerate.utils.bnb"]
            if not hasattr(bnb, "has_4bit_bnb_layers"):
                bnb.has_4bit_bnb_layers = lambda *a, **k: False
            if not hasattr(bnb, "load_and_quantize_model"):
                bnb.load_and_quantize_model = lambda *a, **k: None

            state_mod = sys.modules["accelerate.state"]
            if not hasattr(state_mod, "PartialState"):
                state_mod.PartialState = acc_mod.PartialState

            logging_mod = sys.modules["accelerate.logging"]
            if not hasattr(logging_mod, "get_logger"):
                logging_mod.get_logger = lambda *a, **k: type(
                    "L", (), {"info":lambda *a, **k: None, "warning":lambda *a, **k: None, "debug":lambda *a, **k: None}
                )()

        import sys
        if "accelerate" in sys.modules:
            acc = sys.modules["accelerate"]
            if getattr(acc, "__spec__", None) is None:
                spec = ModuleSpec("accelerate", loader=None)
                try:
                    spec.submodule_search_locations = []
                except Exception:
                    pass
                acc.__spec__ = spec
            if not hasattr(acc, "__path__"):
                acc.__path__ = []
            _ensure_minimal_stubs(acc)
            return
        acc = _ensure_pkg("accelerate")
        _ensure_minimal_stubs(acc)

    _patch_accelerate_to_avoid_urllib3()

    # -------------------------------------------------------------------------
    # Imports / constants
    # -------------------------------------------------------------------------
    from pathlib import Path
    import re, json, socket
    import numpy as np
    import pandas as pd
    import streamlit as st
    from typing import Any, Dict, List, Tuple

    DATA = Path(os.getenv("HEALTHAI_DATA_DIR", "data"))
    RAG_DIR = Path(os.getenv("HEALTHAI_RAG_DIR", "Models")) / "RAG_Model"

    # Sentiment paths
    SENT_BASE     = (Path(os.getenv("HEALTHAI_SENT_DIR", "Models")) / "Sentiment_Model").resolve()
    SENT_HF_DIR   = (SENT_BASE / "model").resolve()
    SENT_ADAPTER  = (SENT_BASE / "model_adapter.json").resolve()
    SENT_LABELMAP = (SENT_BASE / "label_map.json").resolve()
    SK_ABS_HINT   = Path(r"D:\HealthAI Project\Models\Sentiment_Model\sklearn_model.joblib").resolve()

    def _as_hf_local_path(p: Path) -> str:
        return str(p.resolve()).replace("\\", "/")

    DEFAULT_CHATBOT_CSV = Path(os.getenv("CHATBOT_CSV", str(DATA / "doctor_consultation_chatbot_multilingual.csv")))

    def read_csv_robust(path_or_buf, nrows=None):
        encs = ["utf-8","utf-8-sig","cp1252","latin1"]
        for enc in encs:
            try:
                return pd.read_csv(path_or_buf, encoding=enc, nrows=nrows)
            except Exception:
                pass
        return pd.read_csv(path_or_buf, engine="python", nrows=nrows)

    def normalize_space(s: Any) -> str:
        s = "" if s is None else str(s)
        return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _has_internet(host="www.google.com", port=443, timeout=2.0) -> bool:
        try:
            socket.create_connection((host, port), timeout=timeout).close()
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # RAG: FAISS (if present) → sklearn fallback
    # -------------------------------------------------------------------------
    class _SklearnIndex:
        def __init__(self, vectors: np.ndarray):
            from sklearn.neighbors import NearestNeighbors
            self.nn = NearestNeighbors(metric="cosine")
            self.nn.fit(vectors.astype("float32"))
            self.d = vectors.shape[1]
            self.vectors = vectors

        def search(self, q, k):
            dist, idx = self.nn.kneighbors(q, n_neighbors=k)
            # emulate FAISS sign (smaller = closer). FAISS returns distances; keep same semantics.
            return dist.astype("float32"), idx.astype("int64")

    @st.cache_resource(show_spinner=False)
    def load_rag():
        index_path = RAG_DIR / "index.faiss"
        meta_path  = RAG_DIR / "meta.jsonl"
        cfg_path   = RAG_DIR / "config.json"

        if not meta_path.exists():
            st.warning(f"RAG metadata not found at '{meta_path}'.")
            return None, None, None

        # Read metas
        metas = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    metas.append(json.loads(line))
                except Exception:
                    pass

        # Try FAISS index first
        if index_path.exists():
            try:
                import faiss  # type: ignore
                index = faiss.read_index(str(index_path))
                cfg = None
                if cfg_path.exists():
                    try:
                        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    except Exception as e:
                        st.warning(f"Could not read RAG config.json: {e}")
                return index, metas, cfg
            except Exception as e:
                st.warning(f"FAISS index read failed: {e}. Falling back to sklearn index…")

        # Fallback: build sklearn index from stored vectors (expects vectors.npy alongside meta)
        vecs_path = RAG_DIR / "vectors.npy"
        if vecs_path.exists():
            try:
                vecs = np.load(vecs_path)
                index = _SklearnIndex(vecs)
                cfg = None
                if cfg_path.exists():
                    try:
                        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    except Exception as e:
                        st.warning(f"Could not read RAG config.json: {e}")
                return index, metas, cfg
            except Exception as e:
                st.error(f"Could not build sklearn fallback index: {e}")
                return None, None, None

        st.warning(f"No FAISS index or vectors found in '{RAG_DIR}'.")
        return None, None, None

    @st.cache_resource(show_spinner=False)
    def load_embedder(cfg: Dict[str, Any] | None):
        backend = (cfg or {}).get("backend", "sbert")
        if backend == "tfidf":
            try:
                import joblib
                tfidf_bundle = joblib.load(RAG_DIR / "tfidf_svd.joblib")
                return ("tfidf", tfidf_bundle)
            except Exception as e:
                st.error(f"TF-IDF backend specified but tfidf_svd.joblib is missing or unreadable: {e}")

        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            model = SentenceTransformer(model_name)
            return ("sbert", model)
        except Exception as e:
            st.warning(f"SBERT embedder load failed ({e}). Using random-vector fallback.")
            return ("random", None)

    def embed_query(text: str, backend_model):
        kind, obj = backend_model
        if kind == "sbert":
            vec = obj.encode([normalize_space(text)], convert_to_numpy=True, normalize_embeddings=True)
            return vec.astype("float32")
        if kind == "tfidf":
            from sklearn.preprocessing import normalize as sk_normalize
            X = obj["vectorizer"].transform([normalize_space(text)])
            if obj.get("svd") is not None:
                X = obj["svd"].transform(X)
                X = sk_normalize(X)
                return X.astype("float32")
            X = sk_normalize(X, norm="l2").toarray()
            return X.astype("float32")
        v = np.random.randn(1, 384)
        v /= max(np.linalg.norm(v), 1e-8)
        return v.astype("float32")

    def check_dim_and_pad(vec: np.ndarray, index) -> np.ndarray:
        d_index = getattr(index, "d", vec.shape[1])
        d_vec = vec.shape[1]
        if d_vec == d_index:
            return vec
        if d_vec < d_index:
            pad = np.zeros((vec.shape[0], d_index - d_vec), dtype=vec.dtype)
            return np.concatenate([vec, pad], axis=1)
        return vec[:, :d_index]

    def search(index, metas, qvec, k=5):
        D, I = index.search(qvec, k)
        out = []
        for d, i in zip(D[0], I[0]):
            if i == -1: continue
            out.append((float(d), metas[i]))
        return out

    def synthesize_answer(question: str, hits: List[Tuple[float, dict]]) -> str:
        if not hits:
            return "I couldn't find a grounded answer in the retrieved context."
        answer = normalize_space(hits[0][1].get("bot_response"))
        if not answer:
            answer = "I couldn't find a grounded answer in the retrieved context."
        return answer

    # -------------------------------------------------------------------------
    # Translator: Marian → MBART-50 → deep-translator → identity
    # -------------------------------------------------------------------------
    @st.cache_resource(show_spinner=False)
    def get_translator(src_lang: str, tgt_lang: str):
        # 1) Try transformers (Marian first, then MBART-50)
        try:
            from transformers import (
                MarianMTModel, MarianTokenizer,
                MBart50TokenizerFast, MBartForConditionalGeneration
            )
            marian_map = {
                ("en","ta"): "Helsinki-NLP/opus-mt-en-ta",
                ("ta","en"): "Helsinki-NLP/opus-mt-ta-en",
                ("en","hi"): "Helsinki-NLP/opus-mt-en-hi",
                ("hi","en"): "Helsinki-NLP/opus-mt-hi-en",
            }
            key = (src_lang, tgt_lang)
            if key in marian_map:
                tok = MarianTokenizer.from_pretrained(marian_map[key])
                mdl = MarianMTModel.from_pretrained(marian_map[key])
                return "marian", tok, mdl, {}

            lang_map = {"en":"en_XX","ta":"ta_IN","hi":"hi_IN","te":"te_IN","mr":"mr_IN","bn":"bn_IN","kn":"kn_IN","ml":"ml_IN","pa":"pa_IN","ur":"ur_PK"}
            tok = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            mdl = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
            return "mbart50", tok, mdl, {"src": lang_map.get(src_lang,"en_XX"),
                                         "tgt": lang_map.get(tgt_lang,"ta_IN")}
        except Exception:
            pass

        # 2) deep-translator (online)
        if _has_internet():
            try:
                from deep_translator import GoogleTranslator
                return "deep-translator", None, GoogleTranslator(source=src_lang, target=tgt_lang), {}
            except Exception:
                pass

        # 3) identity fallback
        class IdentityTranslator:
            def translate(self, text: str) -> str:
                return f"[translation unavailable] {text}"
        return "identity", None, IdentityTranslator(), {}

    def translate_text(backend: str, tok, model, meta: dict, text: str) -> str:
        if backend == "marian":
            batch = tok([text], return_tensors="pt", padding=True)
            out = model.generate(**batch, max_new_tokens=256)
            return tok.batch_decode(out, skip_special_tokens=True)[0]
        if backend == "mbart50":
            tok.src_lang = meta.get("src", "en_XX")
            bos = tok.lang_code_to_id.get(meta.get("tgt", "ta_IN"))
            batch = tok([text], return_tensors="pt", padding=True)
            gen_kwargs = {"max_new_tokens": 256}
            if bos is not None:
                gen_kwargs["forced_bos_token_id"] = bos
            out = model.generate(**batch, **gen_kwargs)
            return tok.batch_decode(out, skip_special_tokens=True)[0]
        if backend == "deep-translator":
            try:
                return model.translate(text)
            except Exception as e:
                return f"[translation unavailable] {e}"
        return f"[translation unavailable] {text}"

    # -------------------------------------------------------------------------
    # Sentiment (HF → sklearn → VADER)
    # -------------------------------------------------------------------------
    class SentimentBackend:
        def __init__(self, hf=None, sk_pipe=None, id2label=None):
            self.hf = hf
            self.sk_pipe = sk_pipe
            self.id2label = id2label or {}
        def available(self) -> bool:
            return (self.hf is not None) or (self.sk_pipe is not None)
        def predict_one(self, text: str) -> Dict[str, Any]:
            if self.hf is not None:
                from torch.nn.functional import softmax
                tok, mdl = self.hf
                inputs = tok([text], return_tensors="pt", truncation=True, max_length=256)
                import torch
                with torch.no_grad():
                    logits = mdl(**inputs).logits
                probs = softmax(logits, dim=-1)[0].detach().cpu().numpy().tolist()
                idx = int(np.argmax(probs))
                label = mdl.config.id2label.get(idx, str(idx))
                return {"label": str(label), "score": float(max(probs))}
            if self.sk_pipe is not None:
                if hasattr(self.sk_pipe, "decision_function"):
                    margin = self.sk_pipe.decision_function([text])
                    if isinstance(margin, np.ndarray) and margin.ndim == 2 and margin.shape[1] > 1:
                        idx = int(np.argmax(margin, axis=1)[0])
                        score = float(sigmoid(margin[0, idx]))
                    else:
                        pred = int(self.sk_pipe.predict([text])[0])
                        score = float(sigmoid(abs(float(np.atleast_1d(margin)[0]))))
                        idx = pred
                else:
                    idx = int(self.sk_pipe.predict([text])[0])
                    score = 1.0
                label = self.id2label.get(idx, str(idx)).upper()
                return {"label": label, "score": score}
            return {"label": "unknown", "score": 0.0}

    @st.cache_resource(show_spinner=False)
    def load_sentiment() -> SentimentBackend:
        # 1) HF local model (if available)
        try:
            if SENT_HF_DIR.is_dir() and (SENT_HF_DIR / "config.json").exists():
                try:
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                except Exception:
                    _patch_accelerate_to_avoid_urllib3()
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                hf_local = _as_hf_local_path(SENT_HF_DIR)
                tok = AutoTokenizer.from_pretrained(hf_local, local_files_only=True)
                mdl = AutoModelForSequenceClassification.from_pretrained(hf_local, local_files_only=True)
                return SentimentBackend(hf=(tok, mdl))
        except Exception as e:
            st.warning(f"HF sentiment model load failed; trying sklearn. ({e})")

        # 2) sklearn pipeline (robust path resolution)
        try:
            import joblib
            candidates: List[Path] = []

            if SENT_ADAPTER.exists():
                try:
                    meta = json.loads(SENT_ADAPTER.read_text(encoding="utf-8"))
                    if meta.get("type") == "sklearn":
                        raw_path = meta.get("path", "sklearn_model.joblib")
                        fixed = raw_path.replace("\\", "/").replace("/Models/sentiment/", "/Models/Sentiment_Model/")
                        p = Path(fixed)
                        if not p.is_absolute():
                            p = (SENT_BASE / p).resolve()
                        candidates.append(p)
                except Exception:
                    pass

            candidates.append((SENT_BASE / "sklearn_model.joblib").resolve())
            candidates.append(SK_ABS_HINT)

            sk_path_found = next((p for p in candidates if p.exists()), None)
            if sk_path_found:
                sk_pipe = joblib.load(str(sk_path_found))
                id2label = {}
                if SENT_LABELMAP.exists():
                    lm = json.loads(SENT_LABELMAP.read_text(encoding="utf-8"))
                    id2label = {int(k): v for k, v in lm.get("id2label", {}).items()} or {
                        i: name for i, name in enumerate(lm.get("labels", []))
                    }
                st.info(f"Loaded sklearn sentiment model from: {sk_path_found}")
                return SentimentBackend(sk_pipe=sk_pipe, id2label=id2label)
            else:
                st.warning(f"sklearn model not found. Tried: {[str(p) for p in candidates]}")
        except Exception as e:
            st.warning(f"sklearn sentiment model load failed. ({e})")

        # 3) VADER fallback
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            import nltk; nltk.download('vader_lexicon', quiet=True)
            sia = SentimentIntensityAnalyzer()
            class VaderBackend(SentimentBackend):
                def __init__(self): super().__init__()
                def predict_one(self, text: str) -> Dict[str, Any]:
                    s = sia.polarity_scores(text)
                    lab = "POSITIVE" if s["compound"] >= 0 else "NEGATIVE"
                    return {"label": lab, "score": float(abs(s["compound"]))}
            return VaderBackend()
        except Exception:
            return SentimentBackend()

    def run_sentiment(backend: SentimentBackend, text: str) -> Dict[str, Any]:
        if backend is None or not backend.available():
            return {"label":"unknown", "score":0.0}
        return backend.predict_one(text)

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------
    tab_chat, tab_trans, tab_sent = st.tabs([
        "🤖 Chatbot (RAG)",
        "🌐 Translator",
        "😊 Sentiment"
    ])

    with tab_chat:
        st.subheader("Patient Triage & FAQ Chatbot")
        st.caption("Retrieves answers (bot_response).")
        index, metas, cfg = load_rag()
        emb = load_embedder(cfg) if index is not None else None

        q = st.text_input("Ask a question (any language):", placeholder="e.g., I have chest pain and shortness of breath...")
        K_FIXED = 5
        if st.button("Answer") and q and index is not None and emb is not None:
            qvec = embed_query(q, emb)
            qvec = check_dim_and_pad(qvec, index)
            hits = search(index, metas, qvec, k=K_FIXED)
            ans = synthesize_answer(q, hits)
            st.markdown(f"**Answer:** {ans}")

    with tab_trans:
        st.subheader("Doctor–Patient Translator")
        st.caption("Translate between English and regional languages (Hindi/Tamil/Telugu/Marathi/Bengali/Kannada/Malayalam/Punjabi/Urdu).")
        lang_pairs = {
            "English ↔ Hindi": ("en","hi"),
            "English ↔ Tamil": ("en","ta"),
            "English ↔ Telugu": ("en","te"),
            "English ↔ Marathi": ("en","mr"),
            "English ↔ Bengali": ("en","bn"),
            "English ↔ Kannada": ("en","kn"),
            "English ↔ Malayalam": ("en","ml"),
            "English ↔ Punjabi": ("en","pa"),
            "English ↔ Urdu": ("en","ur")
        }
        pick = st.selectbox("Language pair", list(lang_pairs.keys()))
        src, tgt = lang_pairs[pick]
        direction = st.radio("Direction", [f"{src}→{tgt}", f"{tgt}→{src}"], horizontal=True)
        text = st.text_area("Text", height=120, placeholder="Type the sentence to translate...")

        if st.button("Translate"):
            s, t = (src, tgt) if direction.endswith("→"+tgt) else (tgt, src)
            backend, tok, mdl, info = get_translator(s, t)
            # ✅ Correct argument order:
            out = translate_text(backend, tok, mdl, info, text)
            if backend == "identity":
                st.info("Translation temporarily unavailable; showing original text.")
            st.success(out)

    with tab_sent:
        st.subheader("Patient Feedback Sentiment")
        st.caption("Loads your finetuned sentiment model (HF) or sklearn model.")
        backend = load_sentiment()
        tx = st.text_area("Feedback / review", placeholder="Doctors were polite but waiting time was long.")
        if st.button("Analyze Sentiment"):
            res = run_sentiment(backend, tx)
            st.metric("Prediction", res["label"], delta=f"confidence {res['score']:.2f}")

        st.divider()
        st.caption("Batch scoring (CSV; auto-detects text columns like Feedback/Review/Comment/Text)")
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None and backend is not None and backend.available():
            df = read_csv_robust(up)
            def norm(s): return str(s).strip().lower().replace(" ", "")
            candidates = ["feedback","review","comment","text","notes","message"]
            text_col = None
            cols_norm = {norm(c): c for c in df.columns}
            for c in candidates:
                if c in cols_norm:
                    text_col = cols_norm[c]; break
            if text_col is None and "Feedback" in df.columns:
                text_col = "Feedback"

            if text_col is None:
                st.error(f"Could not find a text column. Tried: {candidates + ['Feedback']}")
            else:
                preds = [backend.predict_one(str(s)) for s in df[text_col].astype(str).tolist()]
                df["sent_label"] = [p["label"] for p in preds]
                df["sent_score"] = [p["score"] for p in preds]
                st.dataframe(df.head(20))
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download scored CSV", csv, file_name="scored_feedback.csv", mime="text/csv")


elif page.startswith("⚙️"):
    st.subheader("Settings & Logging")
    st.write("Toggle MLflow logging and run a quick test.")
    enable_mlflow = st.checkbox("Enable MLflow (if installed)", value=True)
    if enable_mlflow and mlflow is not None:
        try:
            mlflow.set_experiment("healthai_suite_v7")
            with mlflow.start_run(run_name="streamlit-demo"):
                mlflow.log_param("version","v7")
                mlflow.log_metric("metric_example", 0.9)
            st.success("Logged to MLflow experiment: healthai_suite_v7")
        except Exception as e:
            st.warning(f"MLflow error: {e}")
    else:
        st.info("MLflow disabled or not installed.")

    st.write("SHAP quick test on a tiny synthetic tree:")
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import RandomForestClassifier
        data = load_breast_cancer()
        X, y = data.data, data.target
        clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X,y)
        if shap is not None:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X[:100])
            st.write("Computed SHAP shapes:", [np.array(sv).shape for sv in (shap_values if isinstance(shap_values, list) else [shap_values])])
        else:
            st.info("SHAP not installed.")
    except Exception as e:
        st.warning(f"SHAP demo error: {e}")
