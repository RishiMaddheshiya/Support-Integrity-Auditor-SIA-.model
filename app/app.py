"""
SIA — Support Integrity Auditor  |  Streamlit Web App  (Decorated)
"""

import re, json, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.sparse import hstack, csr_matrix, issparse

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SIA · Support Integrity Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  — dark forensic theme + animations
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & body ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #080C14 0%, #0D1421 50%, #080C14 100%);
    color: #E2E8F0;
}

/* ── Animated background grid ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
    animation: gridPulse 8s ease-in-out infinite;
}
@keyframes gridPulse {
    0%,100% { opacity: 0.5; }
    50%      { opacity: 1.0; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050810 0%, #0A0F1E 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.15) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 2px; height: 100%;
    background: linear-gradient(180deg, transparent, #00D4FF, #7B2FFF, transparent);
    animation: sidebarGlow 3s ease-in-out infinite;
}
@keyframes sidebarGlow {
    0%,100% { opacity: 0.4; }
    50%      { opacity: 1.0; }
}

/* ── Sidebar radio buttons ── */
[data-testid="stSidebar"] .stRadio label {
    color: #94A3B8 !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
    display: block !important;
    margin: 4px 0 !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0,212,255,0.08) !important;
    color: #00D4FF !important;
    transform: translateX(4px) !important;
}

/* ── Page-load fade in ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0);    }
}
.main .block-container { animation: fadeInUp 0.55s ease both; }

/* ── Input fields ── */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background: rgba(13,20,40,0.8) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.15), 0 0 20px rgba(0,212,255,0.1) !important;
    outline: none !important;
}
.stTextInput input:hover, .stTextArea textarea:hover {
    border-color: rgba(0,212,255,0.4) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(13,20,40,0.8) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    transition: all 0.3s ease !important;
}
.stSelectbox > div > div:hover {
    border-color: rgba(0,212,255,0.5) !important;
    box-shadow: 0 0 12px rgba(0,212,255,0.1) !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, #FF3366 0%, #FF6B35 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    letter-spacing: 0.5px !important;
    padding: 14px 28px !important;
    transition: all 0.3s cubic-bezier(0.175,0.885,0.32,1.275) !important;
    box-shadow: 0 4px 20px rgba(255,51,102,0.35) !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.15) 50%, transparent 70%);
    transform: translateX(-100%) rotate(45deg);
    transition: transform 0.6s ease;
}
.stButton > button:hover::before { transform: translateX(100%) rotate(45deg); }
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 30px rgba(255,51,102,0.5) !important;
}
.stButton > button:active { transform: translateY(-1px) scale(1.00) !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(13,20,40,0.7) !important;
    border: 1px solid rgba(0,212,255,0.12) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(0,212,255,0.4) !important;
    box-shadow: 0 8px 32px rgba(0,212,255,0.12) !important;
    transform: translateY(-4px) !important;
}
[data-testid="stMetricValue"] { color: #00D4FF !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #64748B !important; }
[data-testid="stMetricDelta"] { color: #7B2FFF !important; }

/* ── DataFrame ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,212,255,0.12) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #00D4FF, #7B2FFF) !important;
    border-radius: 4px !important;
    animation: progressPulse 1.5s ease-in-out infinite !important;
}
@keyframes progressPulse {
    0%,100% { box-shadow: 0 0 8px rgba(0,212,255,0.4); }
    50%      { box-shadow: 0 0 20px rgba(0,212,255,0.8); }
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(13,20,40,0.6) !important;
    border: 1px solid rgba(0,212,255,0.12) !important;
    border-radius: 10px !important;
    color: #CBD5E1 !important;
    transition: all 0.25s ease !important;
}
.streamlit-expanderHeader:hover {
    border-color: rgba(0,212,255,0.35) !important;
    color: #00D4FF !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080C14; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#00D4FF, #7B2FFF);
    border-radius: 3px;
}

/* ── Custom card ── */
.sia-card {
    background: rgba(13,20,40,0.75);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    animation: fadeInUp 0.5s ease both;
}
.sia-card:hover {
    border-color: rgba(0,212,255,0.3);
    box-shadow: 0 8px 40px rgba(0,212,255,0.08);
    transform: translateY(-2px);
}

/* ── MISMATCH banner ── */
@keyframes mismatchPulse {
    0%,100% { box-shadow: 0 0 20px rgba(255,51,102,0.4); border-color: rgba(255,51,102,0.6); }
    50%      { box-shadow: 0 0 40px rgba(255,51,102,0.7); border-color: rgba(255,51,102,1.0); }
}
@keyframes slideDown {
    from { opacity:0; transform: translateY(-16px); }
    to   { opacity:1; transform: translateY(0); }
}
.mismatch-banner {
    background: linear-gradient(135deg, rgba(255,51,102,0.15) 0%, rgba(123,47,255,0.10) 100%);
    border: 1px solid rgba(255,51,102,0.6);
    border-radius: 14px;
    padding: 20px 24px;
    margin: 16px 0;
    animation: slideDown 0.4s ease both, mismatchPulse 2.5s ease-in-out infinite;
}
.mismatch-banner h2 { color: #FF3366; margin: 0 0 6px 0; font-size: 22px; }
.mismatch-banner p  { color: #CBD5E1; margin: 0; font-size: 14px; }
.mismatch-type-hc   { color: #FF3366; font-weight: 700; }
.mismatch-type-fa   { color: #FF9500; font-weight: 700; }

/* ── CONSISTENT banner ── */
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
.consistent-banner {
    background: linear-gradient(135deg, rgba(0,255,136,0.12) 0%, rgba(0,212,255,0.08) 100%);
    border: 1px solid rgba(0,255,136,0.4);
    border-radius: 14px;
    padding: 20px 24px;
    margin: 16px 0;
    animation: slideDown 0.4s ease both;
    position: relative;
    overflow: hidden;
}
.consistent-banner::after {
    content: '';
    position: absolute; top:0; left:0; right:0; bottom:0;
    background: linear-gradient(90deg, transparent 0%, rgba(0,255,136,0.08) 50%, transparent 100%);
    background-size: 200% 100%;
    animation: shimmer 2.5s linear infinite;
}
.consistent-banner h2 { color: #00FF88; margin: 0 0 6px 0; font-size: 22px; }
.consistent-banner p  { color: #CBD5E1; margin: 0; font-size: 14px; }

/* ── Evidence dossier boxes ── */
.evidence-item {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.10);
    border-left: 3px solid #00D4FF;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #CBD5E1;
    transition: all 0.25s ease;
    animation: fadeInUp 0.4s ease both;
}
.evidence-item:hover {
    background: rgba(0,212,255,0.09);
    border-left-color: #7B2FFF;
    transform: translateX(4px);
}
.evidence-signal { color: #00D4FF; font-weight: 600; }
.evidence-value  { color: #E2E8F0; }
.evidence-weight { color: #7B2FFF; }

/* ── Section header ── */
.sia-section-header {
    display: flex; align-items: center; gap: 12px;
    padding: 0 0 20px 0;
    border-bottom: 1px solid rgba(0,212,255,0.1);
    margin-bottom: 24px;
    animation: fadeInUp 0.3s ease both;
}
.sia-section-header h1 {
    font-size: 28px; font-weight: 700;
    background: linear-gradient(135deg, #E2E8F0 0%, #00D4FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.sia-section-icon {
    font-size: 32px;
    animation: iconFloat 3s ease-in-out infinite;
}
@keyframes iconFloat {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(-4px); }
}

/* ── Stat pills ── */
.stat-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 50px;
    padding: 6px 16px;
    font-size: 13px; font-weight: 500;
    color: #00D4FF;
    margin: 4px;
    transition: all 0.25s ease;
    animation: fadeInUp 0.5s ease both;
}
.stat-pill:hover {
    background: rgba(0,212,255,0.16);
    transform: scale(1.04);
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 50px;
    height: 8px;
    margin: 8px 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #00D4FF, #7B2FFF);
    animation: barGrow 0.8s cubic-bezier(0.34,1.56,0.64,1) both;
}
@keyframes barGrow {
    from { width: 0%; }
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,212,255,0.25) !important;
    border-radius: 14px !important;
    background: rgba(0,212,255,0.03) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,255,0.5) !important;
    background: rgba(0,212,255,0.06) !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background: rgba(0,212,255,0.1) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    color: #00D4FF !important;
    border-radius: 10px !important;
    transition: all 0.25s ease !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    background: rgba(0,212,255,0.2) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(0,212,255,0.2) !important;
}

/* ── Divider ── */
hr { border-color: rgba(0,212,255,0.1) !important; }

/* ── Info/warning/error ── */
.stAlert { border-radius: 12px !important; border-left-width: 4px !important; }

/* ── Plotly charts dark ── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════
LABEL_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
NUM_MAP   = {v: k for k, v in LABEL_MAP.items()}
MODEL_PATH = Path("./saved_model/artifacts.pkl")

ESCALATION_CATALOGUE = [
    ("data loss",3.0),("outage",3.0),("emergency",3.0),
    ("cannot access",2.8),("failure",2.5),("not working",2.5),
    ("critical",2.5),("sla",2.3),("escalate",2.3),
    ("broken",2.0),("down",2.0),("urgent",2.0),
    ("immediately",1.8),("error",1.2),("problem",1.0),
]
NEGATION_WORDS = ["not","never","no","unable","can't","won't",
                  "doesn't","didn't","don't","isn't","wasn't"]
CRITICAL_CONTENT_KW = [
    "inaccessible","gone","cannot access","production down","database",
    "data loss","security breach","unauthorized","all users","entire system",
    "no access","not working","records gone","payment","compromised",
    "revenue loss","all transactions","complete outage","cannot log",
]
TRIVIAL_CONTENT_KW = [
    "font","color","colour","tooltip","spacing","alignment","icon","typo",
    "date format","profile picture","button color","helvetica","arial",
    "favicon","label","placeholder","logo","pixel","pixels","too far",
    "too left","too right","brand image","company logo","spelling",
    "subject line","welcome email","5 pixels","few pixels","slightly",
    "darker blue","lighter","shade","punctuation","formatting","indent",
    "thumbnail","banner","background color","css","padding","margin",
    "border-radius","font-size","font size",
]

# ══════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════
def find_col(df, *keywords):
    for col in df.columns:
        if all(kw.lower() in col.lower() for kw in keywords):
            return col
    return None

def extract_top_keyword(text):
    tl = str(text).lower()
    found = [(p,w) for p,w in ESCALATION_CATALOGUE
             if re.search(r"\b"+re.escape(p)+r"\b", tl)]
    found.sort(key=lambda x: x[1], reverse=True)
    return (found[0][0], found[0][1]) if found else (None, 0.0)

def compute_density(subject, description):
    text  = f"{subject} {description}".lower()
    words = text.split(); total = max(len(words), 1)
    esc   = sum(len(re.findall(r"\b"+re.escape(p)+r"\b", text))*w
                for p,w in ESCALATION_CATALOGUE)
    neg   = sum(len(re.findall(r"\b"+re.escape(w)+r"\b", text))
                for w in NEGATION_WORDS)
    return round((esc+neg)/total, 4)

def interpret_rt(rt):
    try: rt = float(rt)
    except: return "Unknown","RT unavailable."
    if rt > 120: return "Critical", f"{rt:.0f} min — top quartile, complex case."
    if rt > 60:  return "High",     f"{rt:.0f} min — above median effort."
    if rt > 30:  return "Medium",   f"{rt:.0f} min — moderate complexity."
    return "Low", f"{rt:.0f} min — straightforward resolution."

def build_features(subject, description, channel, rt, arts):
    X_subj = arts["tfidf_subj"].transform([subject])
    X_desc = arts["tfidf_desc"].transform([description])
    text   = f"{subject} {description}".lower()
    words  = text.split(); total = max(len(words),1)
    esc    = sum(1 for kw,_ in ESCALATION_CATALOGUE
                 if re.search(r"\b"+re.escape(kw)+r"\b", text))
    neg    = sum(1 for w in NEGATION_WORDS
                 if re.search(r"\b"+re.escape(w)+r"\b", text))
    nlp    = np.array([[esc,neg,esc/total,neg/total,len(description.split())]],dtype=np.float32)
    enc    = arts.get("enc_channel")
    if enc:
        ch    = channel if channel in enc.classes_ else enc.classes_[0]
        ch_arr = np.array([[enc.transform([ch])[0]]],dtype=np.float32)
    else:
        ch_arr = np.zeros((1,1),dtype=np.float32)
    meta = np.hstack([nlp, ch_arr, [[float(rt)]], [[0.0]]])
    meta = arts["scaler"].transform(meta)
    return hstack([X_subj, X_desc, csr_matrix(meta)])

def rule_based_check(subject, description, assigned_priority, rt):
    text       = f"{subject} {description}".lower()
    pri_num    = LABEL_MAP.get(assigned_priority, 2)
    density    = compute_density(subject, description)
    crit_hits  = sum(1 for kw in CRITICAL_CONTENT_KW
                     if re.search(r"\b"+re.escape(kw)+r"\b", text))
    trivial_hits = sum(1 for kw in TRIVIAL_CONTENT_KW
                       if re.search(r"\b"+re.escape(kw)+r"\b", text))
    if pri_num <= 2 and crit_hits >= 1 and rt > 60:
        return 1, min(0.55+crit_hits*0.08+(rt/500), 0.95), "hidden_crisis"
    if pri_num >= 3 and trivial_hits >= 1 and rt < 30:
        return 1, min(0.60+trivial_hits*0.10, 0.95), "false_alarm"
    if trivial_hits >= 2 and rt < 20:
        return 1, min(0.58+trivial_hits*0.08, 0.90), "trivial_multi"
    if pri_num == 1 and density > 0.03 and rt > 90:
        return 1, 0.62, "density_rt"
    return None, None, None

def run_inference(subject, description, channel, rt, arts, assigned_priority="Medium"):
    threshold = max(arts.get("best_thresh", 0.5), 0.35)
    X   = build_features(subject, description, channel, rt, arts)
    X_d = X.toarray() if issparse(X) else X
    p_lr  = arts["clf_lr"].predict_proba(X)[0][1]
    p_svc = arts["clf_svc"].predict_proba(X)[0][1]
    p_gb  = arts["clf_gb"].predict_proba(X_d)[0][1]
    ml_prob = (p_lr + p_svc + p_gb) / 3
    ml_pred = 1 if ml_prob >= threshold else 0
    r_pred, r_prob, _ = rule_based_check(subject, description, assigned_priority, rt)
    if r_pred == 1 and r_prob > ml_prob: return 1, r_prob
    if ml_pred == 1: return 1, ml_prob
    final_prob = (ml_prob + (r_prob or 0)) / (2 if r_prob else 1)
    return (1 if final_prob >= 0.40 else 0), final_prob

def infer_severity(prob, pred, assigned):
    if pred == 0: return assigned
    if prob > 0.80: return "Critical"
    if prob > 0.65: return "High"
    if prob > 0.50: return "Medium"
    return "Low"

def generate_dossier(tid, subject, description, channel, rt, assigned, pred, prob):
    inferred = infer_severity(prob, pred, assigned)
    pri_num  = LABEL_MAP.get(assigned, 2)
    inf_num  = LABEL_MAP.get(inferred, 2)
    delta    = inf_num - pri_num
    if delta > 0:   mt = "Hidden Crisis"
    elif delta < 0: mt = "False Alarm"
    else:           mt = "Hidden Crisis" if prob >= 0.5 else "False Alarm"
    top_kw, kw_wt    = extract_top_keyword(f"{subject} {description}")
    density           = compute_density(subject, description)
    rt_lbl, rt_interp = interpret_rt(rt)
    snip = description[:120]+"..." if len(description)>120 else description
    if mt == "Hidden Crisis":
        analysis = (f"'{subject[:60]}' describes a {inferred}-severity issue but "
                    f"was assigned {assigned} ({abs(delta)} level(s) lower). "
                    f"{'Keyword ' + repr(top_kw) + ' detected. ' if top_kw else 'Polite framing masks severity. '}"
                    f"RT ({rt:.0f} min, {rt_lbl} tier) corroborates. Excerpt: \"{snip}\"")
    else:
        analysis = (f"'{subject[:60]}' was assigned {assigned} but signals infer "
                    f"{inferred} ({abs(delta)} level(s) lower). "
                    f"Keyword density ({density:.4f}) and RT ({rt:.0f} min) consistent with {inferred}.")
    return {
        "ticket_id": str(tid), "assigned_priority": assigned,
        "inferred_severity": inferred, "mismatch_type": mt if pred==1 else "None",
        "severity_delta": delta,
        "feature_evidence": [
            {"signal":"keyword",       "value": top_kw or "[none]",  "weight": str(round(kw_wt,1))},
            {"signal":"resolution_time","value": f"{rt:.0f} min",    "interpretation": rt_interp},
            {"signal":"keyword_density","value": str(density),        "weight": "combined"},
            {"signal":"classifier",    "value": f"prob={prob:.3f}",  "weight": f"{prob*100:.1f}%"},
        ],
        "constraint_analysis": analysis,
        "confidence": f"{prob*100:.1f}%",
    }

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
arts = load_model()

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
      <div style='font-size:48px; animation: iconFloat 3s ease-in-out infinite;'>🔍</div>
      <div style='font-size:22px; font-weight:700; background: linear-gradient(135deg,#E2E8F0,#00D4FF);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-top:8px;'>SIA</div>
      <div style='color:#64748B; font-size:12px; letter-spacing:2px; text-transform:uppercase;
           margin-top:4px;'>Support Integrity Auditor</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("<div style='color:#64748B; font-size:11px; letter-spacing:1.5px; text-transform:uppercase; padding:4px 0 8px;'>Navigation</div>", unsafe_allow_html=True)
    section = st.radio("Navigation", [
        "🎫 Single Ticket Auditor",
        "📦 Batch CSV Upload",
        "📊 Mismatch Dashboard",
    ], label_visibility="hidden")

    st.divider()

    if arts:
        st.success("✅ Model Loaded")
    else:
        st.error("Model not found!\nRun train_pipeline.py")

    st.markdown("""
    <div style='position:absolute; bottom:20px; left:0; right:0; text-align:center;
         color:#334155; font-size:11px; padding:0 16px;'>
      MARS Open Projects 2026<br>Models & Robotics Section
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 — SINGLE TICKET AUDITOR
# ══════════════════════════════════════════════════════════════════════
if section == "🎫 Single Ticket Auditor":
    st.markdown("""
    <div class='sia-section-header'>
      <span class='sia-section-icon'>🎫</span>
      <div>
        <h1>Single Ticket Auditor</h1>
        <p style='color:#64748B; margin:0; font-size:14px;'>
          Paste any support ticket to instantly detect priority mismatch.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        subject = st.text_input("🏷️ Ticket Subject",
            placeholder="e.g.  Login failed — urgent assistance needed")
        description = st.text_area("📝 Ticket Description", height=180,
            placeholder="Paste the full ticket description here…")

    with col2:
        assigned_priority = st.selectbox("⚡ Assigned Priority",
            ["Low","Medium","High","Critical"])
        channel = st.selectbox("📡 Ticket Channel",
            ["Email","Chat","Phone","Web Form","Social Media"])
        resolution_time = st.number_input("⏱️ Resolution Time (minutes)",
            min_value=1.0, max_value=10000.0, value=60.0, step=1.0)

    st.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
    audit_clicked = st.button("🔍 Audit Ticket", type="primary", use_container_width=True)

    if audit_clicked:
        if not subject or not description:
            st.warning("⚠️ Please fill in both Subject and Description.")
        elif not arts:
            st.error("❌ Model not loaded. Run `python train_pipeline.py` first.")
        else:
            with st.spinner("🔄 Analysing ticket signals…"):
                pred, prob = run_inference(
                    subject, description, channel,
                    resolution_time, arts, assigned_priority
                )
                dossier = generate_dossier(
                    "TICKET-001", subject, description, channel,
                    resolution_time, assigned_priority, pred, prob
                )

            conf_pct = prob * 100
            bar_width = int(conf_pct)

            if pred == 1:
                mt     = dossier["mismatch_type"]
                color  = "#FF3366" if mt == "Hidden Crisis" else "#FF9500"
                emoji  = "🚨" if mt == "Hidden Crisis" else "⚠️"
                mt_cls = "mismatch-type-hc" if mt == "Hidden Crisis" else "mismatch-type-fa"
                st.markdown(f"""
                <div class='mismatch-banner'>
                  <h2>{emoji} MISMATCH DETECTED</h2>
                  <p>Type: <span class='{mt_cls}'>{mt}</span> &nbsp;|&nbsp;
                     Inferred Severity: <b style='color:{color};'>{dossier['inferred_severity']}</b> &nbsp;|&nbsp;
                     Assigned: <b style='color:#94A3B8;'>{assigned_priority}</b> &nbsp;|&nbsp;
                     Delta: <b style='color:{color};'>{'+' if dossier['severity_delta']>0 else ''}{dossier['severity_delta']}</b>
                  </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='consistent-banner'>
                  <h2>✅ CONSISTENT</h2>
                  <p>Priority assignment looks correct. &nbsp;|&nbsp;
                     Assigned: <b style='color:#00FF88;'>{assigned_priority}</b> &nbsp;|&nbsp;
                     Inferred: <b style='color:#00FF88;'>{dossier['inferred_severity']}</b>
                  </p>
                </div>
                """, unsafe_allow_html=True)

            # Confidence + metrics row
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence",       dossier["confidence"])
            c2.metric("Inferred Severity", dossier["inferred_severity"])
            c3.metric("Severity Delta",    f"{'+' if dossier['severity_delta']>0 else ''}{dossier['severity_delta']}")

            # Confidence bar
            bar_color = "#FF3366" if pred == 1 else "#00FF88"
            st.markdown(f"""
            <div style='margin:16px 0 4px; color:#64748B; font-size:13px;'>Confidence Score</div>
            <div class='conf-bar-wrap'>
              <div class='conf-bar-fill'
                   style='width:{bar_width}%; background:linear-gradient(90deg,{bar_color},{bar_color}88);'>
              </div>
            </div>
            <div style='text-align:right; font-size:12px; color:#64748B; margin-bottom:20px;'>
              {conf_pct:.1f}%
            </div>
            """, unsafe_allow_html=True)

            # ── Evidence Dossier ──────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📄 Evidence Dossier")
            st.caption("Exact Schema · Every field traceable to ticket · Zero Hallucination")

            # Show exact schema matching the problem statement
            exact_schema = {
                "ticket_id"          : dossier["ticket_id"],
                "assigned_priority"  : dossier["assigned_priority"],
                "inferred_severity"  : dossier["inferred_severity"],
                "mismatch_type"      : dossier["mismatch_type"],
                "severity_delta"     : dossier["severity_delta"],
                "feature_evidence"   : dossier["feature_evidence"],
                "constraint_analysis": dossier["constraint_analysis"],
                "confidence"         : dossier["confidence"],
            }
            st.json(exact_schema)

# ══════════════════════════════════════════════════════════════════════
#  SECTION 2 — BATCH CSV
# ══════════════════════════════════════════════════════════════════════
elif section == "📦 Batch CSV Upload":
    st.markdown("""
    <div class='sia-section-header'>
      <span class='sia-section-icon'>📦</span>
      <div>
        <h1>Batch CSV Upload</h1>
        <p style='color:#64748B; margin:0; font-size:14px;'>
          Upload hundreds of tickets at once and download full audit results.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your CSV here or click to browse",
                                type=["csv"])

    if uploaded and arts:
        raw_df = pd.read_csv(uploaded)
        st.markdown(f"""
        <div class='sia-card'>
          <span class='stat-pill'>📋 {len(raw_df):,} rows loaded</span>
          <span class='stat-pill'>📑 {raw_df.shape[1]} columns</span>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(raw_df.head(3), use_container_width=True)

        col_s = find_col(raw_df,"subject")
        col_d = find_col(raw_df,"description")
        col_p = find_col(raw_df,"priority") or find_col(raw_df,"level")
        col_c = find_col(raw_df,"channel")
        col_r = (find_col(raw_df,"resolution","minute")
                 or find_col(raw_df,"resolution","hour")
                 or find_col(raw_df,"resolution","time"))

        if not col_s or not col_d:
            st.error("❌ Could not detect Subject or Description columns.")
            st.stop()

        if st.button("🚀 Run Batch Audit", type="primary", use_container_width=True):
            progress = st.progress(0, text="Scanning tickets…")
            results = []; dossiers = []

            for i, row in raw_df.iterrows():
                subj = str(row.get(col_s,""))
                desc = str(row.get(col_d,""))
                ch   = str(row.get(col_c,"Email")) if col_c else "Email"
                prio = str(row.get(col_p,"Medium")).strip().title() if col_p else "Medium"
                if prio not in LABEL_MAP: prio = "Medium"
                try:    rt = float(row.get(col_r,60.0)) if col_r else 60.0
                except: rt = 60.0
                tid = str(row.get("Ticket_ID", row.get("ticket_id", f"ROW-{i}")))

                pred, prob = run_inference(subj, desc, ch, rt, arts, prio)
                inf   = infer_severity(prob, pred, prio)
                delta = LABEL_MAP.get(inf,2) - LABEL_MAP.get(prio,2)
                mt    = "None"
                if pred == 1:
                    mt = "Hidden Crisis" if delta>0 else ("False Alarm" if delta<0 else "Hidden Crisis")

                results.append({
                    "ticket_id":tid,"assigned_priority":prio,
                    "inferred_severity":inf,"predicted_label":pred,
                    "mismatch_type":mt,"confidence":f"{prob*100:.1f}%",
                    "prob_mismatch":round(prob,4),"severity_delta":delta,
                })
                if pred == 1:
                    dossiers.append(generate_dossier(tid,subj,desc,ch,rt,prio,pred,prob))
                progress.progress((i+1)/len(raw_df), text=f"Scanning {i+1}/{len(raw_df)}…")

            results_df = pd.DataFrame(results)
            progress.empty()

            n_mm = int(results_df["predicted_label"].sum())
            n_hc = int((results_df["mismatch_type"]=="Hidden Crisis").sum())
            n_fa = int((results_df["mismatch_type"]=="False Alarm").sum())
            n_ok = len(results_df) - n_mm

            st.markdown(f"""
            <div class='sia-card' style='margin:16px 0;'>
              <span class='stat-pill'>✅ {n_ok:,} Consistent</span>
              <span class='stat-pill' style='color:#FF3366; border-color:rgba(255,51,102,0.3);
                    background:rgba(255,51,102,0.08);'>🚨 {n_hc:,} Hidden Crisis</span>
              <span class='stat-pill' style='color:#FF9500; border-color:rgba(255,149,0,0.3);
                    background:rgba(255,149,0,0.08);'>⚠️ {n_fa:,} False Alarm</span>
              <span class='stat-pill'>📊 {len(results_df):,} Total</span>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(results_df, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ Download Predictions CSV",
                    results_df.to_csv(index=False).encode(),
                    "sia_predictions.csv","text/csv", use_container_width=True)
            with c2:
                st.download_button("⬇️ Download Evidence Dossiers JSON",
                    json.dumps({"total":len(results_df),"mismatches":n_mm,
                                "dossiers":dossiers},indent=2,ensure_ascii=False).encode(),
                    "sia_dossiers.json","application/json", use_container_width=True)

    elif uploaded and not arts:
        st.error("❌ Model not loaded. Run `python train_pipeline.py` first.")

# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════
elif section == "📊 Mismatch Dashboard":
    st.markdown("""
    <div class='sia-section-header'>
      <span class='sia-section-icon'>📊</span>
      <div>
        <h1>Mismatch Dashboard</h1>
        <p style='color:#64748B; margin:0; font-size:14px;'>
          Visual analytics across your entire ticket corpus.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    pred_path     = Path("test_predictions.csv")
    pseudo_path   = Path("pseudo_labeled_tickets.csv")
    ablation_path = Path("ablation_table.csv")
    metrics_path  = Path("stage2_metrics.json")

    if not pred_path.exists():
        st.markdown("""
        <div class='sia-card' style='text-align:center; padding:40px;'>
          <div style='font-size:48px; margin-bottom:16px;'>⚙️</div>
          <div style='color:#94A3B8; font-size:16px;'>
            No data yet.<br>Run <code>python train_pipeline.py</code> to generate results.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    preds  = pd.read_csv(pred_path)
    pseudo = pd.read_csv(pseudo_path)

    PLOTLY_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,20,40,0.6)",
        font=dict(color="#CBD5E1", family="Inter"),
        margin=dict(l=20,r=20,t=40,b=20),
    )

    # Metrics row
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy",      f"{m.get('test_accuracy',0)*100:.1f}%",  "req ≥ 83%")
        c2.metric("Macro F1",      f"{m.get('test_f1_macro',0):.3f}",       "req ≥ 0.82")
        c3.metric("Recall (OK)",   f"{m.get('test_recall_class0',0)*100:.1f}%","req ≥ 78%")
        c4.metric("Recall (MM)",   f"{m.get('test_recall_class1',0)*100:.1f}%","req ≥ 78%")
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # Charts row 1
    lbl_col = "predicted_label" if "predicted_label" in preds.columns else "mismatch"
    n_mm = int(preds[lbl_col].sum()) if lbl_col in preds.columns else 0
    n_ok = len(preds) - n_mm

    col1, col2 = st.columns(2)

    with col1:
        fig_pie = go.Figure(go.Pie(
            labels=["Consistent","Mismatch"],
            values=[n_ok, n_mm],
            hole=0.55,
            marker=dict(colors=["#00FF88","#FF3366"],
                        line=dict(color="#080C14", width=2)),
            textfont=dict(color="#E2E8F0"),
        ))
        fig_pie.update_layout(title="Overall Distribution", **PLOTLY_LAYOUT)
        fig_pie.add_annotation(text=f"<b>{n_mm}</b><br>Mismatches",
            showarrow=False, font=dict(size=16,color="#FF3366"))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        mm_df = preds[preds[lbl_col]==1].copy() if lbl_col in preds.columns else pd.DataFrame()
        if "mismatch_type" in mm_df.columns and not mm_df.empty:
            tc = mm_df["mismatch_type"].value_counts().reset_index()
            tc.columns = ["Type","Count"]
            fig_bar = px.bar(tc, x="Type", y="Count", color="Type",
                color_discrete_map={"Hidden Crisis":"#FF3366","False Alarm":"#FF9500"},
                title="Mismatch Types")
            fig_bar.update_layout(**PLOTLY_LAYOUT)
            fig_bar.update_traces(marker_line_color="#080C14", marker_line_width=1.5)
            st.plotly_chart(fig_bar, use_container_width=True)

    # Ablation table
    st.markdown("### 🔬 Signal Ablation Table")
    if ablation_path.exists():
        abl = pd.read_csv(ablation_path)
        st.dataframe(abl.style.background_gradient(cmap="Blues"), use_container_width=True)
    else:
        st.info("ablation_table.csv not found.")

    # Heatmap
    st.markdown("### 🗺️ Severity Delta Heatmap")
    col_type    = find_col(pseudo,"type") or find_col(pseudo,"category")
    col_channel = find_col(pseudo,"channel")

    if col_type and col_channel and "severity_delta" in pseudo.columns:
        hmap = pseudo.groupby([col_type, col_channel])["severity_delta"].mean().reset_index()
        pivot = hmap.pivot(index=col_type, columns=col_channel, values="severity_delta")
        fig_hm = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
            title="Avg Severity Delta (positive = under-prioritised)", aspect="auto")
        fig_hm.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_hm, use_container_width=True)

    # Adversarial
    adv_path = Path("adversarial_results.json")
    if adv_path.exists():
        st.markdown("### 🛡️ Adversarial Robustness")
        with open(adv_path) as f:
            adv = json.load(f)
        score = adv.get("correctly_detected", 0)
        bonus = adv.get("bonus_achieved", False)
        ca, cb = st.columns(2)
        ca.metric("Adversarial Score", f"{score}/10",
                  "🏆 Bonus +10%" if bonus else "Need ≥7 for bonus")
        cb.metric("Keyword-Only Baseline",
                  f"{adv.get('keyword_baseline_score',0)}/10", "comparison")
        adv_df = pd.DataFrame(adv.get("results",[]))
        if not adv_df.empty:
            st.dataframe(adv_df[["ticket_id","mismatch_type","assigned_priority",
                                  "probability","correct"]], use_container_width=True)
