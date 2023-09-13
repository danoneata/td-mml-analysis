import json
import os
import pdb
import random

import pandas as pd
import streamlit as st
import seaborn as sns

from matplotlib import pyplot as plt


random.seed(1337)
st.set_page_config(layout="wide")
sns.set_theme("notebook")


def agg_scores(datum, weights):
    scores = datum["scores"]
    return sum(weights[k] * scores[k] for k in scores.keys())


def load_scored_data(lang):
    with open(f"data/cc/analysis-{lang}-cache.json") as f:
        return json.load(f)


lang_to_iso3 = {
    "ar": "ARB",
    "bg": "BEG",
    "bn": "BUL",
    "da": "DAN",
    "de": "DEU",
    "el": "ELL",
    "es": "EST",
    "et": "EST",
    "fr": "FRA",
    "id": "IND",
    "ja": "JPN",
    "ko": "KOR",
    "pt": "POR",
    "ru": "RUS",
    "sw": "SWA",
    "ta": "TAM",
    "tr": "TUR",
    "vi": "VIE",
    "zh": "CMN",
}


langs = "ar bg bn da de el es et fr id ja ko pt ru sw ta tr vi zh".split()  # IGLUE languages
weights_default = {
    "sim-tgt-src-bleu": 1.0,
    "uniformity": 1.0,
}
sort_funcs = {
    "sim-tgt-src-bleu": lambda d, *_: d["scores"]["sim-tgt-src-bleu"],
    "uniformity": lambda d, *_: d["scores"]["uniformity"],
}

u = 0.5
s_script = 0.1
s_non_indo = 0.5
s_indo = 0.7

threshs = {
    "ar": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "bg": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "bn": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "da": {
        "sim-tgt-src-bleu": s_indo,
        "uniformity": u,
    },
    "de": {
        "sim-tgt-src-bleu": s_indo,
        "uniformity": u,
    },
    "el": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "es": {
        "sim-tgt-src-bleu": s_indo,
        "uniformity": u,
    },
    "et": {
        "sim-tgt-src-bleu": s_non_indo,
        "uniformity": u,
    },
    "fr": {
        "sim-tgt-src-bleu": s_indo,
        "uniformity": u,
    },
    "id": {
        "sim-tgt-src-bleu": s_non_indo,
        "uniformity": u,
    },
    "ja": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "ko": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "pt": {
        "sim-tgt-src-bleu": s_indo,
        "uniformity": u,
    },
    "ru": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "sw": {
        "sim-tgt-src-bleu": s_non_indo,
        "uniformity": u,
    },
    "ta": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
    "tr": {
        "sim-tgt-src-bleu": s_non_indo,
        "uniformity": u,
    },
    "vi": {
        "sim-tgt-src-bleu": s_non_indo,
        "uniformity": u,
    },
    "zh": {
        "sim-tgt-src-bleu": s_script,
        "uniformity": u,
    },
}


with st.sidebar:
    lang = st.selectbox("language", langs, index=0)
    st.markdown("---")

    help_text = {
        "sim-tgt-src-bleu": "BLEU score between the source English text and the target translation",
        "uniformity": "number of unique tokens in the translation divided by total number of tokens",
    }

    st.markdown("### thresholds")
    τ = {}
    for f in "sim-tgt-src-bleu uniformity".split():
        τ[f] = st.number_input(f, value=threshs[lang][f])


def proportion_lost(lang):
    data = load_scored_data(lang)
    df = pd.DataFrame([datum["scores"] for datum in data])
    idxs1 = df["uniformity"] > threshs[lang]["uniformity"]
    idxs2 = df["sim-tgt-src-bleu"] > threshs[lang]["sim-tgt-src-bleu"]
    return len(df[idxs1 | idxs2]) / len(df)


ps = [{"lang": lang, "prop": 2.77 * (1 - proportion_lost(lang))} for lang in langs]
fig, ax = plt.subplots(figsize=(6.4, 0.7 * 4.8))
psdf = pd.DataFrame(ps)
psdf = psdf.replace({"lang": lang_to_iso3})
psdf = psdf.sort_values("lang")
sns.barplot(data=psdf, x="lang", y="prop", ax=ax, color="b")
ax.bar_label(ax.containers[0], fmt="%.1f", fontsize="small")
ax.set_xlabel("language")
ax.set_ylabel("num. sentences (× $10^6$)")
ax.tick_params(axis="x", labelrotation=90)
fig.savefig("num-sentences-after-filtering.pdf", bbox_inches="tight")
st.code(sum(p["prop"] for p in ps) + 2.77)

st.markdown("### Kept data")
st.markdown("""
Below we show the fraction of data that is lost by fitlering based on the default thresholds;
these were set as follows

- threshold on uniformity to 0.5 set to all languages.
- threshold on source-target BLEU similarity:
    - 0.1 for languages with different script
    - 0.5 for non Indo European languages with Latin script
    - 0.9 for Indo Europen languages with Latin script
""")

col, _ = st.columns([1, 1])
col.pyplot(fig)
data = load_scored_data(lang)
df = pd.DataFrame([datum["scores"] for datum in data])

fig, axs = plt.subplots(ncols=3, figsize=(6.4 * 3, 4.8))
sns.ecdfplot(df["sim-tgt-src-bleu"], ax=axs[0])
sns.ecdfplot(df["uniformity"], ax=axs[1])
sns.scatterplot(data=df, x="uniformity", y="sim-tgt-src-bleu", ax=axs[2])

axs[0].vlines(τ["sim-tgt-src-bleu"], 0, 1, color="gray")
axs[1].vlines(τ["uniformity"], 0, 1, color="gray")
axs[2].hlines(τ["sim-tgt-src-bleu"], 0, 1, color="gray")
axs[2].vlines(τ["uniformity"], 0, 1, color="gray")

st.markdown(f"### Distribution of scores for `{lang}`")
st.pyplot(fig)

idxs1 = df["uniformity"] > τ["uniformity"]
idxs2 = df["sim-tgt-src-bleu"] > τ["sim-tgt-src-bleu"]
p = len(df[idxs1 | idxs2]) / len(df)
st.markdown("- fraction of above any of the two thresholds: {:.3%}".format(p))
st.markdown("---")

def get_color(s, f):
    return "red" if s > τ[f] else "blue"


def get_colored_score(datum, f):
    s = datum["scores"][f]
    c = get_color(s, f)
    return f"{f} · <span style='color: {c}'>{s:.3f}</span>"


st.markdown(f"### Ranked samples for `{lang}`")
col1, col2 = st.columns(2)
col1.markdown("- sort the translations in decreasing order of one of the features")
sort_by = col2.selectbox("sort by", sort_funcs.keys(), index=0)
st.markdown("---")

data = sorted(data, reverse=True, key=lambda d: sort_funcs[sort_by](d, weights_default))

for rank, datum in enumerate(data, 1):
    score = agg_scores(datum, weights_default)
    st.markdown(
        "{} ◇ `{}` ◇ {} · {}</span>".format(
            rank,
            datum["key"],
            get_colored_score(datum, "sim-tgt-src-bleu"),
            get_colored_score(datum, "uniformity"),
        ),
        unsafe_allow_html=True,
    )
    st.code("en : {}\n{} : {}".format(datum["text-src"], lang, datum["text-tgt"]))
    st.markdown("---")
