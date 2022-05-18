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


def agg_scores(datum, weights):
    scores = datum["scores"]
    return sum(weights[k] * scores[k] for k in scores.keys())


langs = "ar bg bn da de el fr id ja ko pt ru sw ta tr vi zh".split()  # IGLUE languages
weights_default = {
    "sim-tgt-src-bleu": 1.0,
    "uniformity": 1.0,
}
sort_funcs = {
    "sim-tgt-src-bleu": lambda d, *_: d["scores"]["sim-tgt-src-bleu"],
    "uniformity": lambda d, *_: d["scores"]["uniformity"],
}

threshs = {
    "ar": {
        "sim-tgt-src-bleu": 0.2,
        "uniformity": 0.6,
    },
    "bg": {
        "sim-tgt-src-bleu": 0.2,
        "uniformity": 0.6,
    },
    "bn": {
        "sim-tgt-src-bleu": 0.2,
        "uniformity": 0.5,
    },
    "da": {
        "sim-tgt-src-bleu": 0.7,
        "uniformity": 0.6,
    },
    "de": {
        "sim-tgt-src-bleu": 0.7,
        "uniformity": 0.6,
    },
    "el": {
        "sim-tgt-src-bleu": 0.2,
        "uniformity": 0.6,
    },
    "fr": {
        "sim-tgt-src-bleu": 0.7,
        "uniformity": 0.6,
    },
    "id": {
        "sim-tgt-src-bleu": 0.7,
        "uniformity": 0.6,
    },
    "ja": {
        "sim-tgt-src-bleu": 0.2,
        "uniformity": 0.6,
    },
    "ko": {
        "sim-tgt-src-bleu": 0.2,
        "uniformity": 0.6,
    },
    "pt": {
        "sim-tgt-src-bleu": 0.7,
        "uniformity": 0.6,
    },
    "ru": {
        "sim-tgt-src-bleu": 0.2,
        "uniformity": 0.6,
    },
    "sw": {
        "sim-tgt-src-bleu": 0.7,
        "uniformity": 0.6,
    },
    "ta": {
        "sim-tgt-src-bleu": 0.2,
        "uniformity": 0.6,
    },
    "tr": {
        "sim-tgt-src-bleu": 0.5,
        "uniformity": 0.6,
    },
    "vi": {
        "sim-tgt-src-bleu": 0.5,
        "uniformity": 0.6,
    },
    "zh": {
        "sim-tgt-src-bleu": 0.2,
        "uniformity": 0.6,
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


def load_scored_data(lang):
    with open(f"data/cc/analysis-{lang}-cache.json") as f:
        return json.load(f)


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

st.markdown("### Distribution of scores")
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


st.markdown("### Ranked samples")
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
