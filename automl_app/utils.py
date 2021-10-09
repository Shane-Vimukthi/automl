import streamlit as st
import streamlit.components.v1 as stc

# EDA Pkgs
import pandas as pd

# NLP Pkgs
import spacy
from spacy import displacy
# nlp = spacy.load('en_core_web_sm') # Fixes Error For Deployment for shortlink
from textblob import TextBlob
from collections import Counter


# Data Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import altair as alt

import nltk_utils





HTML_RANDOM_TEMPLATE = """
<div style='padding:10px;background-color:#E1E2E1;
			border-radius: 8px 34px 9px 26px;
-moz-border-radius: 8px 34px 9px 26px;
-webkit-border-radius: 8px 34px 9px 26px;
border: 2px ridge #000000;'>
<h5>Verse of the Day</h5>
<p>{}</p></div>
"""


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
HTML_BANNER = """
    <div style="background-color:#464e5f;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">SuperAutoml App </h1>
    </div>
    """
