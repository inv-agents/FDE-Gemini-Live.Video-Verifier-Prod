"""
Gemini Live Video Verifier Tool - Main Entry Point

A video analysis tool that provides text recognition, language fluency analysis, and speaker diarization.
The tool includes quality assurance validation and a multi-page Streamlit interface.
"""

import streamlit as st


st.set_page_config(
    page_title="Gemini Live Video Verifier",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

video_analyzer_page = st.Page(
    "pages/video_analyzer.py",
    title="Video Analyzer",
    icon="🎥",
    default=True
)

metrics_page = st.Page(
    "pages/metrics.py",
    title="Metrics",
    icon="📊"
)

help_page = st.Page(
    "pages/help.py",
    title="Help",
    icon="❓"
)

st.logo("assets/inv_logo.jpg", size="large")

pg = st.navigation([video_analyzer_page, metrics_page, help_page])

pg.run()