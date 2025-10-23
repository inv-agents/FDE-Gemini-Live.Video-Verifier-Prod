"""
Gemini Live Video Verifier Tool - Main Entry Point

A dual-video comparison tool that analyzes both Gemini and Competitor videos, providing text recognition,
language fluency analysis, and speaker diarization. The tool includes quality assurance validation for both
videos and a multi-page Streamlit interface.
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

help_page = st.Page(
    "pages/help.py",
    title="Help",
    icon="❓"
)

st.logo("assets/inv_logo.jpg", size="large")

pg = st.navigation([video_analyzer_page, help_page])

pg.run()