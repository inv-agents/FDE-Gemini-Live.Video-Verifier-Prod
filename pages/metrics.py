"""
Admin Dashboard for Gemini Live Video Verifier Tool.

This page provides metrics and analytics for system performance,
user feedback analysis, and video analysis results monitoring.
"""

import logging
import os
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st
from streamlit_gsheets import GSheetsConnection

from shared_components import (
    ConfigurationManager, GoogleSheetsService, GlobalSidebar,
    get_session_manager
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


class AdminMetricsCollector:
    """Collect and analyze metrics for admin dashboard."""
    
    def __init__(self):
        self.service = GoogleSheetsService.get_sheets_service()
        config = ConfigurationManager.get_secure_config()
        self.spreadsheet_id = config.get("verifier_sheet_id")
    
    @st.cache_data(ttl=300, show_spinner=False)
    def _read_sheet_data(_self, sheet_name: str) -> List[List[str]]:
        """Read data from a specific sheet."""
        if not _self.service or not _self.spreadsheet_id:
            return []
        
        try:
            range_name = f"{sheet_name}!A:Z"
            result = _self.service.spreadsheets().values().get(
                spreadsheetId=_self.spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            return values
            
        except Exception as e:
            logger.error(f"Error reading sheet {sheet_name}: {e}")
            return []
    
    def get_user_feedback_metrics(self) -> Dict[str, Any]:
        """Get user feedback analysis metrics."""
        try:
            feedback_data = self._read_sheet_data("User Feedback")
            
            if len(feedback_data) <= 1:
                return {
                    'total_feedback': 0,
                    'positive_feedback': 0,
                    'negative_feedback': 0,
                    'positive_rate': 0.0,
                    'common_issues': [],
                    'recent_feedback': []
                }
            
            feedback_rows = feedback_data[1:]
            
            total_feedback = len(feedback_rows)
            positive_feedback = 0
            negative_feedback = 0
            issue_counts = {}
            recent_feedback = []
            
            for row in feedback_rows:
                if len(row) >= 6:
                    try:
                        timestamp = row[0]
                        feedback_rating = int(row[4]) if row[4].isdigit() else 0
                        feedback_text = row[5] if len(row) > 5 else ""
                        issues = row[6] if len(row) > 6 else ""
                        
                        if feedback_rating == 1:
                            positive_feedback += 1
                        elif feedback_rating == 0:
                            negative_feedback += 1
                        
                        if issues and issues != "None":
                            for issue in issues.split(", "):
                                issue = issue.strip()
                                if issue:
                                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
                        
                        if len(recent_feedback) < 10:
                            recent_feedback.append({
                                'timestamp': timestamp,
                                'rating': feedback_text,
                                'issues': issues
                            })
                            
                    except (ValueError, IndexError):
                        continue
            
            positive_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
            
            common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'total_feedback': total_feedback,
                'positive_feedback': positive_feedback,
                'negative_feedback': negative_feedback,
                'positive_rate': positive_rate,
                'common_issues': common_issues,
                'recent_feedback': recent_feedback
            }
            
        except Exception as e:
            logger.error(f"Error getting user feedback metrics: {e}")
            return {
                'total_feedback': 0,
                'positive_feedback': 0,
                'negative_feedback': 0,
                'positive_rate': 0.0,
                'common_issues': [],
                'recent_feedback': []
            }
    
    def get_video_analysis_metrics(self) -> Dict[str, Any]:
        """Get video analysis results metrics."""
        try:
            analysis_data = self._read_sheet_data("Video Analysis Results")
            
            if len(analysis_data) <= 1:
                return {
                    'total_submissions': 0,
                    'qa_pass_rate': 0.0,
                    'language_distribution': {},
                    'detection_rates': {},
                    'recent_submissions': []
                }
            
            analysis_rows = analysis_data[1:]
            
            total_submissions = len(analysis_rows)
            passed_qa = 0
            language_counts = {}
            detection_stats = {
                'flash_detected': 0,
                'alias_detected': 0,
                'eval_detected': 0,
                'language_passed': 0,
                'voice_passed': 0
            }
            recent_submissions = []
            
            for row in analysis_rows:
                if len(row) >= 20:
                    try:
                        timestamp = row[0]
                        question_id = row[1]
                        submission_status = row[20] if len(row) > 20 else ""
                        
                        if submission_status == "ELIGIBLE":
                            passed_qa += 1
                        
                        detected_language = row[15] if len(row) > 15 else ""
                        if detected_language and detected_language != "Unknown":
                            language_counts[detected_language] = language_counts.get(detected_language, 0) + 1
                        
                        flash_status = row[4] if len(row) > 4 else ""
                        alias_status = row[7] if len(row) > 7 else ""
                        eval_status = row[10] if len(row) > 10 else ""
                        language_status = row[13] if len(row) > 13 else ""
                        voice_status = row[17] if len(row) > 17 else ""
                        
                        if flash_status == "PASS":
                            detection_stats['flash_detected'] += 1
                        if alias_status == "PASS":
                            detection_stats['alias_detected'] += 1
                        if eval_status == "PASS":
                            detection_stats['eval_detected'] += 1
                        if language_status == "PASS":
                            detection_stats['language_passed'] += 1
                        if voice_status == "PASS":
                            detection_stats['voice_passed'] += 1
                        
                        if len(recent_submissions) < 10:
                            recent_submissions.append({
                                'timestamp': timestamp,
                                'question_id': question_id,
                                'status': submission_status,
                                'language': detected_language
                            })
                            
                    except (ValueError, IndexError):
                        continue
            
            qa_pass_rate = (passed_qa / total_submissions * 100) if total_submissions > 0 else 0
            
            detection_rates = {}
            for key, count in detection_stats.items():
                detection_rates[key] = (count / total_submissions * 100) if total_submissions > 0 else 0
            
            return {
                'total_submissions': total_submissions,
                'qa_pass_rate': qa_pass_rate,
                'eligible_submissions': passed_qa,
                'language_distribution': language_counts,
                'detection_rates': detection_rates,
                'recent_submissions': recent_submissions
            }
            
        except Exception as e:
            logger.error(f"Error getting video analysis metrics: {e}")
            return {
                'total_submissions': 0,
                'qa_pass_rate': 0.0,
                'eligible_submissions': 0,
                'language_distribution': {},
                'detection_rates': {},
                'recent_submissions': []
            }
    
    def get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance and usage metrics."""
        try:
            session_manager = get_session_manager()
            active_sessions = session_manager.get_active_sessions()
            
            return {
                'active_sessions': len(active_sessions),
                'total_active_sessions': len(active_sessions),
                'system_status': 'Operational',
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            }
            
        except Exception as e:
            logger.error(f"Error getting system performance metrics: {e}")
            return {
                'active_sessions': 0,
                'total_active_sessions': 0,
                'system_status': 'Unknown',
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            }


class AdminDashboard:
    """Admin dashboard for system metrics and management."""
    
    def __init__(self):
        self.metrics_collector = AdminMetricsCollector()
    
    @staticmethod
    def render():
        """Render the admin dashboard screen."""
        AdminDashboard._render_authenticated_dashboard()
    
    @staticmethod
    def _render_authenticated_dashboard():
        """Render the authenticated admin dashboard."""
        dashboard = AdminDashboard()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("📊 Metrics Dashboard")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
                AdminMetricsCollector._read_sheet_data.clear()
                st.rerun()
        
        st.markdown("---")
        
        with st.spinner("Loading dashboard metrics..."):
            feedback_metrics = dashboard.metrics_collector.get_user_feedback_metrics()
            analysis_metrics = dashboard.metrics_collector.get_video_analysis_metrics()
            system_metrics = dashboard.metrics_collector.get_system_performance_metrics()
        
        st.markdown("### 📈 Overview")
        AdminDashboard._render_summary_cards(feedback_metrics, analysis_metrics, system_metrics)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📈 Analysis Metrics", "💭 User Feedback"])
        
        with tab1:
            AdminDashboard._render_analysis_metrics(analysis_metrics)
        
        with tab2:
            AdminDashboard._render_feedback_metrics(feedback_metrics)
    
    @staticmethod
    def _render_summary_cards(feedback_metrics, analysis_metrics, system_metrics):
        """Render summary metric cards."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%); padding: 24px; border-radius: 12px; border-left: 5px solid #4CAF50; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; color: #2e7d32; font-size: 2.5rem; font-weight: 700;">{analysis_metrics.get('total_submissions', 0)}</h3>
                <p style="margin: 8px 0 0 0; color: #555; font-weight: 600; font-size: 0.95rem;">Total Videos</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            qa_pass_rate = analysis_metrics.get('qa_pass_rate', 0)
            color = "#4CAF50" if qa_pass_rate >= 80 else "#FF9800" if qa_pass_rate >= 60 else "#f44336"
            bg_color = "#e8f5e8" if qa_pass_rate >= 80 else "#fff3e0" if qa_pass_rate >= 60 else "#ffebee"
            
            st.markdown(f"""
            <div style="background: {bg_color}; padding: 24px; border-radius: 12px; border-left: 5px solid {color}; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; color: {color}; font-size: 2.5rem; font-weight: 700;">{qa_pass_rate:.1f}%</h3>
                <p style="margin: 8px 0 0 0; color: #555; font-weight: 600; font-size: 0.95rem;">QA Pass Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            positive_rate = feedback_metrics.get('positive_rate', 0)
            color = "#4CAF50" if positive_rate >= 80 else "#FF9800" if positive_rate >= 60 else "#f44336"
            bg_color = "#e8f5e8" if positive_rate >= 80 else "#fff3e0" if positive_rate >= 60 else "#ffebee"
            
            st.markdown(f"""
            <div style="background: {bg_color}; padding: 24px; border-radius: 12px; border-left: 5px solid {color}; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; color: {color}; font-size: 2.5rem; font-weight: 700;">{positive_rate:.1f}%</h3>
                <p style="margin: 8px 0 0 0; color: #555; font-weight: 600; font-size: 0.95rem;">Positive Feedback</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            active_sessions = system_metrics.get('active_sessions', 0)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f0f8ff 100%); padding: 24px; border-radius: 12px; border-left: 5px solid #2196F3; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; color: #1565C0; font-size: 2.5rem; font-weight: 700;">{active_sessions}</h3>
                <p style="margin: 8px 0 0 0; color: #555; font-weight: 600; font-size: 0.95rem;">Active Sessions</p>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def _render_analysis_metrics(analysis_metrics):
        """Render detailed analysis metrics."""
        st.markdown("#### 📊 Submission Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_subs = analysis_metrics.get('total_submissions', 0)
            st.markdown(f"""
            <div style="background: var(--background-color, #f8f9fa); padding: 16px; border-radius: 8px; text-align: center; border: 2px solid #4CAF50;">
                <h2 style="margin: 0; color: var(--text-color, #333); font-size: 2rem;">{total_subs}</h2>
                <p style="margin: 8px 0 0 0; color: var(--text-color, #666); font-size: 0.9rem;">Total Submissions</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            eligible_subs = analysis_metrics.get('eligible_submissions', 0)
            st.markdown(f"""
            <div style="background: var(--background-color, #f8f9fa); padding: 16px; border-radius: 8px; text-align: center; border: 2px solid #2196F3;">
                <h2 style="margin: 0; color: var(--text-color, #333); font-size: 2rem;">{eligible_subs}</h2>
                <p style="margin: 8px 0 0 0; color: var(--text-color, #666); font-size: 0.9rem;">Eligible Submissions</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            qa_pass = analysis_metrics.get('qa_pass_rate', 0)
            qa_color = "#4CAF50" if qa_pass >= 80 else "#FF9800" if qa_pass >= 60 else "#f44336"
            st.markdown(f"""
            <div style="background: var(--background-color, #f8f9fa); padding: 16px; border-radius: 8px; text-align: center; border: 2px solid {qa_color};">
                <h2 style="margin: 0; color: {qa_color}; font-size: 2rem;">{qa_pass:.1f}%</h2>
                <p style="margin: 8px 0 0 0; color: var(--text-color, #666); font-size: 0.9rem;">QA Pass Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 QA Check Performance")
            detection_rates = analysis_metrics.get('detection_rates', {})
            
            if detection_rates:
                for check_name, rate in detection_rates.items():
                    check_display = {
                        'flash_detected': '🔦 2.5 Flash Detection',
                        'alias_detected': '👤 Alias Name Detection', 
                        'eval_detected': '⚙️ Eval Mode Detection',
                        'language_passed': '🗣️ Language Fluency',
                        'voice_passed': '🎤 Voice Audibility'
                    }.get(check_name, check_name)
                    
                    color = "#4CAF50" if rate >= 80 else "#FF9800" if rate >= 60 else "#f44336"
                    
                    st.markdown(f"""
                    <div style="margin: 12px 0; padding: 12px; background: var(--background-color, #ffffff); border-radius: 8px; border: 1px solid var(--border-color, #e0e0e0);">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span style="font-weight: 600; color: var(--text-color, #333);">{check_display}</span>
                            <span style="color: {color}; font-weight: bold; font-size: 1.1rem;">{rate:.1f}%</span>
                        </div>
                        <div style="background: var(--border-color, #e0e0e0); height: 10px; border-radius: 5px; overflow: hidden;">
                            <div style="background: {color}; height: 10px; width: {rate}%; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No QA check data available yet.")
        
        with col2:
            st.markdown("#### 🌍 Language Distribution")
            language_dist = analysis_metrics.get('language_distribution', {})
            
            if language_dist:
                sorted_languages = sorted(language_dist.items(), key=lambda x: x[1], reverse=True)[:10]
                total_count = sum(language_dist.values())
                
                for lang, count in sorted_languages:
                    percentage = (count / total_count * 100) if total_count > 0 else 0
                    
                    st.markdown(f"""
                    <div style="margin: 12px 0; padding: 12px; background: var(--background-color, #ffffff); border-radius: 8px; border: 1px solid var(--border-color, #e0e0e0);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600; color: var(--text-color, #333); font-size: 1rem;">{lang}</span>
                            <span style="color: var(--text-color, #666); font-weight: 500;">{count} <span style="color: var(--text-color, #999);">({percentage:.1f}%)</span></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No language data available yet.")
    
    @staticmethod
    def _render_feedback_metrics(feedback_metrics):
        """Render user feedback metrics."""
        if feedback_metrics.get('total_feedback', 0) == 0:
            st.info("ℹ️ No user feedback data available yet.")
            return
        
        st.markdown("#### 📊 Feedback Summary")
        
        total = feedback_metrics.get('total_feedback', 0)
        positive = feedback_metrics.get('positive_feedback', 0)
        negative = feedback_metrics.get('negative_feedback', 0)
        positive_rate = feedback_metrics.get('positive_rate', 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: var(--background-color, #f8f9fa); padding: 16px; border-radius: 8px; text-align: center; border: 2px solid #2196F3;">
                <h2 style="margin: 0; color: var(--text-color, #333); font-size: 2rem;">{total}</h2>
                <p style="margin: 8px 0 0 0; color: var(--text-color, #666); font-size: 0.9rem;">Total Feedback</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: var(--background-color, #f8f9fa); padding: 16px; border-radius: 8px; text-align: center; border: 2px solid #4CAF50;">
                <h2 style="margin: 0; color: #4CAF50; font-size: 2rem;">{positive}</h2>
                <p style="margin: 8px 0 0 0; color: var(--text-color, #666); font-size: 0.9rem;">Positive ({positive_rate:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: var(--background-color, #f8f9fa); padding: 16px; border-radius: 8px; text-align: center; border: 2px solid #f44336;">
                <h2 style="margin: 0; color: #f44336; font-size: 2rem;">{negative}</h2>
                <p style="margin: 8px 0 0 0; color: var(--text-color, #666); font-size: 0.9rem;">Negative ({100 - positive_rate:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("#### ⚠️ Common Issues Reported")
        
        common_issues = feedback_metrics.get('common_issues', {})
        if common_issues:
            total_feedback = feedback_metrics.get('total_feedback', 1)
            
            cols = st.columns(2)
            for idx, (issue, count) in enumerate(common_issues):
                percentage = (count / total_feedback * 100)
                
                issue_display = {
                    'flash_presence': '🔦 2.5 Flash Detection Issues',
                    'alias_name_presence': '👤 Alias Name Detection Issues',
                    'eval_mode_presence': '⚙️ Eval Mode Detection Issues',
                    'language_fluency': '🗣️ Language Fluency Issues',
                    'voice_audibility': '🎤 Voice Audibility Issues'
                }.get(issue, issue)
                
                with cols[idx % 2]:
                    st.markdown(f"""
                    <div style="margin: 12px 0; padding: 14px; background: var(--background-color, #fff3cd); border-radius: 8px; border-left: 4px solid #ffc107; border: 1px solid var(--border-color, #ffc107);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 600; color: var(--text-color, #856404); font-size: 0.95rem;">{issue_display}</span>
                            <span style="color: var(--text-color, #856404); font-weight: 700; font-size: 1.1rem;">{count} <span style="font-size: 0.9rem;">({percentage:.1f}%)</span></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("✅ No specific issues reported.")

st.markdown("""
    <style>
    /* Main content styling */
    .main > div { 
        padding-top: 1rem; 
    }

    .stAlert { 
        margin-top: 1rem; 
    }

    /* Ensure proper spacing between sections */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }

    /* Improve metric cards spacing */
    [data-testid="column"] {
        padding: 0.5rem;
    }

    /* Better theme compatibility for custom elements */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #262730;
            --text-color: #fafafa;
            --border-color: #464646;
        }
    }

    @media (prefers-color-scheme: light) {
        :root {
            --background-color: #ffffff;
            --text-color: #333333;
            --border-color: #e0e0e0;
        }
    }

    /* Responsive design for smaller screens */
    @media (max-width: 768px) {
        .stColumns {
            flex-direction: column;
        }
        
        [data-testid="column"] {
            width: 100% !important;
            margin-bottom: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

GlobalSidebar.render_sidebar()

AdminDashboard.render()