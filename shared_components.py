"""
Shared components for the Gemini Live Video Verifier Tool.

This module contains common classes and utilities used across multiple pages
to avoid code duplication and maintain consistency.
"""

import gc
import logging
import os
import re
import shutil
import time
import uuid
import cv2
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st
from streamlit_gsheets import GSheetsConnection
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


class Config:
    """Centralized application configuration constants."""
    
    # Core processing parameters
    DEFAULT_FRAME_INTERVAL: float = 10.0
    
    # Resource limits
    MAX_FILE_SIZE: int = 200 * 1024 * 1024  # 200 MB
    MAX_VIDEO_DURATION: int = 600  # 10 minutes
    MIN_VIDEO_DURATION: int = 30   # 30 seconds
    
    # Detection rule default parameters
    DEFAULT_MIN_FLUENCY_SCORE: float = 0.6
    DEFAULT_MIN_CONFIDENCE: float = 0.3
    DEFAULT_MIN_DURATION: float = 3.0

    DEFAULT_WARNING_SNR_DB: float = 20.0
    DEFAULT_FAIL_SNR_DB: float = 15.0
    DEFAULT_WARNING_NOISE_RATIO: float = 0.25
    DEFAULT_FAIL_NOISE_RATIO: float = 0.40 
    
    # Language configuration for display and Whisper mapping
    LANGUAGE_CONFIG: Dict[str, Tuple[str, str]] = {
        'es-419': ('es-419', 'es'), 'hi-IN': ('hi-IN', 'hi'), 'ja-JP': ('ja-JP', 'ja'), 'ko-KR': ('ko-KR', 'ko'),
        'de-DE': ('de-DE', 'de'), 'en-IN': ('en-IN', 'en'), 'fr-FR': ('fr-FR', 'fr'), 'ar-EG': ('ar-EG', 'ar'),
        'pt-BR': ('pt-BR', 'pt'), 'id-ID': ('id-ID', 'id'), 'ko-JA': ('ko-JA', 'ko'), 'zh-CN': ('zh-CN', 'zh'),
        'ru-RU': ('ru-RU', 'ru'), 'ml-IN': ('ml-IN', 'ml'), 'sv-SE': ('sv-SE', 'sv'), 'te-IN': ('te-IN', 'te'),
        'vi-VN': ('vi-VN', 'vi'), 'tr-TR': ('tr-TR', 'tr'), 'bn-IN': ('bn-IN', 'bn'), 'it-IT': ('it-IT', 'it'),
        'zh-TW': ('zh-TW', 'zh'), 'pl-PL': ('pl-PL', 'pl'), 'nl-NL': ('nl-NL', 'nl'), 'th-TH': ('th-TH', 'th'),
        'ko-ZH': ('ko-ZH', 'ko'),
    }

    @classmethod
    def get_language_display_name(cls, language_code: str) -> str:
        """Get display name for a language code."""
        if language_code and language_code in cls.LANGUAGE_CONFIG:
            return cls.LANGUAGE_CONFIG[language_code][0]
        return language_code if language_code else "unknown"
    
    @classmethod
    def locale_to_whisper_language(cls, locale_code: str) -> str:
        """Convert locale code to Whisper language code."""
        return cls.LANGUAGE_CONFIG.get(locale_code, (None, locale_code))[1]

    @classmethod
    def whisper_language_to_locale(cls, whisper_language: str, target_locale: str = None) -> str:
        """Convert Whisper language code back to locale format."""
        if target_locale and cls.locale_to_whisper_language(target_locale) == whisper_language:
            return target_locale
        return whisper_language if whisper_language else "unknown"

    @classmethod
    def is_portrait_mobile_resolution(cls, width: int, height: int) -> bool:
        """Check if video resolution is portrait mobile phone format."""
        if width >= height:
            return False
        aspect_ratio = height / width
        return 1.2 <= aspect_ratio <= 2.5 and 360 <= width <= 1600
    
    @classmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def validate_video_properties(cls, video_path: str) -> Dict[str, Any]:
        """Validate and extract basic video properties including duration and resolution."""
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': 'Cannot open video file'}
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            duration = total_frames / fps if fps > 0 else 0.0
            
            duration_valid = duration >= cls.MIN_VIDEO_DURATION
            resolution_valid = cls.is_portrait_mobile_resolution(width, height)
            
            return {
                'duration_valid': duration_valid,
                'resolution_valid': resolution_valid,
                'duration': duration,
                'width': width,
                'height': height,
                'min_duration_required': cls.MIN_VIDEO_DURATION
            }
            
        except Exception as e:
            logger.error(f"Video validation failed: {e}")
            return {'error': str(e)}
        finally:
            if cap is not None:
                cap.release()


class ConfigurationManager:
    """Centralized configuration management."""
    
    @classmethod
    @st.cache_data(ttl=600, show_spinner=False)
    def get_secure_config(cls) -> Dict[str, str]:
        """Get secure configuration from Streamlit secrets."""
        try:
            google_config = st.secrets["google"]
            
            return {
                "verifier_sheet_url": google_config["verifier_sheet_url"],
                "verifier_sheet_id": google_config["verifier_sheet_id"],
                "verifier_sheet_name": google_config["verifier_sheet_name"],
                "agent_email_sheet_url": google_config.get("agent_email_sheet_url", "https://docs.google.com/spreadsheets/d/1MpSWC9r3VvkiOmA-N-Knf84bdbwPdXAvgU9-v2T1Pyo"),
                "agent_email_sheet_id": google_config.get("agent_email_sheet_id", "1MpSWC9r3VvkiOmA-N-Knf84bdbwPdXAvgU9-v2T1Pyo")
            }
            
        except KeyError as e:
            raise RuntimeError(f"Required configuration missing: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Cannot load secure configuration: {e}") from e
    
    @classmethod
    @st.cache_data(ttl=600, show_spinner=False)
    def get_google_service_account_info(cls) -> Optional[Dict[str, Any]]:
        """Get Google service account credentials from secrets."""
        try:
            config = st.secrets["connections"]["gsheets"]
            return dict(config) if config else None
        except (KeyError, TypeError) as e:
            logger.debug(f"Google service account configuration not found: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load Google service account info: {e}")
            return None


class GoogleSheetsService:
    """Centralized Google Sheets service for export operations."""
    
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def get_sheets_service() -> Optional['googleapiclient.discovery.Resource']:
        """Initialize Google Sheets service using service account."""
        try:
            scopes = ['https://www.googleapis.com/auth/spreadsheets']
            credentials = None
            
            try:
                service_account_info = ConfigurationManager.get_google_service_account_info()
                if service_account_info:
                    credentials = Credentials.from_service_account_info(service_account_info, scopes=scopes)
            except Exception as e:
                logger.warning(f"Could not load credentials from secrets for Sheets: {e}")
            
            if not credentials:
                logger.error("No Google credentials found for Sheets export")
                return None
            
            return build('sheets', 'v4', credentials=credentials, cache_discovery=False)
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets service: {e}")
            return None


class SessionManager:
    """Session and resource management system."""
    
    def __init__(self):
        self._session_base_dir = Path("sessions")
        self._session_base_dir.mkdir(exist_ok=True)
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return str(uuid.uuid4())
    
    def create_session(self, session_id: str) -> str:
        """Create a new session directory and return its path."""
        session_dir = self._session_base_dir / session_id
        already_exists = session_dir.exists()
        session_dir.mkdir(exist_ok=True)
        
        # Only log when actually creating a new directory
        if not already_exists:
            logger.info(f"Created session directory: {session_dir}")
        
        return str(session_dir)
    
    def get_session_directory(self, session_id: str) -> Optional[str]:
        """Get the directory path for a session."""
        session_dir = self._session_base_dir / session_id
        return str(session_dir) if session_dir.exists() else None
    
    def save_frame(self, session_id: str, frame, filename: str) -> Optional[str]:
        """Save a frame to the session directory."""
        import numpy as np
        
        try:
            session_dir = self.get_session_directory(session_id)
            if not session_dir or frame is None or frame.size == 0:
                return None
            
            filepath = os.path.join(session_dir, filename)
            return filepath if (cv2.imwrite(filepath, frame) and 
                             os.path.exists(filepath) and 
                             os.path.getsize(filepath) > 0) else None
                
        except Exception as e:
            logger.error(f"Frame save error: {e}")
            return None
    
    def create_temp_file(self, session_id: str, prefix: str, suffix: str = ".tmp") -> str:
        """Create a temporary file within the session directory."""
        session_dir = self.get_session_directory(session_id)
        if not session_dir:
            raise ValueError(f"Session {session_id} not found")
        
        safe_prefix = re.sub(r'[^\w-]', '_', prefix)[:50]
        filename = f"{safe_prefix}_{int(datetime.now().timestamp())}_{str(uuid.uuid4())[:8]}{suffix}"
        return os.path.join(session_dir, filename)
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up a session and all its files."""
        try:
            session_dir = self.get_session_directory(session_id)
            if session_dir and os.path.exists(session_dir):
                shutil.rmtree(session_dir)
                logger.info(f"Removed session directory: {session_dir}")
            gc.collect()
                
        except Exception as e:
            logger.error(f"Session cleanup error for {session_id}: {e}")
            gc.collect()
    
    def cleanup_old_sessions(self) -> None:
        """Clean up sessions older than 20 minutes."""
        current_time = time.time()
        max_age_seconds = 20 * 60  # 20 minutes
        active_session_id = getattr(st.session_state, 'session_id', None) if hasattr(st, 'session_state') else None
        
        try:
            for session_dir in self._session_base_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                try:
                    if (session_dir.name == active_session_id or 
                        current_time - session_dir.stat().st_mtime <= max_age_seconds):
                        continue
                    
                    shutil.rmtree(session_dir)
                    logger.info(f"Cleaned up old session: {session_dir.name}")
                    
                except Exception as e:
                    logger.error(f"Error cleaning session {session_dir.name}: {e}")
                    
        except Exception as e:
            logger.error(f"Old sessions cleanup error: {e}")
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        try:
            return [d.name for d in self._session_base_dir.iterdir() if d.is_dir()]
        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            return []


_session_manager = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


class BaseGoogleSheetsExporter:
    """Base class for Google Sheets export operations."""
    
    def __init__(self):
        self.service = GoogleSheetsService.get_sheets_service()
    
    def _export_to_sheet(self, sheet_name: str, row_data: List[str], 
                        operation_name: str, identifier: str = "") -> bool:
        """Export data to Google Sheets."""
        if not self.service:
            logger.error(f"Google Sheets service not available for {operation_name}")
            return False
        
        try:
            config = ConfigurationManager.get_secure_config()
            spreadsheet_id = config.get("verifier_sheet_id")
                
            if not spreadsheet_id:
                logger.error(f"No spreadsheet ID configured for {operation_name}")
                return False
            
            range_name = f"{sheet_name}!A:Z"
            
            body = {
                'values': [row_data]
            }
            
            result = self.service.spreadsheets().values().append(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()
            
            if identifier:
                logger.info(f"{operation_name} exported successfully for {identifier}")
            else:
                logger.info(f"{operation_name} exported successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export {operation_name} to Google Sheets: {e}")
            return False


class GlobalSidebar:
    """Centralized sidebar rendering across all pages."""
    
    @staticmethod
    def render_sidebar():
        """Render the main sidebar with help and session information."""
        with st.sidebar:
            st.markdown("""
            <div style="text-align: center; padding: 10px; margin-bottom: 20px;">
                <h2 style="color: #1f77b4; margin: 0;">Gemini Live Video Verifier</h2>
                <p style="color: #666; margin: 5px 0 0 0; font-size: 0.9em;">Multi-Modal Analysis Tool</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            GlobalSidebar._render_sidebar_session_info()
    
    @staticmethod
    def _render_sidebar_session_info():
        """Render current session information in sidebar."""
        current_screen = st.session_state.get('current_screen', 'input')
        
        has_session_data = (
            hasattr(st.session_state, 'question_id') and 
            st.session_state.question_id and
            hasattr(st.session_state, 'alias_email') and
            st.session_state.alias_email and
            current_screen != 'input'
        )
        
        if has_session_data:
            st.markdown("### 📋 Current Session")
            
            session_info = GlobalSidebar._get_session_display_info()
            
            session_html = f"""
            <div style="background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%); padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; margin-bottom: 15px;">
                <h4 style="margin: 0 0 10px 0; color: #2e7d32;">Session ID: <code style="background: #e0e0e0; padding: 2px 4px; border-radius: 3px; font-size: 0.7em;">{session_info['session_id']}</code></h4>
                <p style="margin: 5px 0; color: #333;"><strong>Question ID:</strong><br>{session_info['question_id']}</p>
                <p style="margin: 5px 0; color: #333;"><strong>Alias Email:</strong><br>{session_info.get('email', '')}</p>"""
            
            session_html += f"""
                <p style="margin: 5px 0; color: #333;"><strong>Agent Email:</strong><br>{session_info.get('agent_email', '')}</p>
                <p style="margin: 5px 0; color: #333;"><strong>Initial Prompt:</strong><br>{session_info.get('initial_prompt', '')}</p>
                <p style="margin: 5px 0; color: #333;"><strong>Quality Comparison:</strong><br>{session_info.get('quality_comparison', '')}</p>
                <p style="margin: 5px 0; color: #333;"><strong>Language:</strong><br>{session_info['language']}</p>
                <p style="margin: 5px 0; color: #333;"><strong>Task Type:</strong><br>{session_info['task_type']}</p>"""
            
            session_html += "</div>"
            
            st.markdown(session_html, unsafe_allow_html=True)
            
            if current_screen in ['analysis', 'qa']:
                GlobalSidebar._render_sidebar_analysis_status()
    
    @staticmethod
    def _get_session_display_info():
        """Get formatted session information for display."""
        language_code = st.session_state.get('selected_language')
        language_display = Config.get_language_display_name(language_code)
        
        task_type = st.session_state.get('task_type')
        
        # Get video names from URLs
        gemini_url = st.session_state.get('gemini_video_url', '')
        gemini_name = gemini_url.split('/')[-1] if gemini_url else 'Not uploaded'
        competitor_url = st.session_state.get('competitor_video_url', '')
        competitor_name = competitor_url.split('/')[-1] if competitor_url else 'Not uploaded'
        
        session_info = {
            'question_id': st.session_state.get('question_id'),
            'email': st.session_state.get('alias_email'),
            'agent_email': st.session_state.get('agent_email', ''),
            'initial_prompt': st.session_state.get('initial_prompt', ''),
            'quality_comparison': st.session_state.get('quality_comparison', ''),
            'gemini_video': gemini_name,
            'competitor_video': competitor_name,
            'language': language_display,
            'task_type': task_type,
            'frame_interval': f"{st.session_state.get('frame_interval', Config.DEFAULT_FRAME_INTERVAL)}s",
            'session_id': st.session_state.session_id
        }
        
        # Use Gemini video validation for display
        video_validation = st.session_state.get('gemini_video_validation', {})
        if video_validation:
            session_info['video_properties'] = {
                'duration': f"{video_validation.get('duration', 0):.1f}s",
                'resolution': f"{video_validation.get('width', 0)}x{video_validation.get('height', 0)}",
                'audio_properties': video_validation.get('audio_properties', {})
            }
        
        return session_info
    
    @staticmethod
    def _render_sidebar_analysis_status():
        """Render analysis status information in sidebar, split per video (Gemini / Competitor)."""
        # Only render if we have analysis results
        gemini_results = st.session_state.get('gemini_analysis_results', []) or []
        competitor_results = st.session_state.get('competitor_analysis_results', []) or []
        
        # Don't render if no results exist
        if not gemini_results and not competitor_results:
            return
        
        # Render per-video analysis if available
        def _render_video_section(title: str, title_color: str, results: List[object], qa_checker: Optional[object] = None):
            text_results = [r for r in results if 'Text Detection' in r.rule_name]
            language_results = [r for r in results if 'Language Detection' in r.rule_name]
            voice_results = [r for r in results if 'Voice Audibility' in r.rule_name]
            noise_results = [r for r in results if 'Background Noise' in r.rule_name]

            analysis_details = []

            flash_results = [r for r in text_results if '2.5 Flash' in r.rule_name]
            alias_results = [r for r in text_results if 'Alias Name' in r.rule_name]
            eval_results = [r for r in text_results if 'Eval Mode' in r.rule_name]

            if flash_results:
                flash_detected = any(r.detected for r in flash_results)
                flash_qa_info = None
                try:
                    flash_qa_info = GlobalSidebar._get_qa_info_for_rule_type('flash_presence')
                except Exception:
                    pass
                flash_status = "✅" if flash_qa_info and flash_qa_info.get('passed', False) else "❌"
                status_text = "Detected" if flash_detected else "Not Found"
                analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>2.5 Flash:</strong><br>{flash_status} {status_text}</p>")

            if alias_results:
                alias_detected = any(r.detected for r in alias_results)
                alias_qa_info = None
                try:
                    alias_qa_info = GlobalSidebar._get_qa_info_for_rule_type('alias_name_presence')
                except Exception:
                    pass
                alias_status = "✅" if alias_qa_info and alias_qa_info.get('passed', False) else "❌"
                analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Alias Name:</strong><br>{alias_status} {'Detected' if alias_detected else 'Not Found'}</p>")

            if eval_results:
                eval_detected = any(r.detected for r in eval_results)
                eval_qa_info = None
                try:
                    eval_qa_info = GlobalSidebar._get_qa_info_for_rule_type('eval_mode_presence')
                except Exception:
                    pass
                eval_status = "✅" if eval_qa_info and eval_qa_info.get('passed', False) else "❌"
                analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Eval Mode:</strong><br>{eval_status} {'Detected' if eval_detected else 'Not Found'}</p>")

            if language_results:
                language_result = language_results[0]
                language_qa_info = None
                try:
                    language_qa_info = GlobalSidebar._get_qa_info_for_rule_type('language_fluency')
                except Exception:
                    pass
                language_status = "✅" if language_qa_info and language_qa_info.get('passed', False) else "❌"
                if language_result.details and 'analysis_failed_reason' in language_result.details:
                    analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Language Fluency:</strong><br>{language_status} No Voices Detected</p>")
                else:
                    detected_lang = language_result.details.get('detected_language', 'unknown') if language_result.details else 'unknown'
                    target_lang = language_result.details.get('target_language', 'unknown') if language_result.details else 'unknown'
                    if detected_lang != 'unknown':
                        locale_format = Config.whisper_language_to_locale(detected_lang, target_lang)
                        display_name = Config.get_language_display_name(locale_format)
                        analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Language Fluency:</strong><br>{language_status} {display_name}</p>")
                    else:
                        analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Language Fluency:</strong><br>{language_status} Unknown</p>")

            if voice_results:
                voice_result = voice_results[0]
                voice_qa_info = None
                try:
                    voice_qa_info = GlobalSidebar._get_qa_info_for_rule_type('voice_audibility')
                except Exception:
                    pass
                voice_status = "✅" if voice_qa_info and voice_qa_info.get('passed', False) else "❌"
                if voice_result.details:
                    num_voices = voice_result.details.get('num_audible_voices', 0)
                    analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Voice Audibility:</strong><br>{voice_status} {num_voices} Voice{'s' if num_voices != 1 else ''}</p>")
                else:
                    analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Voice Audibility:</strong><br>{voice_status} Unknown</p>")

            noise_qa_info = None
            try:
                noise_qa_info = GlobalSidebar._get_qa_info_for_rule_type('background_noise') if st.session_state.get('qa_checker') else None
            except Exception:
                noise_qa_info = None
            if noise_qa_info or noise_results:
                noise_status = "✅" if noise_qa_info and noise_qa_info.get('passed', False) else "❌"
                level_source = None
                if noise_qa_info and isinstance(noise_qa_info.get('noise_level'), str):
                    level_source = noise_qa_info.get('noise_level')
                elif noise_results:
                    noise_result = noise_results[0]
                    if noise_result.details and isinstance(noise_result.details.get('noise_level'), str):
                        level_source = noise_result.details.get('noise_level')
                if isinstance(level_source, str) and level_source.strip():
                    noise_level = level_source.strip().split()[0].capitalize()
                else:
                    noise_level = "Unknown"
                analysis_details.append(f"<p style=\"margin: 5px 0; color: #333;\"><strong>Background Noise:</strong><br>{noise_status} {noise_level}</p>")

            details_html = "".join(analysis_details) if analysis_details else "<p style=\"margin: 5px 0; color: #333;\">No detailed analysis available</p>"
            return f"<h4 style=\"margin: 8px 0; color: {title_color}; font-weight: bold;\">{title}</h4>" + details_html

        gemini_html = _render_video_section('🔵 Gemini', '#1565C0', gemini_results)
        competitor_html = _render_video_section('🔴 Competitor', '#c62828', competitor_results)

        total_analysis_time = GlobalSidebar._get_total_analysis_time_for_sidebar()
        time_html = f"<p style=\"margin: 5px 0; color: #333;\"><strong>Analysis Time:</strong><br>⏱️ {total_analysis_time:.2f} seconds</p>" if total_analysis_time > 0 else ""

        st.markdown("### 📊 Analysis Report")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f0f8ff 100%); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            {gemini_html}
            <hr style="border: 1px solid #ccc; margin: 10px 0;"/>
            {competitor_html}
            {time_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def _get_qa_info_for_rule_type(rule_type: str):
        """Get QA information for a specific rule type."""
        qa_results = st.session_state.qa_checker.get_detailed_results()
        return qa_results.get(rule_type)

    @staticmethod
    def _get_total_analysis_time_for_sidebar() -> float:
        """Get the total analysis time from the analyzer instance for sidebar display."""
        analyzer = st.session_state.get('analyzer_instance')
        return analyzer.performance_metrics.get('total_analysis_time', 0.0) if analyzer else 0.0