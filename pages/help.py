"""
Help page for the Gemini Live Video Verifier Tool.

This page provides comprehensive guidance on how to use the video analysis tool,
including step-by-step instructions, tips for best results, and troubleshooting information.
"""

import streamlit as st
from shared_components import GlobalSidebar

def main():
    """Main function for the Help page."""
    GlobalSidebar.render_sidebar()
    
    st.title("❓ Help & User Guide")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Getting Started", 
        "🔍 Analysis Details", 
        "📊 Understanding Results", 
        "💡 Tips & Troubleshooting"
    ])
    
    with tab1:
        render_getting_started()
    
    with tab2:
        render_analysis_details()
    
    with tab3:
        render_understanding_results()
    
    with tab4:
        render_tips_and_troubleshooting()

def render_getting_started():
    """Render the Getting Started section."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 How to Use This Tool
        
        Follow these three simple steps to analyze and submit your video:
        """)
        
        with st.container():
            st.markdown("### 1️⃣ Input Parameters")
            st.info("""
            **Required Information:**
            - **Question ID**: Enter a unique identifier for this analysis session
                - Target language and task type will be automatically inferred from the Question ID
                - Must be authorized in the Question IDs sheet
            - **Alias Email Address**: Provide the email associated with your Question ID
                - Must be authorized in the Alias Emails sheet
            - **Video File**: Upload your MP4 video file
                - Supported format: MP4 only
                - File size: Up to 200mb
                - Duration: Minimum 30 seconds, maximum 10 minutes
                - Resolution: Portrait mobile format required
            """)
        
        with st.container():
            st.markdown("### 2️⃣ Video Analysis")
            st.success("""
            **Automatic Processing:**
            - The system will process your video automatically
            - Real-time progress will be displayed
            - **Text Detection**: Looks for specific content using OCR techniques:
                - "2.5 Flash" (to verify correct model usage)
                - "Roaring Tiger" (to confirm correct alias usage)
                - "Eval Mode: Native Audio Output" (to ensure proper eval mode)
            - **Audio Analysis**: Checks language fluency and voice audibility
                - Ensures both user and model voices are clearly heard
                - Verifies spoken language matches expected language
            - **Detailed Results**: Review analysis with screenshots and audio summaries
            - **Quality Feedback**: If issues are detected, detailed feedback helps you understand what needs to be fixed
            """)
        
        with st.container():
            st.markdown("### 3️⃣ Submit Video")
            st.warning("""
            **Final Submission:**
            - Once all quality checks pass, you can submit your video
            - Complete the submission process through the tool
            - If quality checks fail, you'll need to re-record and upload a new video
            """)
    
    with col2:
        st.markdown("### 🎯 Quick Checklist")
        checklist_items = [
            "✅ Question ID is authorized",
            "✅ Alias email is authorized", 
            "✅ Video is MP4 format",
            "✅ Video is portrait orientation",
            "✅ Video duration ≥ 30 seconds",
            "✅ Video shows '2.5 Flash' text",
            "✅ Video shows 'Roaring Tiger' alias",
            "✅ Video shows 'Eval Mode' text",
            "✅ Both voices are audible",
            "✅ Language matches expected"
        ]
        
        for item in checklist_items:
            st.markdown(item)
        
        st.markdown("---")
        st.info("💡 **Tip**: Make sure all checklist items are satisfied before uploading your video to avoid quality check failures.")

def render_analysis_details():
    """Render the Analysis Details section."""
    st.header("🔍 What Gets Analyzed")
    
    st.subheader("📝 Text Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 2.5 Flash Detection
        - **Purpose**: Verify correct model usage
        - **Target**: Exact text "2.5 Flash"
        - **Technology**: OCR (Optical Character Recognition)
        - **Common Issues**: Using "2.5 Pro" instead of "2.5 Flash", "2.5 Flash not shown at the start of the video"
        """)
        
        st.markdown("""
        #### Eval Mode Detection  
        - **Purpose**: Ensure proper evaluation mode
        - **Target**: "Eval Mode: Native Audio Output"
        - **Importance**: Confirms correct system configuration
        """)
    
    with col2:
        st.markdown("""
        #### Alias Name Detection
        - **Purpose**: Confirm correct alias usage
        - **Target**: "Roaring Tiger" text
        - **Variations**: Handles common OCR errors and spacing
        """)
    
    st.markdown("---")
    
    st.subheader("🎧 Audio Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Language Detection & Fluency
        - **Purpose**: Verify spoken language matches expected language
        - **Task Types Supported**:
            - **Monolingual**: Single language detection
            - **Code Mixed and Language Learning**: Multiple languages (target + English)
        - **Scoring**: Fluency assessment based on language consistency
        """)
    
    with col2:
        st.markdown("""
        #### Voice Audibility Analysis
        - **Purpose**: Ensure both user and AI voices are clearly audible
        - **Checks**:
            - Voice activity detection
            - Number of distinct speakers
            - Audio quality assessment
            - Background noise levels
        - **Requirements**: Both voices must be clearly distinguishable
        """)

def render_understanding_results():
    """Render the Understanding Results section."""
    st.header("📊 Understanding Your Results")
    
    st.subheader("🎯 Quality Assurance Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        #### ✅ PASS Status
        **What it means:**
        - All quality checks were successful
        - Your video meets all requirements
        - You can proceed to submit your video
        - No re-recording needed
        
        **Next Steps:**
        - Click "Submit Video" button
        - Complete submission process
        """)
    
    with col2:
        st.error("""
        #### ❌ FAIL Status  
        **What it means:**
        - One or more quality checks failed
        - Your video doesn't meet requirements
        - Submission is not allowed
        - Re-recording is necessary
        
        **Next Steps:**
        - Review detailed feedback
        - Address identified issues
        - Re-record and upload new video
        """)
    
    st.markdown("---")
    st.subheader("📋 Reading Detailed Results")
    
    st.markdown("""
    #### Text Detection Results
    - **Positive Detection**: Shows screenshot with detected text highlighted
    - **Timestamp**: Exact moment in video where text was found
    - **Screenshots**: Visual proof of detected elements
    """)
    
    st.markdown("""
    #### Audio Analysis Results  
    - **Language Detected**: Shows identified spoken language
    - **Transcription**: Full text of what was spoken
    - **Voice Count**: Number of distinct speakers detected
    """)
    
    st.markdown("---")
    st.subheader("⚠️ Common Issues & Solutions")
    
    issues_solutions = [
        {
            "issue": "'2.5 Pro' detected instead of '2.5 Flash'",
            "solution": "Change model setting to '2.5 Flash' before recording",
            "critical": True
        },
        {
            "issue": "'Roaring Tiger' alias not found",
            "solution": "Ensure alias is clearly visible and properly spelled",
            "critical": True
        },
        {
            "issue": "Wrong language detected",
            "solution": "Speak clearly in the expected language throughout video",
            "critical": True
        },
        {
            "issue": "Only one voice audible",
            "solution": "Ensure both user and AI responses are clearly recorded",
            "critical": True
        },
        {
            "issue": "Low text detection confidence",
            "solution": "Improve lighting and ensure text is clearly visible",
            "critical": False
        }
    ]
    
    for item in issues_solutions:
        icon = "🚨" if item["critical"] else "⚠️"
        with st.expander(f"{icon} {item['issue']}", expanded=False):
            st.write(f"**Solution**: {item['solution']}")
            if item["critical"]:
                st.error("**Impact**: This issue prevents video submission and requires re-recording.")
            else:
                st.warning("**Impact**: This may affect analysis accuracy but doesn't prevent submission.")

def render_tips_and_troubleshooting():
    """Render the Tips and Troubleshooting section."""
    st.header("💡 Tips for Best Results")
    
    st.subheader("🎬 Video Recording Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📱 Technical Requirements
        - **Format**: Use MP4 format only
        - **Orientation**: Portrait mode (vertical)
        - **Duration**: 30 seconds minimum, 10 minutes maximum
        - **Quality**: Use highest quality settings available
        - **Audio**: Enable high-quality audio recording
        """)
        
        st.markdown("""
        #### 🎯 Model & Alias Guidelines
        - **Always use '2.5 Flash'** - never '2.5 Pro'
        - **Make sure your alias is shown before recording**
        - **Enable 'Eval Mode: Native Audio Output'**
        - **Verify settings** before starting conversation
        - **Keep settings visible** throughout recording
        """)
    
    with col2:
        st.markdown("""
        #### 💡 Environment Setup
        - **Good Lighting**: Ensure text is clearly readable
        - **Quiet Space**: Minimize background noise
        - **Clear Speech**: Speak clearly and at normal pace
        - **Full Conversation**: Record complete interactions
        """)
        
        st.markdown("""
        #### 🗣️ Language Guidelines  
        - **Speak in expected language** (inferred from Question ID)
        - **Maintain consistency** throughout video
        - **Clear pronunciation** for accurate detection
        """)
    
    st.markdown("---")
    st.subheader("🔧 Troubleshooting")
    
    troubleshooting_tabs = st.tabs([
        "🚫 Upload Issues", 
        "⚡ Analysis Failures", 
        "📊 QA Failures", 
        "🔗 Submission Problems"
    ])
    
    with troubleshooting_tabs[0]:
        st.markdown("""
        #### 🚫 Video Upload Issues
        
        **File too large:**
        - Compress video while maintaining quality
        - Maximum size: 200MB
        
        **Wrong format:**
        - Convert to MP4 format
        - Use standard video compression
        
        **Invalid resolution:**
        - Must be portrait orientation (taller than wide)
        - Standard mobile phone ratios supported
        
        **Duration issues:**
        - Minimum: 30 seconds
        - Maximum: 10 minutes
        - Trim video if necessary
        """)
    
    with troubleshooting_tabs[1]:
        st.markdown("""
        #### ⚡ Analysis Process Failures
        
        **Analysis stuck or fails:**
        - Refresh the page and try again
        - Check internet connection
        - Ensure video file isn't corrupted
        
        **OCR text detection fails:**
        - Improve video lighting
        - Ensure text is clearly visible
        - Check for proper model/alias settings
        
        **Audio analysis fails:**
        - Verify audio is not muted
        - Check for background noise
        - Ensure both voices are audible
        """)
    
    with troubleshooting_tabs[2]:
        st.markdown("""
        #### 📊 Quality Assurance Failures
        
        **Text not detected:**
        - Verify correct model usage (2.5 Flash)
        - Check alias is set to 'Roaring Tiger'
        - Ensure 'Eval Mode' is visible
        - Improve lighting and visibility
        
        **Wrong language detected:**
        - Speak clearly in expected language
        - Reduce background noise
        - Ensure consistent language use
        
        **Voice audibility issues:**
        - Record in quiet environment
        - Ensure both user and AI are audible
        - Check device microphone quality
        """)
    
    with troubleshooting_tabs[3]:
        st.markdown("""
        #### 🔗 Google Drive Submission Problems
        
        **Drive folder not generated:**
        - Check internet connection
        - Try refreshing and generating again
        - Contact support if persistent
        
        **Cannot access Drive folder:**
        - Ensure you're signed into Google account
        - Check folder permissions
        - Try opening in incognito mode
        
        **Upload to Drive fails:**
        - Check file size limits
        - Verify stable internet connection
        - Try uploading from different browser
        """)

if __name__ == "__main__":
    main()