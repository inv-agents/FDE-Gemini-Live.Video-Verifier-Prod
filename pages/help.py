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
            - **Initial Prompt**: Enter the initial prompt or instruction used to start both conversations
                - This prompt should be the same across both Gemini and Competitor interactions
                - Required for quality comparison and tracking
            - **Agent Email**: Provide the email address of the agent creating this submission
                - Used for submission tracking and attribution
                - Required for final submission
            - **Quality Comparison**: Rate Gemini's performance compared to the Competitor
                - Select from options ranging from "much better" to "much worse"
                - This assessment helps track relative quality
            - **Gemini Video File**: Upload your Gemini MP4 video file
                - Supported format: MP4 only
                - File size: Up to 200mb
                - Duration: Minimum 30 seconds, maximum 10 minutes
                - Resolution: Portrait mobile format required
                - **Must show "Eval Mode: Native Audio Output"** text
            - **Competitor Video File**: Upload your Competitor MP4 video file
                - Same requirements as Gemini video
                - Must use the same language and conversation flow
                - **May show different Eval Mode** (e.g., "Eval Mode: Server Text-to-Speech")
            """)
        
        with st.container():
            st.markdown("### 2️⃣ Video Analysis")
            st.success("""
            **Automatic Processing:**
            - The system will process **both videos** automatically
            - Real-time progress will be displayed for each video
            - **Text Detection** (Gemini video only): Looks for specific content using OCR techniques:
                - "2.5 Flash" (to verify correct model usage)
                - "Roaring Tiger" (to confirm correct alias usage)
                - "Eval Mode: Native Audio Output" (to ensure proper eval mode for Gemini)
                - Note: Competitor video may have a different Eval Mode (e.g., "Server Text-to-Speech")
            - **Audio Analysis** (both videos): Checks language fluency and voice audibility
                - Ensures both user and model voices are clearly heard
                - Verifies spoken language matches expected language
            - **Detailed Results**: Review analysis with screenshots and audio summaries for both Gemini and Competitor
            - **Quality Feedback**: If issues are detected, detailed feedback helps you understand what needs to be fixed
            """)
        
        with st.container():
            st.markdown("### 3️⃣ Submit Videos")
            st.warning("""
            **Final Submission:**
            - Once all quality checks pass for **both videos**, you can submit them
            - Separate Google Drive folders will be generated for Gemini and Competitor videos
            - Complete the submission process through the tool
            - If quality checks fail for either video, you'll need to re-record and upload new videos
            """)
    
    with col2:
        st.markdown("### 🎯 Quick Checklist")
        checklist_items = [
            "✅ Question ID is authorized",
            "✅ Alias email is authorized",
            "✅ Initial prompt is provided",
            "✅ Agent email is provided",
            "✅ Quality comparison is selected",
            "✅ Both videos are MP4 format",
            "✅ Both videos are portrait orientation",
            "✅ Both videos duration ≥ 30 seconds",
            "✅ Gemini video shows '2.5 Flash' text",
            "✅ Gemini video shows 'Roaring Tiger' alias",
            "✅ Gemini video shows 'Eval Mode: Native Audio Output'",
            "✅ Competitor video shows appropriate Eval Mode",
            "✅ Both voices are audible (in both videos)",
            "✅ Language matches expected (in both videos)"
        ]
        
        for item in checklist_items:
            st.markdown(item)
        
        st.markdown("---")
        st.info("💡 **Tip**: Make sure all checklist items are satisfied before uploading your videos to avoid quality check failures. Both videos should follow the same conversation flow.")

def render_analysis_details():
    """Render the Analysis Details section."""
    st.header("🔍 What Gets Analyzed")
    
    st.subheader("📝 Text Detection (Gemini Video Only)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 2.5 Flash Detection
        - **Purpose**: Verify correct model usage in Gemini video
        - **Target**: Exact text "2.5 Flash"
        - **Technology**: OCR (Optical Character Recognition)
        - **Common Issues**: Using "2.5 Pro" instead of "2.5 Flash", "2.5 Flash not shown at the start of the video"
        - **Note**: Only checked in Gemini video, not in Competitor
        """)
        
        st.markdown("""
        #### Eval Mode Detection  
        - **Purpose**: Ensure proper evaluation mode in Gemini video
        - **Target**: "Eval Mode: Native Audio Output"
        - **Importance**: Confirms correct system configuration for Gemini
        - **Note**: Only checked in Gemini video; Competitor may use different Eval Mode (e.g., "Server Text-to-Speech")
        """)
    
    with col2:
        st.markdown("""
        #### Alias Name Detection
        - **Purpose**: Confirm correct alias usage in Gemini video
        - **Target**: "Roaring Tiger" text
        - **Variations**: Handles common OCR errors and spacing
        - **Note**: Only checked in Gemini video, not in Competitor
        """)
    
    st.markdown("---")
    
    st.subheader("🎧 Audio Analysis (Both Videos)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Language Detection & Fluency
        - **Purpose**: Verify spoken language matches expected language in **both videos**
        - **Task Types Supported**:
            - **Monolingual**: Single language detection
            - **Code Mixed and Language Learning**: Multiple languages (target + English)
        - **Scoring**: Fluency assessment based on language consistency
        - **Note**: Analyzed independently for Gemini and Competitor videos
        """)
    
    with col2:
        st.markdown("""
        #### Voice Audibility Analysis
        - **Purpose**: Ensure both user and AI voices are clearly audible in **both videos**
        - **Checks**:
            - Voice activity detection
            - Number of distinct speakers
            - Audio quality assessment
            - Background noise levels
        - **Requirements**: Both voices must be clearly distinguishable in each video
        - **Note**: Analyzed independently for Gemini and Competitor videos
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
        - All quality checks were successful for **both videos**
        - Your videos meet all requirements
        - You can proceed to submit your videos
        - No re-recording needed
        
        **Next Steps:**
        - Click "Submit Videos" button
        - Upload to separate Google Drive folders
        - Complete submission process
        """)
    
    with col2:
        st.error("""
        #### ❌ FAIL Status  
        **What it means:**
        - One or more quality checks failed in either video
        - Your videos don't meet requirements
        - Submission is not allowed
        - Re-recording is necessary for failed video(s)
        
        **Next Steps:**
        - Review detailed feedback for each video
        - Address identified issues
        - Re-record and upload new videos
        """)
    
    st.markdown("---")
    st.subheader("📋 Reading Detailed Results")
    
    st.markdown("""
    #### Text Detection Results (Gemini Video)
    - **Positive Detection**: Shows screenshot with detected text highlighted
    - **Timestamp**: Exact moment in video where text was found
    - **Screenshots**: Visual proof of detected elements
    - **Note**: Text detection only applies to Gemini video
    """)
    
    st.markdown("""
    #### Audio Analysis Results (Both Videos)
    - **Language Detected**: Shows identified spoken language for each video
    - **Transcription**: Full text of what was spoken in each video
    - **Voice Count**: Number of distinct speakers detected in each video
    - **Comparison**: Review results side-by-side for Gemini vs Competitor
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
            "issue": "Only one voice audible (in either video)",
            "solution": "Ensure both user and AI responses are clearly recorded in both videos",
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
        - **Good Lighting**: Ensure text is clearly readable (especially in Gemini video)
        - **Quiet Space**: Minimize background noise in both recordings
        - **Clear Speech**: Speak clearly and at normal pace in both videos
        - **Full Conversation**: Record complete interactions for both
        - **Consistency**: Use same conversation flow in both videos
        """)
        
        st.markdown("""
        #### 🗣️ Language Guidelines  
        - **Speak in expected language** (inferred from Question ID) in **both videos**
        - **Maintain consistency** throughout both videos
        - **Clear pronunciation** for accurate detection in both
        - **Same conversation**: Follow the same flow in Gemini and Competitor videos
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
        - Maximum size: 200MB per video
        - Both Gemini and Competitor videos must meet size limit
        
        **Wrong format:**
        - Convert both videos to MP4 format
        - Use standard video compression
        
        **Invalid resolution:**
        - Both videos must be portrait orientation (taller than wide)
        - Standard mobile phone ratios supported
        
        **Duration issues:**
        - Minimum: 30 seconds per video
        - Maximum: 10 minutes per video
        - Trim videos if necessary
        - Both videos should have similar durations
        """)
    
    with troubleshooting_tabs[1]:
        st.markdown("""
        #### ⚡ Analysis Process Failures
        
        **Analysis stuck or fails:**
        - Refresh the page and try again
        - Check internet connection
        - Ensure both video files aren't corrupted
        - Analysis processes videos independently
        
        **OCR text detection fails (Gemini video):**
        - Improve video lighting in Gemini video
        - Ensure text is clearly visible
        - Check for proper model/alias settings
        - Note: Text detection only applies to Gemini video
        
        **Audio analysis fails (either video):**
        - Verify audio is not muted in both videos
        - Check for background noise in both
        - Ensure both voices are audible in each video
        """)
    
    with troubleshooting_tabs[2]:
        st.markdown("""
        #### 📊 Quality Assurance Failures
        
        **Text not detected (Gemini video):**
        - Verify correct model usage (2.5 Flash)
        - Check alias is set to 'Roaring Tiger'
        - Ensure 'Eval Mode' is visible
        - Improve lighting and visibility
        - Note: Only checked in Gemini video
        
        **Wrong language detected (either video):**
        - Speak clearly in expected language in both videos
        - Reduce background noise in both videos
        - Ensure consistent language use across both
        
        **Voice audibility issues (either video):**
        - Record both videos in quiet environment
        - Ensure both user and AI are audible in each video
        - Check device microphone quality
        - Both videos must pass voice audibility checks
        """)
    
    with troubleshooting_tabs[3]:
        st.markdown("""
        #### 🔗 Google Drive Submission Problems
        
        **Drive folders not generated:**
        - Check internet connection
        - Try refreshing and generating again
        - Both Gemini and Competitor folders should be created
        - Contact support if persistent
        
        **Cannot access Drive folders:**
        - Ensure you're signed into Google account
        - Check folder permissions for both folders
        - Try opening in incognito mode
        - Verify access to both Gemini and Competitor folders
        
        **Upload to Drive fails:**
        - Check file size limits for both videos
        - Verify stable internet connection
        - Try uploading from different browser
        - Upload videos to their respective folders (Gemini to Gemini folder, Competitor to Competitor folder)
        """)

if __name__ == "__main__":
    main()