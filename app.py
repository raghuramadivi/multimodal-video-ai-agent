import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
import google.generativeai as genai
from google.generativeai import upload_file, get_file

import time
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ── API Setup ──────────────────────────────────────
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# ── Page Config ────────────────────────────────────
st.set_page_config(
    page_title="Multimodal AI Agent - Video Summarizer",
    page_icon="🎥",
    layout="wide"
)
st.title("Phidata Video AI Summarizer Agent 🎥")
st.header("Powered by gemini-3.1-pro-preview")

# ── Agent Init ─────────────────────────────────────
@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-3.1-pro-preview"),  # ← KEEP THIS MODEL
        tools=[DuckDuckGo()],
        markdown=True,
    )

multimodal_Agent = initialize_agent()

# ── File Uploader ──────────────────────────────────
video_file = st.file_uploader(
    "Upload a video file",
    type=['mp4', 'mov', 'avi'],
    help="Upload a video for AI analysis (max 200MB)"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content...",
        help="Provide specific questions about the video."
    )

    if st.button("🔍 Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):

                    # ── Upload video to Google ──────────────────
                    processed_video = upload_file(video_path)

                    # ── Poll until ACTIVE ───────────────────────
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    # ✅ Confirm it's ready
                    if processed_video.state.name != "ACTIVE":
                        st.error(f"Video processing failed: {processed_video.state.name}")
                        st.stop()

                    # ── Build Prompt ────────────────────────────
                    analysis_prompt = f"""
                    You are analyzing an uploaded video.
                    The video has been successfully uploaded and is ready for analysis.
                    
                    Please analyze the video content carefully and answer:
                    {user_query}
                    
                    Provide a detailed, user-friendly and actionable response
                    based on what you observe in the video.
                    Also use web search for any additional context if needed.
                    """

                    # ── Run Agent ───────────────────────────────
                    response = multimodal_Agent.run(
                        analysis_prompt,
                        videos=[processed_video]
                    )

                # ── Show Result ─────────────────────────────────
                st.subheader("Analysis Result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred: {error}")
                st.info("💡 Tip: Make sure your video is under 200MB and in mp4/mov/avi format.")

            finally:
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("⬆️ Upload a video file to begin analysis.")