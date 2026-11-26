# Project is ready to deploy
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from typing import Optional, List, Dict, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
from PIL import Image
import av
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
)

# Custom modules
from pose_estimator import PoseEstimator
from boxing_analyzer import BoxingAnalyzer
from reference_poses import ReferenceBoxingPoses
from video_processor import VideoProcessor
from utils import (
    PerformanceMonitor,
    format_feedback_message,
    create_accuracy_visualization,
    display_progress_with_eta,
    validate_boxing_content
)

# Set page configuration
st.set_page_config(
    page_title="Boxing Technique Analyzer",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    "Initialize all variables"
    if 'pose_estimator' not in st.session_state:
        st.session_state.pose_estimator = PoseEstimator()
    if 'boxing_analyzer' not in st.session_state:
        st.session_state.boxing_analyzer = BoxingAnalyzer()
    if 'reference_poses' not in st.session_state:
        st.session_state.reference_poses = ReferenceBoxingPoses()
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = VideoProcessor()
    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()

def main():
    init_session_state()

    st.title("🥊 Boxing Technique Analysis System")
    st.markdown("Compare your boxing technique against reference videos or analyze against profession forms")

    # Sidebar configuration
    st.sidebar.title("⚙️ Analysis Settings")

    # Analysis mode selection
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Side-by-Side Comparison", "User Form Analysis", "Live Camera Analysis"]
    )

    # Frame rate settings
    frame_rate = st.sidebar.slider(
        "Processing Frame Rate (FPS)",
        min_value=5,
        max_value=15,
        value=8,
        help="Lower frame rates process faster but may miss details"
    )

    # Maximum frames to process
    max_frames = st.sidebar.slider(
        "Maximum Frames to Process",
        min_value=50,
        max_value=300,
        value=150,
        help="Limit processing to prevent memory issues"
    )

    if analysis_mode == "Side-by-Side Comparison":
        video_comparison_interface(frame_rate, max_frames)
    elif analysis_mode == "User Form Analysis":
        technique_analysis_interface(frame_rate, max_frames)
    else:
        realtime_camera_interface(frame_rate)

def video_comparison_interface(frame_rate: int, max_frames: int):
    # Interface for comparing two videos side by side
    st.header("📹 Side-by-Side Video Comparison")
    st.markdown("Upload two videos to compare boxing techniques frame by frame")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Reference Video")
        ref_video = st.file_uploader(
            "Upload reference video",
            type=['mp4', 'mov', 'avi'],
            key="ref_video"
        )

        if ref_video:
            st.video(ref_video)

    with col2:
        st.subheader("👤 User Video")
        user_video = st.file_uploader(
            "Upload user video",
            type=['mp4', 'mov' , 'avi'],
            key='user_video'
        )

        if user_video:
            st.video(user_video)
    
    if ref_video and user_video:
        if st.button("🔍 Start Analysis", type="primary", use_container_width=True):
            analyze_video_comparison(ref_video, user_video, frame_rate, max_frames)

    # Always show the latest comparison if available (no re-processing)
    if "comparison_result" in st.session_state:
        display_comparison_results(
            st.session_state.comparison_result,
            st.session_state.ref_frames,
            st.session_state.user_frames,
            st.session_state.ref_poses,
            st.session_state.user_poses,
        )

def technique_analysis_interface(frame_rate: int, max_frames: int):
    st.header("🎯 Technique Analysis")
    st.markdown("Analyze your boxing technique against professional reference forms")

    # Technique selection
    techniques = st.session_state.reference_poses.get_all_techniques()
    selected_technique = st.selectbox(
        "Select Boxing Technique",
        techniques,
        format_func=lambda technique: technique.title()
    )

    # Display technique info
    if selected_technique:
        technique_info = st.session_state.reference_poses.get_reference_pose(selected_technique)
        st.info(f"**{selected_technique.title()}**: {technique_info['description']}")

        with st.expander("💡 Technique Tips"):
            for tip in technique_info['tips']:
                st.write(f"• {tip}")
    
    # Video upload
    user_video = st.file_uploader(
        "Upload your boxing video",
        type=['mp4', 'mov', 'avi'],
        key="technique_video"
    )

    if user_video:
        st.video(user_video)

        if st.button("🔍 Analysis Technique", type="primary", use_container_width=True):
            analyze_technique(user_video, selected_technique, frame_rate, max_frames)

def realtime_camera_interface(frame_rate: int):
    st.header("📹 Real-time Camera Analysis")
    st.markdown("Use your browser camera to get live feedback on your boxing technique.")

    techniques = st.session_state.reference_poses.get_all_techniques()
    selected_technique = st.selectbox(
        "Select Boxing Technique",
        techniques,
        format_func=lambda technique: technique.title(),
        key="realtime_technique"
    )

    if selected_technique:
        technique_info = st.session_state.reference_poses.get_reference_pose(selected_technique)
        st.info(f"**{selected_technique.title()}**: {technique_info['description']}")

        with st.expander("💡 Technique Tips"):
            for tip in technique_info['tips']:
                st.write(f"• {tip}")

    run_realtime_analysis(selected_technique, frame_rate)

def analyze_video_comparison(ref_video, user_video, frame_rate: int, max_frames: int):
    # Analyze and compare two videos
    st.session_state.performance_monitor.start_timer("video_comparison")

    # Create progress containers
    progress_container = st.container()
    results_container = st.container()

    with progress_container:
        st.subheader("🔄️ Processing Videos...")

        # Validate reference video
        st.info("Validating reference video for boxing content...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(ref_video.getvalue())
            ref_video_path = tmp_file.name
        
        try:
            is_boxing, confidence, reason = validate_boxing_content(
                ref_video_path, st.session_state.pose_estimator
            )
            if not is_boxing:
                st.error(f"❌ Reference Video Validation Failed: {reason}")
                st.warning(f"Confidence Score: {confidence:.1f}%")
                st.info("💡 Please upload a video that contains boxing techniques.")
                os.unlink(ref_video_path)
                return
            else:
                st.success(f"✅ Reference video validated (Confidence: {confidence:.1f}%)")
        finally:
            if os.path.exists(ref_video_path):
                os.unlink(ref_video_path)

        # Validate user video
        st.info("Validating user video for boxing content...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(user_video.getvalue())
            user_video_path = tmp_file.name
        
        try:
            is_boxing, confidence, reason = validate_boxing_content(
                user_video_path, st.session_state.pose_estimator
            )
            if not is_boxing:
                st.error(f"❌ User Video Validation Failed: {reason}")
                st.warning(f"Confidence Score: {confidence:.1f}%")
                st.info("💡 Please upload a video that contains boxing techniques.")
                os.unlink(user_video_path)
                return
            else:
                st.success(f"✅ User video validated (Confidence: {confidence:.1f}%)")
        finally:
            if os.path.exists(user_video_path):
                os.unlink(user_video_path)

        # Process reference video
        st.info("Processing reference video...")
        ref_poses, ref_frames = process_video(ref_video, frame_rate, max_frames, "reference")

        if not ref_poses:
            st.error("Failed to process reference video. Please try a different video.")
            return
        
        # Process user video
        st.info("Processing user video...")
        user_poses, user_frames = process_video(user_video, frame_rate, max_frames, "user")

        if not user_poses:
            st.error("Failed to process user video. Please try a different video.")
            return
        
        st.success("✅ Video Processing completed!")

        # Perform comparison
        st.info("Comparing poses...")
        comparison_result = st.session_state.boxing_analyzer.compare_videos(
            ref_poses, user_poses, ref_frames, user_frames
        )

    progress_container.empty()

    # Store results in session state so we can reuse them without reprocessing
    st.session_state.comparison_result = comparison_result
    st.session_state.ref_frames = ref_frames
    st.session_state.user_frames = user_frames
    st.session_state.ref_poses = ref_poses
    st.session_state.user_poses = user_poses

    processing_time = st.session_state.performance_monitor.end_timer("video_comparison")
    st.sidebar.success(f"⏱️ Processing completed in {processing_time:.2f}s")

    # processing_time = st.session_state.performance_monitor.end_timer("video_comparison")
    # st.sidebar.success(f"⏱️ Processing completed in {processing_time:.2f}s")

def analyze_technique(user_video, technique: str, frame_rate: int, max_frames: int):
    # Analyze user technique against reference.
    st.session_state.performance_monitor.start_timer("technique_analysis")

    progress_container = st.container()
    results_container = st.container()

    with progress_container:
        st.subheader("🔄️ Analyzing Technique...")

        # Validate user video for boxing content
        st.info("Validating video for boxing content...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(user_video.getvalue())
            user_video_path = tmp_file.name
        
        try:
            is_boxing, confidence, reason = validate_boxing_content(
                user_video_path, st.session_state.pose_estimator
            )
            if not is_boxing:
                st.error(f"❌ Video Validation Failed: {reason}")
                st.warning(f"Confidence Score: {confidence:.1f}%")
                st.info("💡 Please upload a video that contains boxing techniques.")
                os.unlink(user_video_path)
                return
            else:
                st.success(f"✅ Video validated (Confidence: {confidence:.1f}%)")
        finally:
            if os.path.exists(user_video_path):
                os.unlink(user_video_path)

        # Process user video
        st.info("Processing your video...")
        user_poses, user_frames = process_video(user_video, frame_rate, max_frames, "User")

        if not user_poses:
            st.error("Failed to process video. Please try a different video.")
            return
        
        st.success("✅ Video Processing completed!")

        # Analyze technique
        st.info("Analyzing technique...")
        analysis_result = st.session_state.boxing_analyzer.analyze_technique(user_poses, technique)

    # Clear progress and show results
    progress_container.empty()

    with results_container:
        display_technique_results(analysis_result, user_frames, user_poses, technique)
    
    processing_time = st.session_state.performance_monitor.end_timer("technique_analysis")
    st.sidebar.success(f"⏱️ Analysis completed in {processing_time:.2f}s")

def process_video(video_file, frame_rate: int, max_frames: int, video_type: str) -> Tuple[List[Dict], List[np.ndarray]]:
    # Process video file and extract poses.
    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.getvalue())
        video_path = tmp_file.name

    try:
        poses, frames = st.session_state.video_processor.process_video(
            video_path, frame_rate, max_frames, video_type
        )
        return poses, frames
    finally:
        os.unlink(video_path)

# ... existing imports and code ...

def display_comparison_results(
    comparison_result: Dict,
    ref_frames: List,
    user_frames: List,
    ref_poses: List,
    user_poses: List,
):
    st.header("📊 Comparison Results")

    if "error" in comparison_result:
        st.error(f"Analysis Error: {comparison_result['error']}")
        return

    overall_similarity = comparison_result.get("overall_similarity", 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Similarity", f"{overall_similarity:.1f}%")
    with col2:
        st.metric("Frames Compared", comparison_result.get("frames_compared", 0))
    with col3:
        similarity_color = "🟢" if overall_similarity >= 70 else "🟡" if overall_similarity >= 50 else "🔴"
        st.metric(
            "Quality",
            f"{similarity_color} "
            f"{'Excellent' if overall_similarity >= 70 else 'Good' if overall_similarity >= 50 else 'Needs Work'}",
        )

    # Joint Similarities chart
    joint_similarities = comparison_result.get("joint_similarities", {})
    if joint_similarities:
        st.subheader("🦴 Joint Analysis")

        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(joint_similarities.keys()),
                    y=list(joint_similarities.values()),
                    marker_color=[
                        "green" if v >= 70 else "orange" if v >= 50 else "red"
                        for v in joint_similarities.values()
                    ],
                )
            ]
        )

        fig.update_layout(
            title="Joint Similarity Scores",
            xaxis_title="Body Joints",
            yaxis_title="Similarity (%)",
            yaxis=dict(range=[0, 100]),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Frame-by-frame comparison
    st.subheader("🎞️ Frame-by-Frame Analysis")

    if ref_frames and user_frames:
        # Max usable frames for each side (respecting both frames and poses)
        ref_max = min(len(ref_frames), len(ref_poses))
        user_max = min(len(user_frames), len(user_poses))

        if ref_max > 0 and user_max > 0:
            # Separate sliders: one for reference, one for user
            slider_col_ref, slider_col_user = st.columns(2)

            with slider_col_ref:
                ref_idx = st.slider(
                    "Reference frame",
                    0,
                    ref_max - 1,
                    ref_max // 2,
                    key="ref_frame_slider",
                )

            with slider_col_user:
                user_idx = st.slider(
                    "User frame",
                    0,
                    user_max - 1,
                    user_max // 2,
                    key="user_frame_slider",
                )

            # Display frames side by side
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Reference Video**")
                ref_frame = ref_frames[ref_idx]
                ref_pose = ref_poses[ref_idx] if ref_idx < len(ref_poses) else None

                if ref_pose:
                    annotated_ref = st.session_state.pose_estimator.draw_pose(ref_frame, ref_pose)
                    st.image(cv2.cvtColor(annotated_ref, cv2.COLOR_BGR2RGB))
                else:
                    st.image(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB))

            with col2:
                st.write("**User Video**")
                user_frame = user_frames[user_idx]
                user_pose = user_poses[user_idx] if user_idx < len(user_poses) else None

                if user_pose:
                    # Compare the currently selected reference and user frames
                    differences = {}
                    if ref_pose:
                        differences = calculate_frame_differences(ref_pose, user_pose)

                    annotated_user = st.session_state.pose_estimator.draw_pose(
                        user_frame,
                        user_pose,
                        differences,
                    )
                    st.image(cv2.cvtColor(annotated_user, cv2.COLOR_BGR2RGB))
                else:
                    st.image(cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB))




def display_technique_results(analysis_result: Dict, user_frames: List, user_poses: List, technique: str):
    # Display technique analysis results.
    st.header("📊 Technique Analysis Results")

    if 'error' in analysis_result.get('feedback', {}):
        st.error("Analysis Error: " + analysis_result['feedback']['error'][0])
        return
    
    overall_accuracy = analysis_result.get('overall_accuracy', 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Accuracy", f"{overall_accuracy:.1f}%")
    with col2:
        st.metric("Technique", technique.title())
    with col3:
        accuracy_color = "🟢" if overall_accuracy >= 80 else "🟡" if overall_accuracy >=60 else "🔴"
        st.metric("Grade", f"{accuracy_color} {'A' if overall_accuracy >= 80 else 'B' if overall_accuracy >= 60 else 'C'}")

    # Joint accuracies
    joint_accuracies = analysis_result.get('joint_accuracies', {})
    if joint_accuracies:
        st.subheader("🦴 Joint Analysis")

        # Visualization
        viz_data = create_accuracy_visualization(joint_accuracies)

        fig = go.Figure(data= [
            go.Bar(
                x=viz_data['joints'],
                y=viz_data['accuracies'],
                marker_color=viz_data['colors'],
                text=[f"{acc:.1f}% " for acc in viz_data['accuracies']],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title="Joint Accuracy Breakdown",
            xaxis_title="Body Joint",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    feedback = analysis_result.get('feedback', {})
    if feedback:
        st.subheader("💡 Improvement Suggestions")
        formatted_feedback = format_feedback_message(feedback)
        st.markdown(formatted_feedback)

    # Best frame analysis
    best_frame_idx = analysis_result.get('best_frame_idx', 0)
    if user_frames and user_poses and best_frame_idx < len(user_frames):
        st.subheader("🎯 Best Frame Analysis")
        st.write(f"Frame {best_frame_idx + 1} shows your best technique execution")

        if best_frame_idx < len(user_poses) and user_poses[best_frame_idx]:
            annoted_frame = st.session_state.pose_estimator.draw_pose(
                user_frames[best_frame_idx], user_poses[best_frame_idx]
            )
            st.image(cv2.cvtColor(annoted_frame, cv2.COLOR_BGR2RGB),
                     caption=f"Best technique frame (Frame {best_frame_idx + 1})")
        else:
            st.image(cv2.cvtColor(user_frames[best_frame_idx], cv2.COLOR_BGR2RGB),
                     caption=f"Frame {best_frame_idx + 1}")
    
    frame_scores = analysis_result.get('frame_scores', [])
    if frame_scores:
        st.subheader("📈 Technique Consistency")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=frame_scores,
            mode='lines+markers',
            name='Accuracy Score',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title="Technique Accuracy Over Time",
            xaxis_title="Frame Number",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)

def calculate_frame_differences(ref_pose: Optional[Dict], user_pose: Dict) -> Dict[str, float]:
    if not ref_pose or not user_pose:
        return {}

    # fixed key: 'angles' (was 'anagles')
    ref_angles = ref_pose.get("angles", {})
    user_angles = user_pose.get("angles", {})

    differences: Dict[str, float] = {}
    for angle_name, ref_val in ref_angles.items():
        if angle_name in user_angles:
            differences[angle_name] = abs(ref_val - user_angles[angle_name])

    return differences

class BoxingVideoProcessor(VideoProcessorBase):
    def __init__(self, technique: str):
        self.technique = technique
        self.pose_estimator = PoseEstimator()
        self.boxing_analyzer = BoxingAnalyzer()
        self.reference_poses = ReferenceBoxingPoses()
        self.accuracy_history = []

    def recv(self, frame):
        bgr_frame = frame.to_ndarray(format="bgr24")
        bgr_frame = cv2.flip(bgr_frame, 1)

        reference_pose = self.reference_poses.get_reference_pose(self.technique)
        pose_data = self.pose_estimator.estimate_pose(bgr_frame)

        if pose_data and reference_pose:
            user_angles = pose_data.get("angles", {})
            reference_angles = reference_pose["angles"]

            joint_accuracies = self.boxing_analyzer._calculate_joint_accuracies(
                user_angles,
                reference_angles,
            )
            overall_accuracy = self.boxing_analyzer._calculate_overall_accuracy(
                joint_accuracies,
                self.boxing_analyzer.angle_weights,
            )

            self.accuracy_history.append(overall_accuracy)
            if len(self.accuracy_history) > 30:
                self.accuracy_history.pop(0)

            average_accuracy = float(np.mean(self.accuracy_history))

            if overall_accuracy >= 70:
                feedback_color = (0, 255, 0)
                status = "Excellent"
            elif overall_accuracy >= 50:
                feedback_color = (0, 165, 255)
                status = "Good"
            else:
                feedback_color = (0, 0, 255)
                status = "Practice"

            annotated_frame = self.pose_estimator.draw_pose(bgr_frame, pose_data)
            cv2.putText(
                annotated_frame,
                f"Accuracy: {overall_accuracy:.1f}%  Average: {average_accuracy:.1f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                feedback_color,
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Technique: {self.technique.title()}  [{status}]",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
        else:
            annotated_frame = bgr_frame.copy()
            cv2.putText(
                annotated_frame,
                "No pose detected - move into camera view",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


def run_realtime_analysis(technique: str, frame_rate: int):
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_ctx = webrtc_streamer(
        key=f"boxing-camera-{technique}",
        video_processor_factory=lambda: BoxingVideoProcessor(technique),
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": frame_rate, "max": frame_rate},
            },
            "audio": False,
        },
        async_processing=True,
    )

    if webrtc_ctx and webrtc_ctx.state.playing:
        st.success("Browser camera is streaming")
    elif webrtc_ctx:
        st.info("Allow camera access, then click START.")

if __name__ == "__main__":
    main()
