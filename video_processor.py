# Video processing module for MyoCouch using Qwen2-VL
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import tempfile
import ffmpeg
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, ColorClip, clips_array, ImageClip
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime

logger = logging.getLogger('MyoCouch.VideoProcessor')


class VideoCoachingProcessor:
    """Processes videos for coaching analysis using AI vision language model."""
    
    def __init__(self, model_size: str = "2B"):
        """
        Initialize the video processor with AI vision model.
        
        Args:
            model_size: Model size - "2B" (default) or "7B"
        """
        self.model_size = model_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Qwen2-VL model
        model_mapping = {
            "2B": "Qwen/Qwen2-VL-2B-Instruct",
            "7B": "Qwen/Qwen2-VL-7B-Instruct"
        }
        
        model_path = model_mapping.get(model_size, model_mapping["7B"])
        logger.info(f"Loading Qwen2-VL model: {model_path}")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
                
        # Video processing parameters
        self.target_fps = 30
        self.chunk_size = 120  # frames per chunk (4 seconds at 30fps)
        self.frames_per_analysis = 8  # frames to sample from each chunk for analysis
        self.max_chunks = 6  # Maximum chunks to process to prevent timeout
        
        self.coaching_prompt = """You are MyoCouch, an expert fitness coach analyzing workout videos. 
        Analyze this exercise video segment and provide specific, actionable coaching advice.
        Focus on:
        1. Form and technique issues
        2. Body positioning and alignment
        3. Movement quality and control
        4. Safety concerns
        Keep your response concise and practical (max 3 key points).
        Format: Provide direct advice without numbering or bullet points."""
        
    def downsample_video(self, input_path: str, output_path: str, target_fps: int = 30) -> str:
        """
        Downsample video to target FPS using ffmpeg.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            target_fps: Target frames per second
            
        Returns:
            Path to downsampled video
        """
        try:
            (
                ffmpeg
                .input(input_path)
                .filter('fps', fps=target_fps)
                .output(output_path, video_bitrate='2M')
                .overwrite_output()
                .run(quiet=True)
            )
            return output_path
        except Exception as e:
            logger.error(f"Error downsampling video: {e}")
            raise
    
    def extract_video_chunks(self, video_path: str) -> List[np.ndarray]:
        """
        Extract video chunks of specified frame count.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of video chunks (each chunk is array of frames)
        """
        cap = cv2.VideoCapture(video_path)
        chunks = []
        current_chunk = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Save last chunk if it has frames
                if current_chunk:
                    chunks.append(np.array(current_chunk))
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_chunk.append(frame_rgb)
            
            # Check if chunk is complete
            if len(current_chunk) >= self.chunk_size:
                chunks.append(np.array(current_chunk))
                current_chunk = []
        
        cap.release()
        return chunks
    
    def sample_frames_from_chunk(self, chunk: np.ndarray, num_frames: int = 8) -> List[Image.Image]:
        """
        Sample frames uniformly from a video chunk.
        
        Args:
            chunk: Video chunk as numpy array
            num_frames: Number of frames to sample
            
        Returns:
            List of PIL Images
        """
        total_frames = len(chunk)
        if total_frames <= num_frames:
            # Use all frames if chunk is small
            indices = list(range(total_frames))
        else:
            # Sample uniformly
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            frame = chunk[idx]
            pil_image = Image.fromarray(frame)
            frames.append(pil_image)
        
        return frames
    
    def analyze_video_chunk(self, chunk: np.ndarray, chunk_index: int = 0, previous_advice: List[str] = None) -> str:
        """
        Analyze a video chunk using AI vision model and return coaching advice.
        
        Args:
            chunk: Video chunk as numpy array (frames, height, width, channels)
            chunk_index: Index of current chunk
            previous_advice: List of advice from previous chunks to avoid repetition
            
        Returns:
            Coaching advice text
        """
        # Sample frames from the chunk
        frames = self.sample_frames_from_chunk(chunk, self.frames_per_analysis)
        
        # Build context from previous advice
        context = ""
        if previous_advice and len(previous_advice) > 0:
            context = "\n\nPrevious coaching points covered:\n"
            for i, advice in enumerate(previous_advice[-2:]):  # Use last 2 pieces of advice
                context += f"- Segment {i+1}: {advice[:100]}...\n"
            context += "\nPlease provide NEW advice that hasn't been mentioned yet."
        
        # Modified prompt to encourage variety
        prompt = f"""{self.coaching_prompt}
        
        This is segment {chunk_index + 1} of the video.{context}
        
        Focus on NEW observations and avoid repeating previous points."""
        
        # Create the conversation format for the AI model
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this exercise video segment:"},
                ] + [{"type": "image"} for _ in frames]
            }
        ]
        
        # Prepare the input using the processor
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=text,
            images=frames,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate coaching advice
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,  # Increased for more detailed advice
                temperature=0.8,  # Slightly higher for more variety
                do_sample=True,
                top_p=0.9
            )
        
        # Decode the response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "Analyze this exercise video segment:" in response:
            advice = response.split("Analyze this exercise video segment:")[-1].strip()
        else:
            advice = response.strip()
        
        # Clean up the advice
        advice = advice.replace("Assistant:", "").replace("assistant\n", "").replace("assistant", "").strip()
        
        return advice
    
    def summarize_advice(self, advice: str, chunk_index: int, previous_summaries: List[str] = None) -> str:
        """
        Use the AI model to create a non-repetitive summary with memory context.
        
        Args:
            advice: Full coaching advice text
            chunk_index: Index of current chunk
            previous_summaries: List of previous summaries to avoid repetition
            
        Returns:
            AI-generated summary (80 words max, plain text)
        """
        # Build context from previous summaries
        memory_context = ""
        if previous_summaries and len(previous_summaries) > 0:
            memory_context = "\n\nPREVIOUS ADVICE ALREADY GIVEN:\n"
            for i, prev_summary in enumerate(previous_summaries):
                memory_context += f"Segment {i+1}: {prev_summary}\n"
            memory_context += "\nIMPORTANT: Do NOT repeat these points. Provide NEW, different advice."
        
        # Create summarization prompt with specific instructions
        messages = [
            {
                "role": "system",
                "content": f"""You are a fitness coach creating video coaching summaries. 

CRITICAL INSTRUCTIONS:
1) PLEASE DON'T BE REPETITIVE - sound different from previous advice
2) PLEASE USE PLAIN TEXT and maintain in 80 words maximum

Create a concise, practical summary focused on NEW coaching points. Avoid generic advice. Be specific and actionable.{memory_context}"""
            },
            {
                "role": "user",
                "content": f"""This is segment {chunk_index + 1} of a workout video.

DETAILED ANALYSIS:
{advice}

Please create a 80-word summary with NEW coaching advice that hasn't been mentioned before. Use plain text, no formatting, no bullet points."""
            }
        ]
        
        # Prepare the input
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,  # Allow enough tokens for 80 words
                temperature=0.8,  # Higher temperature for more variety
                do_sample=True,
                top_p=0.9
            )
        
        # Decode the full response
        full_response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Find where the AI's actual response starts
        # The response typically comes after the user message
        response_markers = [
            "Please create a 80-word summary",
            "DETAILED ANALYSIS:",
            "assistant\n",
            "Assistant:",
            "\n\n"  # Sometimes the response starts after double newline
        ]
        
        summary = full_response
        for marker in response_markers:
            if marker in summary:
                parts = summary.split(marker)
                # Take the last part which should be the AI's response
                potential_summary = parts[-1].strip()
                if potential_summary and len(potential_summary) > 20:
                    summary = potential_summary
                    break
        
        # Additional cleanup - remove any remaining prompt text
        unwanted_phrases = [
            "with NEW coaching advice",
            "that hasn't been mentioned before",
            "Use plain text",
            "no formatting",
            "no bullet points",
            "CRITICAL INSTRUCTIONS",
            "PLEASE DON'T BE REPETITIVE",
            "PLEASE USE PLAIN TEXT"
        ]
        
        for phrase in unwanted_phrases:
            summary = summary.replace(phrase, "")
        
        # Final cleanup
        summary = summary.replace("Assistant:", "").replace("assistant\n", "").replace("assistant", "").strip()
        summary = summary.replace("•", "").replace("-", "").replace("*", "")
        
        # Clean up any multiple spaces
        summary = ' '.join(summary.split())
        
        # Ensure it's not too long
        words = summary.split()
        if len(words) > 80:
            summary = ' '.join(words[:80])
        
        return summary
        
    
    def create_side_panel(self, text: str, video_size: Tuple[int, int], duration: float) -> VideoFileClip:
        """
        Create a side panel with text instead of overlay.
        
        Args:
            text: Text to display (plain text, ~80 words)
            video_size: (width, height) of original video
            duration: Duration of the clip
            
        Returns:
            VideoFileClip with black panel containing text
        """
        width, height = video_size
        panel_width = width  # Panel is same width as video (100%)
        
        # Clean up the text
        text = text.replace("assistant\n", "").replace("assistant", "").strip()
        
        # Create black image
        panel_img = Image.new('RGB', (panel_width, height), color='black')
        draw = ImageDraw.Draw(panel_img)
        
        # Use a single, readable font for plain text
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Simple text wrapping for plain text
        margin = 30
        max_width = panel_width - (margin * 2)
        y = 50  # Start from top with some padding
        
        # Split into words and wrap
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            
            if bbox[2] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        # Draw the text lines
        line_height = 35  # Spacing between lines
        for line in lines:
            if y + line_height > height - 50:  # Stop if running out of space
                break
            draw.text((margin, y), line, fill='white', font=font)
            y += line_height
        
        # Add segment indicator at the bottom
        segment_num = int(duration // 4) + 1  # Segment number based on 4 seconds per chunk
        segment_text = f"Segment {segment_num}"
        draw.text((margin, height - 40), segment_text, fill='gray', font=font)
        
        # Convert PIL image to numpy array
        panel_array = np.array(panel_img)
        
        # Create video from the static image
        panel_clip = ImageClip(panel_array, duration=duration)
        
        return panel_clip
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process entire video: downsample, chunk, analyze, and create coached video.
        
        Args:
            video_path: Path to input video
            
        Returns:
            Dictionary with processing results and path to coached video
        """
        logger.info(f"Processing video: {video_path}")
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Downsample video to 30 FPS
            downsampled_path = os.path.join(temp_dir, "downsampled.mp4")
            logger.info("Downsampling video to 30 FPS...")
            self.downsample_video(video_path, downsampled_path, self.target_fps)
            
            # Get video info
            cap = cv2.VideoCapture(downsampled_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            # Step 2: Extract video chunks
            logger.info(f"Extracting video chunks ({self.chunk_size} frames each)...")
            chunks = self.extract_video_chunks(downsampled_path)
            
            # Limit chunks to prevent timeout - cut video if too long
            original_chunk_count = len(chunks)
            if len(chunks) > self.max_chunks:
                logger.warning(f"Video has {len(chunks)} chunks, cutting to first {self.max_chunks} chunks (processing first {self.max_chunks * 4} seconds)")
                chunks = chunks[:self.max_chunks]
                
                # Update total frames to reflect the cut
                total_frames = min(total_frames, self.max_chunks * self.chunk_size)
            
            logger.info(f"Processing {len(chunks)} chunks" + 
                       (f" (video cut from {original_chunk_count} chunks)" if original_chunk_count > self.max_chunks else ""))
            
            # Step 3: Analyze each chunk and create overlay videos
            chunk_clips = []
            all_advice = []
            previous_advice_summaries = []  # Track previous summaries for context
            memory_data = {
                'video_path': video_path,
                'chunks': []
            }
            
            for i, chunk in enumerate(tqdm(chunks, desc="Analyzing chunks")):
                # Get coaching advice for this chunk
                logger.info(f"Analyzing chunk {i+1}/{len(chunks)}...")
                # Pass previous advice summaries to the analysis function
                advice = self.analyze_video_chunk(chunk, i, previous_advice_summaries)
                all_advice.append(f"Segment {i+1}: {advice}")
                
                # Summarize the advice for the overlay
                summary = self.summarize_advice(advice, i, previous_advice_summaries)
                previous_advice_summaries.append(summary)  # Add the summary to the list
                
                # Save to memory
                memory_data['chunks'].append({
                    'segment': i + 1,
                    'full_advice': advice,
                    'summary': summary,
                    'timestamp': f"{i * 4.0:.1f}s - {(i + 1) * 4.0:.1f}s"  # 4 seconds per chunk
                })
                
                logger.info(f"Summary for chunk {i+1}: {summary}")
                
                # Create video clip for this chunk with overlay
                chunk_duration = len(chunk) / self.target_fps
                
                # Save chunk as temporary video
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(chunk_path, fourcc, self.target_fps, (width, height))
                
                for frame in chunk:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()
                
                # Create video clip with side panel
                video_clip = VideoFileClip(chunk_path)
                text_panel = self.create_side_panel(summary, (width, height), chunk_duration)
                
                # Place video and panel side by side
                composite = clips_array([[video_clip, text_panel]])
                chunk_clips.append(composite)
            
            # Step 4: Concatenate all chunks
            logger.info("Concatenating coached video segments...")
            final_video = concatenate_videoclips(chunk_clips)
            
            # Save final coached video with reduced file size
            output_path = video_path.replace('.mp4', '_coached.mp4')
            final_video.write_videofile(
                output_path,
                fps=self.target_fps,
                codec='libx264',
                audio_codec='aac' if final_video.audio is not None else None,
                temp_audiofile=os.path.join(temp_dir, 'temp-audio.m4a'),
                remove_temp=True,
                logger=None,  # Suppress moviepy logging
                preset='ultrafast',  # Faster encoding
                ffmpeg_params=['-crf', '28', '-maxrate', '1M', '-bufsize', '2M'],  # Conservative encoding
                verbose=False,
                threads=2  # Limit threads to prevent resource exhaustion
            )
            
            # Clean up
            final_video.close()
            for clip in chunk_clips:
                clip.close()
            
            # Save memory file
            memory_filename = os.path.basename(video_path).replace('.mp4', '_memory.json')
            memory_path = output_path.replace('_coached.mp4', '_memory.json')
            
            # Add metadata to memory
            memory_data['metadata'] = {
                'processed_at': datetime.now().isoformat(),
                'model_used': f'AI Vision-{self.model_size}',
                'total_duration': total_frames / fps,
                'resolution': f"{width}x{height}",
                'chunks_processed': len(chunks)
            }
            
            # Save memory to JSON file
            with open(memory_path, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Coaching memory saved to: {memory_path}")
            
            # Prepare results
            results = {
                'status': 'success',
                'video_info': {
                    'duration_seconds': total_frames / fps,
                    'original_duration_seconds': original_duration,
                    'fps': fps,
                    'resolution': (width, height),
                    'total_frames': total_frames,
                    'chunks_processed': len(chunks),
                    'video_was_cut': original_chunk_count > self.max_chunks
                },
                'coaching_segments': all_advice,
                'output_video_path': output_path,
                'memory_file_path': memory_path,
                'model_used': f'AI Vision-{self.model_size}'
            }
            
            logger.info(f"Video processing complete. Output saved to: {output_path}")
            return results
    
    def compare_motions(self, video1_path: str, video2_path: str) -> Dict:
        """
        Compare two videos side by side and analyze which has better form.
        
        Args:
            video1_path: Path to first video
            video2_path: Path to second video
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing videos: {video1_path} vs {video2_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Downsample both videos to 30 FPS
            logger.info("Downsampling videos to 30 FPS...")
            downsampled1 = os.path.join(temp_dir, "video1_downsampled.mp4")
            downsampled2 = os.path.join(temp_dir, "video2_downsampled.mp4")
            
            self.downsample_video(video1_path, downsampled1, self.target_fps)
            self.downsample_video(video2_path, downsampled2, self.target_fps)
            
            # Step 2: Extract first 5 seconds (150 frames) from each video
            logger.info("Extracting first 5 seconds from each video...")
            frames_to_extract = 150  # 5 seconds at 30 fps
            
            # Extract frames from video 1
            cap1 = cv2.VideoCapture(downsampled1)
            frames1 = []
            for _ in range(frames_to_extract):
                ret, frame = cap1.read()
                if not ret:
                    break
                frames1.append(frame)
            cap1.release()
            
            # Extract frames from video 2
            cap2 = cv2.VideoCapture(downsampled2)
            frames2 = []
            for _ in range(frames_to_extract):
                ret, frame = cap2.read()
                if not ret:
                    break
                frames2.append(frame)
            cap2.release()
            
            # Make both videos same length
            min_frames = min(len(frames1), len(frames2))
            frames1 = frames1[:min_frames]
            frames2 = frames2[:min_frames]
            
            logger.info(f"Extracted {min_frames} frames from each video")
            
            # Get dimensions early for use in analysis
            h1, w1 = frames1[0].shape[:2]
            h2, w2 = frames2[0].shape[:2]
            
            # Step 3: Analyze with VLM first to get the text for the panel
            logger.info("Analyzing motion differences with AI...")
            
            # Sample frames for analysis (every 30 frames = 1 per second)
            sample_indices = range(0, min_frames, 30)[:5]  # 5 frames, one per second
            
            analysis_frames = []
            for idx in sample_indices:
                # Convert BGR to RGB for PIL
                frame1_rgb = cv2.cvtColor(frames1[idx], cv2.COLOR_BGR2RGB)
                frame2_rgb = cv2.cvtColor(frames2[idx], cv2.COLOR_BGR2RGB)
                
                # Resize for analysis
                frame1_small = cv2.resize(frame1_rgb, (320, int(320 * h1 / w1)))
                frame2_small = cv2.resize(frame2_rgb, (320, int(320 * h2 / w2)))
                
                # Create side by side for analysis
                combined_rgb = np.hstack([frame1_small, frame2_small])
                pil_image = Image.fromarray(combined_rgb)
                analysis_frames.append(pil_image)
            
            # Create comparison prompt
            messages = [
                {
                    "role": "system",
                    "content": """You are MyoCouch, an expert fitness coach comparing two exercise videos.
                    Your job is to determine which video is better in the exercise and explain why.
                    
                    RULES:
                    1. You MUST pick a winner - either Video 1 or Video 2
                    2. Be CRITICAL and DIRECT about errors
                    3. Don't sugarcoat problems - point them out clearly
                    
                    Format EXACTLY as:
                    VERDICT: Video [1 or 2] has better form because [two specific reasons]
                    VIDEO 1: [If winner: Why the form is superior. If loser: What specific errors you see. Max 3 short sentences]
                    VIDEO 2: [If winner: Why the form is superior. If loser: What specific errors you see. Max 3 short sentences]
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these two exercise videos. Video 1 is on the left, Video 2 is on the right."},
                    ] + [{"type": "image"} for _ in analysis_frames]
                }
            ]
            
            # Get AI analysis
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=text,
                images=analysis_frames,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # Reduced for concise responses
                    temperature=0.6,  # Lower temperature for more focused output
                    do_sample=True,
                    top_p=0.9
                )
            
            # Decode response
            full_response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract analysis
            analysis = full_response
            if "Compare these two exercise videos" in analysis:
                parts = analysis.split("Compare these two exercise videos")
                if len(parts) > 1:
                    analysis = parts[-1].strip()
            
            # Parse the structured response
            verdict = ""
            video1_analysis = ""
            video2_analysis = ""
            
            lines = analysis.split('\n')
            current_section = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.upper().startswith("VERDICT:"):
                    verdict = line.replace("VERDICT:", "").strip()
                    current_section = "verdict"
                elif line.upper().startswith("VIDEO 1:"):
                    video1_analysis = line.replace("VIDEO 1:", "").replace("Video 1:", "").strip()
                    current_section = "video1"
                elif line.upper().startswith("VIDEO 2:"):
                    video2_analysis = line.replace("VIDEO 2:", "").replace("Video 2:", "").strip()
                    current_section = "video2"
                else:
                    # Continue adding to current section
                    if current_section == "verdict" and verdict and not line.upper().startswith("VIDEO"):
                        verdict += " " + line
                    elif current_section == "video1" and not line.upper().startswith("VIDEO"):
                        video1_analysis += " " + line
                    elif current_section == "video2":
                        video2_analysis += " " + line
            
            # Clean up and ensure we have content
            verdict = verdict.strip()
            video1_analysis = video1_analysis.strip()
            video2_analysis = video2_analysis.strip()
            
            # Step 4: Create side-by-side video with text panel at bottom
            logger.info("Creating side-by-side comparison video with analysis panel...")
            
            # Get dimensions
            h1, w1 = frames1[0].shape[:2]
            h2, w2 = frames2[0].shape[:2]
            
            # Resize to same height
            video_height = min(h1, h2, 360)  # Good balance with efficient text panel
            
            # Calculate new widths maintaining aspect ratio
            new_w1 = int(w1 * video_height / h1)
            new_w2 = int(w2 * video_height / h2)
            
            # Total dimensions
            total_width = new_w1 + new_w2
            panel_height = 200  # Balanced height for efficient text display
            total_height = video_height + panel_height
            
            # Create temporary video without text panel first
            temp_video_path = os.path.join(temp_dir, "temp_comparison.mp4")
            
            # Process with MoviePy for better control
            from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ImageClip, clips_array, concatenate_videoclips
            
            # Create video clips from frames
            video_frames = []
            
            for i in range(min_frames):
                # Resize frames
                frame1_resized = cv2.resize(frames1[i], (new_w1, video_height))
                frame2_resized = cv2.resize(frames2[i], (new_w2, video_height))
                
                # Convert to RGB
                frame1_rgb = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)
                frame2_rgb = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2RGB)
                
                # Concatenate horizontally
                combined_frame = np.hstack([frame1_rgb, frame2_rgb])
                
                # Add labels
                pil_frame = Image.fromarray(combined_frame)
                draw = ImageDraw.Draw(pil_frame)
                
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 30)
                except:
                    font = ImageFont.load_default()
                
                # Draw video labels
                draw.text((20, 20), "Video 1", fill=(100, 200, 255), font=font)  # Light blue
                draw.text((new_w1 + 20, 20), "Video 2", fill=(255, 150, 100), font=font)  # Light orange
                
                video_frames.append(np.array(pil_frame))
            
            # Create video from frames
            video_clip = ImageClip(video_frames[0], duration=1/self.target_fps)
            clips = [ImageClip(frame, duration=1/self.target_fps) for frame in video_frames]
            video_sequence = concatenate_videoclips(clips, method="compose")
            
            # Create text panel
            text_panel_img = Image.new('RGB', (total_width, panel_height), color='black')
            draw = ImageDraw.Draw(text_panel_img)
            
            # Format text for panel
            try:
                header_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 12)
                body_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 12)
            except:
                header_font = ImageFont.load_default()
                body_font = ImageFont.load_default()
            
            # Draw text on panel
            y_offset = 10
            margin = 15
            
            # Verdict - inline with text
            draw.text((margin, y_offset), "Winner:", fill=(255, 255, 255), font=header_font)
            verdict_x_offset = margin + 50  # Start verdict text after "Winner:"
            
            # Wrap verdict text
            verdict_words = verdict.split()
            verdict_lines = []
            current_line = ""
            first_line = True
            for word in verdict_words:
                test_line = current_line + " " + word if current_line else word
                bbox = draw.textbbox((0, 0), test_line, font=body_font)
                max_verdict_width = total_width - verdict_x_offset - margin if first_line else total_width - (margin * 2)
                if bbox[2] <= max_verdict_width:
                    current_line = test_line
                else:
                    if current_line:
                        verdict_lines.append(current_line)
                        first_line = False
                    current_line = word
            if current_line:
                verdict_lines.append(current_line)
            
            # Draw verdict text
            for i, line in enumerate(verdict_lines[:3]):
                if i == 0:
                    draw.text((verdict_x_offset, y_offset), line, fill=(200, 200, 200), font=body_font)
                else:
                    draw.text((margin, y_offset), line, fill=(200, 200, 200), font=body_font)
                y_offset += 14
            
            y_offset += 10  # Space before analysis sections
            
            # Video 1 Analysis
            left_column_x = margin
            analysis_y_start = y_offset
            draw.text((left_column_x, y_offset), "V1:", fill=(100, 200, 255), font=header_font)  # Light blue
            y_offset += 14
            
            # Wrap video 1 analysis - allow more text
            v1_words = video1_analysis.split()
            v1_lines = []
            current_line = ""
            max_width = (total_width // 2) - (margin * 3)  # Adjusted for better fit
            
            for word in v1_words:
                test_line = current_line + " " + word if current_line else word
                bbox = draw.textbbox((0, 0), test_line, font=body_font)
                if bbox[2] <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        v1_lines.append(current_line)
                    current_line = word
            if current_line:
                v1_lines.append(current_line)
            
            v1_y = y_offset
            max_lines = 4  # 2 sentences should fit in 3-4 lines max
            for i, line in enumerate(v1_lines[:max_lines]):
                if v1_y + 14 > panel_height - 10:  # Stop if running out of space
                    break
                draw.text((left_column_x, v1_y), line, fill=(200, 200, 200), font=body_font)
                v1_y += 14
            
            # Video 2 Analysis (right column)
            right_column_x = total_width // 2 + margin
            draw.text((right_column_x, analysis_y_start), "V2:", fill=(255, 150, 100), font=header_font)  # Light orange
            
            # Wrap video 2 analysis
            v2_words = video2_analysis.split()
            v2_lines = []
            current_line = ""
            
            for word in v2_words:
                test_line = current_line + " " + word if current_line else word
                bbox = draw.textbbox((0, 0), test_line, font=body_font)
                if bbox[2] <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        v2_lines.append(current_line)
                    current_line = word
            if current_line:
                v2_lines.append(current_line)
            
            v2_y = analysis_y_start + 14  # Start at same height as video 1 text
            for i, line in enumerate(v2_lines[:max_lines]):
                if v2_y + 14 > panel_height - 10:  # Stop if running out of space
                    break
                draw.text((right_column_x, v2_y), line, fill=(200, 200, 200), font=body_font)
                v2_y += 14
            
            # Convert panel to video clip
            panel_array = np.array(text_panel_img)
            panel_clip = ImageClip(panel_array, duration=video_sequence.duration)
            
            # Stack video and panel vertically
            final_video = clips_array([[video_sequence], [panel_clip]])
            
            # Final output path
            final_output = video1_path.replace('.mp4', '_comparison.mp4')
            
            # Write final video
            final_video.write_videofile(
                final_output,
                fps=self.target_fps,
                codec='libx264',
                audio=False,
                preset='ultrafast',
                logger=None,
                ffmpeg_params=['-crf', '23']  # Better quality
            )
            
            # Clean up
            video_sequence.close()
            panel_clip.close()
            final_video.close()
            
            # Prepare results
            results = {
                'status': 'success',
                'verdict': verdict,
                'video1_analysis': video1_analysis.strip(),
                'video2_analysis': video2_analysis.strip(),
                'output_video_path': final_output,
                'model_used': f'AI Vision-{self.model_size}',
                'frames_analyzed': min_frames,
                'duration_seconds': min_frames / self.target_fps
            }
            
            logger.info(f"Motion comparison complete. Output saved to: {final_output}")
            return results 