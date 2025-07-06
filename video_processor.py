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
        self.chunk_size = 60  # frames per chunk
        self.frames_per_analysis = 8  # frames to sample from each chunk for analysis
        self.max_chunks = 10  # Maximum chunks to process to prevent timeout
        
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
        
        # If we still have prompt text or summary is too short, use a simple fallback
        if len(summary.strip()) < 20 or any(phrase in summary for phrase in ["NEW coaching", "80-word", "DETAILED ANALYSIS"]):
            # Simple, direct fallbacks
            fallbacks = [
                "Watch your foot placement during the movement. Your weight should shift smoothly from heel to toe. Focus on creating a stable base before initiating the exercise. This foundation will help you maintain balance throughout the entire range of motion.",
                "Notice how your breathing affects your performance. Exhale during the exertion phase and inhale during the recovery. Proper breathing helps maintain core stability and provides oxygen to working muscles. Never hold your breath during exercise.",
                "Your tempo seems rushed in this segment. Slow down the movement to feel each muscle working. A controlled pace allows better muscle activation and reduces injury risk. Count two seconds up, two seconds down.",
                "Check the alignment of your joints throughout the movement. Your knees should track over your toes, not cave inward. This proper alignment protects your joints and ensures the right muscles are doing the work.",
                "Consider your range of motion here. You're cutting the movement short. Work through the full range to maximize muscle engagement and flexibility. Start with what's comfortable and gradually increase as you improve."
            ]
            summary = fallbacks[chunk_index % len(fallbacks)]
        
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
        segment_text = f"Segment {duration//2:.0f}"  # Rough segment number
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
            cap.release()
            
            # Step 2: Extract video chunks
            logger.info(f"Extracting video chunks ({self.chunk_size} frames each)...")
            chunks = self.extract_video_chunks(downsampled_path)
            
            # Limit chunks to prevent timeout
            if len(chunks) > self.max_chunks:
                logger.warning(f"Video has {len(chunks)} chunks, limiting to {self.max_chunks} to prevent timeout")
                chunks = chunks[:self.max_chunks]
            
            logger.info(f"Processing {len(chunks)} chunks")
            
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
                    'timestamp': f"{i * 2.0:.1f}s - {(i + 1) * 2.0:.1f}s"  # Assuming 2 seconds per chunk
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
                    'fps': fps,
                    'resolution': (width, height),
                    'total_frames': total_frames,
                    'chunks_processed': len(chunks)
                },
                'coaching_segments': all_advice,
                'output_video_path': output_path,
                'memory_file_path': memory_path,
                'model_used': f'AI Vision-{self.model_size}'
            }
            
            logger.info(f"Video processing complete. Output saved to: {output_path}")
            return results 