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
    
    def summarize_advice(self, advice: str, chunk_index: int) -> str:
        """
        Create a comprehensive summary of the coaching advice.
        
        Args:
            advice: Full coaching advice text
            chunk_index: Index of current chunk
            
        Returns:
            Detailed coaching summary
        """
        # Extract key coaching points from the advice
        summary_parts = []
        
        # Clean the advice text
        advice_lower = advice.lower()
        
        # Form and Technique section
        form_tips = []
        if "stance" in advice_lower:
            form_tips.append("Maintain proper stance with feet shoulder-width apart")
        if "grip" in advice_lower:
            form_tips.append("Check your grip - hold firmly but not too tight")
        if "posture" in advice_lower or ("back" in advice_lower and "straight" in advice_lower):
            form_tips.append("Keep your back straight and shoulders relaxed")
        if "knee" in advice_lower and "bent" in advice_lower:
            form_tips.append("Keep knees slightly bent for better balance")
        
        if form_tips:
            summary_parts.append("FORM & TECHNIQUE:")
            summary_parts.extend([f"• {tip}" for tip in form_tips])
        
        # Movement and Control section
        movement_tips = []
        if "smooth" in advice_lower:
            movement_tips.append("Focus on smooth, controlled movements")
        if "balance" in advice_lower:
            movement_tips.append("Maintain good balance throughout the motion")
        if "core" in advice_lower:
            movement_tips.append("Engage your core muscles for stability")
        if "follow" in advice_lower and "through" in advice_lower:
            movement_tips.append("Complete your follow-through motion")
        if "timing" in advice_lower:
            movement_tips.append("Work on timing and rhythm")
        
        if movement_tips:
            summary_parts.append("\nMOVEMENT QUALITY:")
            summary_parts.extend([f"• {tip}" for tip in movement_tips])
        
        # Body Positioning section
        positioning_tips = []
        if "alignment" in advice_lower or "aligned" in advice_lower:
            positioning_tips.append("Keep your body properly aligned")
        if "weight" in advice_lower and "balance" in advice_lower:
            positioning_tips.append("Distribute weight evenly on both feet")
        if "shoulder" in advice_lower:
            positioning_tips.append("Keep shoulders level and relaxed")
        if "head" in advice_lower and ("up" in advice_lower or "neutral" in advice_lower):
            positioning_tips.append("Maintain neutral head position")
        
        if positioning_tips:
            summary_parts.append("\nBODY POSITION:")
            summary_parts.extend([f"• {tip}" for tip in positioning_tips])
        
        # Safety and Tips section
        safety_tips = []
        if "safety" in advice_lower or "gear" in advice_lower:
            safety_tips.append("Always wear appropriate safety equipment")
        if "warm" in advice_lower and "up" in advice_lower:
            safety_tips.append("Warm up properly before exercising")
        if "control" in advice_lower:
            safety_tips.append("Stay in control of your movements")
        if "injury" in advice_lower:
            safety_tips.append("Avoid movements that could cause injury")
        
        # Add chunk-specific tips based on progression
        progression_tips = [
            "Start slowly and build up intensity gradually",
            "Focus on consistency over speed",
            "Practice the movement pattern without equipment first",
            "Record yourself to check your form regularly",
            "Take breaks when you feel fatigued"
        ]
        
        if chunk_index < len(progression_tips):
            safety_tips.append(progression_tips[chunk_index])
        
        if safety_tips:
            summary_parts.append("\nSAFETY & TIPS:")
            summary_parts.extend([f"• {tip}" for tip in safety_tips])
        
        # If we didn't extract much, provide a structured fallback
        if not summary_parts:
            fallback_sections = [
                ("FUNDAMENTALS:", ["Focus on proper form", "Maintain good posture", "Control your breathing"]),
                ("TECHNIQUE:", ["Start with basic movements", "Build muscle memory", "Practice consistently"]),
                ("PROGRESSION:", ["Increase difficulty gradually", "Focus on quality over quantity", "Listen to your body"]),
                ("MINDSET:", ["Stay focused and present", "Be patient with progress", "Celebrate small improvements"]),
                ("RECOVERY:", ["Allow adequate rest", "Stay hydrated", "Stretch after exercise"])
            ]
            
            section = fallback_sections[chunk_index % len(fallback_sections)]
            summary_parts.append(section[0])
            summary_parts.extend([f"• {tip}" for tip in section[1]])
        
        # Join all parts
        summary = "\n".join(summary_parts)
        
        # Add a motivational footer
        motivational_quotes = [
            "\n💪 Remember: Progress, not perfection!",
            "\n🎯 Focus on form first, speed second!",
            "\n⭐ Every rep is a step toward improvement!",
            "\n🔥 Consistency beats intensity!",
            "\n🏆 You're building strength with every movement!"
        ]
        
        summary += motivational_quotes[chunk_index % len(motivational_quotes)]
        
        return summary
        
    
    def create_side_panel(self, text: str, video_size: Tuple[int, int], duration: float) -> VideoFileClip:
        """
        Create a side panel with text instead of overlay.
        
        Args:
            text: Text to display
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
        
        # Try to use fonts of different sizes for headers and body text
        try:
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 22)
            body_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 18)
        except:
            header_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
        
        # Split text into lines and process
        lines = text.split('\n')
        y = 30  # Start 30px from top
        margin = 20  # Left margin
        max_width = panel_width - (margin * 2)  # Account for both margins
        
        for line in lines:
            if not line.strip():
                y += 10  # Small gap for empty lines
                continue
            
            # Check if this is a header (ends with colon and is all caps)
            is_header = line.strip().endswith(':') and any(c.isupper() for c in line.strip())
            current_font = header_font if is_header else body_font
            
            # Use different colors for headers
            color = 'yellow' if is_header else 'white'
            
            # Word wrap for long lines
            if len(line) > 60 or draw.textbbox((0, 0), line, font=current_font)[2] > max_width:
                words = line.split()
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    bbox = draw.textbbox((0, 0), test_line, font=current_font)
                    
                    if bbox[2] <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            draw.text((margin, y), current_line, fill=color, font=current_font)
                            y += 25
                        current_line = word
                
                if current_line:
                    draw.text((margin, y), current_line, fill=color, font=current_font)
                    y += 25
            else:
                draw.text((margin, y), line, fill=color, font=current_font)
                y += 25
            
            # Add extra spacing after headers
            if is_header:
                y += 5
            
            # Stop if we're running out of space
            if y > height - 50:
                break
        
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
            logger.info(f"Extracted {len(chunks)} chunks")
            
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
                summary = self.summarize_advice(advice, i)
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
            
            # Save final coached video
            output_path = video_path.replace('.mp4', '_coached.mp4')
            final_video.write_videofile(
                output_path,
                fps=self.target_fps,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=os.path.join(temp_dir, 'temp-audio.m4a'),
                remove_temp=True,
                logger=None  # Suppress moviepy logging
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