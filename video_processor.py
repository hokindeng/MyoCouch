# Video processing module for MyoCouch using Qwen2-VL
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import tempfile
import ffmpeg
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
import os
from tqdm import tqdm
from PIL import Image

logger = logging.getLogger('MyoCouch.VideoProcessor')


class VideoCoachingProcessor:
    """Processes videos for coaching analysis using Qwen2-VL Video Language Model."""
    
    def __init__(self, model_size: str = "7B"):
        """
        Initialize the video processor with Qwen2-VL.
        
        Args:
            model_size: Model size - "2B" or "7B" (default)
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
        
        try:
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to 2B model if 7B fails
            if model_size == "7B":
                logger.info("Falling back to 2B model...")
                self.model_size = "2B"
                model_path = model_mapping["2B"]
                self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
        
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
    
    def analyze_video_chunk(self, chunk: np.ndarray) -> str:
        """
        Analyze a video chunk using Qwen2-VL and return coaching advice.
        
        Args:
            chunk: Video chunk as numpy array (frames, height, width, channels)
            
        Returns:
            Coaching advice text
        """
        try:
            # Sample frames from the chunk
            frames = self.sample_frames_from_chunk(chunk, self.frames_per_analysis)
            
            # Create the conversation format for Qwen2-VL
            messages = [
                {
                    "role": "system",
                    "content": self.coaching_prompt
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
                    max_new_tokens=150,
                    temperature=0.7,
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
            advice = advice.replace("Assistant:", "").strip()
            
            return advice
            
        except Exception as e:
            logger.error(f"Error analyzing chunk: {e}")
            return "Focus on maintaining proper form throughout the movement."
    
    def create_overlay_text(self, text: str, video_size: Tuple[int, int], duration: float) -> TextClip:
        """
        Create a text overlay for video.
        
        Args:
            text: Text to overlay
            video_size: (width, height) of video
            duration: Duration of the clip
            
        Returns:
            TextClip object
        """
        width, height = video_size
        
        # Format text for better readability
        lines = text.split('. ')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.endswith('.'):
                line += '.'
            # Wrap long lines
            if len(line) > 60:
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + word) < 60:
                        current_line += word + " "
                    else:
                        if current_line:
                            formatted_lines.append(current_line.strip())
                        current_line = word + " "
                if current_line:
                    formatted_lines.append(current_line.strip())
            elif line:
                formatted_lines.append(line)
        
        formatted_text = '\n'.join(formatted_lines[:4])  # Limit to 4 lines
        
        # Create text clip with semi-transparent background
        txt_clip = TextClip(
            formatted_text,
            fontsize=int(height * 0.03),  # 3% of video height
            font='Arial',
            color='white',
            bg_color='black',
            size=(int(width * 0.9), None),  # 90% of video width
            method='caption',
            align='center'
        ).set_duration(duration)
        
        # Position at bottom of video
        txt_clip = txt_clip.set_position(('center', int(height * 0.85)))
        
        # Add fade in/out effect
        txt_clip = txt_clip.crossfadein(0.5).crossfadeout(0.5)
        
        return txt_clip
    
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
            
            for i, chunk in enumerate(tqdm(chunks, desc="Analyzing chunks")):
                # Get coaching advice for this chunk
                logger.info(f"Analyzing chunk {i+1}/{len(chunks)}...")
                advice = self.analyze_video_chunk(chunk)
                all_advice.append(f"Segment {i+1}: {advice}")
                
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
                
                # Create video clip with overlay
                video_clip = VideoFileClip(chunk_path)
                text_overlay = self.create_overlay_text(advice, (width, height), chunk_duration)
                
                # Composite video with text overlay
                composite = CompositeVideoClip([video_clip, text_overlay])
                chunk_clips.append(composite)
            
            # Step 4: Concatenate all chunks
            logger.info("Concatenating coached video segments...")
            final_video = concatenate_videoclips(chunk_clips)
            
            # Save final coached video with better error handling
            output_path = video_path.replace('.mp4', '_coached.mp4')
            
            try:
                # Check if the original video has audio
                original_clip = VideoFileClip(video_path)
                has_audio = original_clip.audio is not None
                original_clip.close()
                
                logger.info(f"Writing final video (audio: {has_audio})...")
                
                # Write video with appropriate settings
                if has_audio:
                    final_video.write_videofile(
                        output_path,
                        fps=self.target_fps,
                        codec='libx264',
                        audio_codec='aac',
                        audio_bitrate='128k',
                        temp_audiofile=os.path.join(temp_dir, 'temp-audio.m4a'),
                        remove_temp=True,
                        threads=4,  # Use multiple threads for stability
                        preset='medium',  # Balance between speed and compression
                        logger=None  # Suppress moviepy logging
                    )
                else:
                    # Write without audio if original has no audio
                    final_video.write_videofile(
                        output_path,
                        fps=self.target_fps,
                        codec='libx264',
                        audio=False,  # No audio track
                        threads=4,
                        preset='medium',
                        logger=None
                    )
                    
            except Exception as e:
                logger.error(f"Error writing video file: {e}")
                # Try alternative approach with different settings
                try:
                    logger.info("Attempting alternative video encoding...")
                    final_video.write_videofile(
                        output_path,
                        fps=self.target_fps,
                        codec='mpeg4',  # More compatible codec
                        audio_codec='mp3' if has_audio else None,
                        audio=has_audio,
                        bitrate='2000k',
                        threads=2,
                        logger=None
                    )
                except Exception as fallback_error:
                    logger.error(f"Alternative encoding also failed: {fallback_error}")
                    raise Exception(f"Could not encode video: {str(e)}")
            
            finally:
                # Clean up resources
                try:
                    final_video.close()
                    for clip in chunk_clips:
                        clip.close()
                except:
                    pass
            
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
                'model_used': f'Qwen2-VL-{self.model_size}'
            }
            
            logger.info(f"Video processing complete. Output saved to: {output_path}")
            return results 