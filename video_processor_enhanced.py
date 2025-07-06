# Enhanced Video processing module for MyoCouch using Qwen2-VL
import cv2
import numpy as np
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import tempfile
import ffmpeg
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, ImageClip
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import supervision as sv
from dataclasses import dataclass
from collections import defaultdict
import gc

logger = logging.getLogger('MyoCouch.VideoProcessor')


@dataclass
class PersonInfo:
    """Information about a detected person."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    gender: Optional[str] = None
    track_id: Optional[int] = None


@dataclass
class ExerciseContext:
    """Context about the exercise being performed."""
    exercise_type: str
    movement_phase: str
    key_points: List[str]
    person_info: Optional[PersonInfo] = None


class EnhancedVideoCoachingProcessor:
    """Enhanced video processor with agentic capabilities."""
    
    def __init__(self, model_size: str = "7B"):
        """Initialize the enhanced video processor."""
        self.model_size = model_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and processor to None
        self.model = None
        self.processor = None
        
        # Load Qwen2-VL model with the correct imports
        model_mapping = {
            "2B": "Qwen/Qwen2-VL-2B-Instruct",
            "7B": "Qwen/Qwen2-VL-7B-Instruct"
        }
        
        model_path = model_mapping.get(model_size, model_mapping["7B"])
        logger.info(f"Loading Qwen2-VL model: {model_path}")
        
        try:
            # Use the specific Qwen2VL classes
            self.processor = Qwen2VLProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            logger.info(f"Successfully loaded model: {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            
            # Try fallback to 2B model if we were trying 7B
            if model_size == "7B":
                logger.info("Attempting fallback to 2B model...")
                self.model_size = "2B"
                model_path = model_mapping["2B"]
                
                try:
                    self.processor = Qwen2VLProcessor.from_pretrained(model_path, trust_remote_code=True)
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True
                    )
                    
                    if not torch.cuda.is_available():
                        self.model = self.model.to(self.device)
                    
                    logger.info(f"Successfully loaded fallback model: {model_path}")
                    
                except Exception as fallback_error:
                    logger.error(f"Failed to load fallback model: {fallback_error}")
                    raise RuntimeError(f"Could not load any AI vision model. Original error: {e}, Fallback error: {fallback_error}")
            else:
                # If we're already on 2B or model loading failed completely
                raise RuntimeError(f"Could not load AI vision model: {e}")
        
        # Verify model and processor are loaded
        if self.model is None or self.processor is None:
            raise RuntimeError("AI vision model failed to initialize properly")
        
        # Load YOLO for human detection
        logger.info("Loading YOLO for human detection...")
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Nano model for speed
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.yolo_model = None
        
        # Initialize trackers and annotators
        self.byte_tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
        
        # OPTIMIZED: Balanced settings for quality and memory efficiency
        self.target_fps = 24  # Increased from 15 for smoother video
        self.chunk_size = 30  # Keep smaller chunks for memory
        self.frames_per_analysis = 4  # Keep at 4 for memory efficiency
        self.max_resolution = (1280, 720)  # Increased from 640x480 for better quality
        self.analysis_resolution = (640, 480)  # Lower resolution only for AI analysis
        
        # Coaching memory - track what advice has been given
        self.coaching_history = []
        self.exercise_type = None
        self.person_gender = None
        
    def resize_frame_if_needed(self, frame: np.ndarray, target_resolution: tuple = None) -> np.ndarray:
        """Resize frame if it exceeds maximum resolution."""
        height, width = frame.shape[:2]
        max_width, max_height = target_resolution or self.max_resolution
        
        if width > max_width or height > max_height:
            # Calculate scaling factor
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize frame
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
        return frame
    
    def clear_gpu_memory(self):
        """Aggressively clear GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
    
    def identify_exercise(self, frames: List[Image.Image]) -> str:
        """Identify the type of exercise being performed."""
        prompt = """Analyze these video frames and identify the specific exercise being performed.
        Be precise and specific (e.g., "barbell back squat", "dumbbell bicep curl", "push-up", etc.).
        Respond with ONLY the exercise name, nothing else."""
        
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What exercise is this?"},
                ] + [{"type": "image"} for _ in frames]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process with half precision
        with torch.cuda.amp.autocast():
            inputs = self.processor(
                text=text,
                images=frames,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.3,
                    do_sample=True
                )
            
            # Move outputs to CPU immediately
            outputs = outputs.cpu()
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Clear GPU memory after inference
        del inputs, outputs
        self.clear_gpu_memory()
        
        exercise = response.split("What exercise is this?")[-1].strip()
        return exercise.replace("Assistant:", "").strip()
    
    def identify_gender(self, frames: List[Image.Image], person_bbox: Optional[Tuple] = None) -> str:
        """Identify the gender of the person exercising."""
        prompt = """Look at the person in these frames and identify their apparent gender based on visual cues.
        Respond with ONLY one word: "male", "female", or "unknown"."""
        
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the person's gender?"},
                ] + [{"type": "image"} for _ in frames]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process with half precision
        with torch.cuda.amp.autocast():
            inputs = self.processor(
                text=text,
                images=frames,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.3,
                    do_sample=True
                )
            
            # Move outputs to CPU immediately
            outputs = outputs.cpu()
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Clear GPU memory after inference
        del inputs, outputs
        self.clear_gpu_memory()
        
        gender = response.split("What is the person's gender?")[-1].strip().lower()
        gender = gender.replace("assistant:", "").strip()
        
        if gender in ["male", "female"]:
            return gender
        return "unknown"
    
    def detect_humans(self, frame: np.ndarray) -> List[PersonInfo]:
        """Detect humans in a frame using YOLO."""
        # Return empty list if YOLO failed to load
        if self.yolo_model is None:
            logger.warning("YOLO model not available, skipping human detection")
            return []
            
        try:
            # Resize frame for YOLO if needed
            resized_frame = self.resize_frame_if_needed(frame, self.analysis_resolution)
            
            results = self.yolo_model(resized_frame, classes=[0], conf=0.5)  # class 0 is person
            
            people = []
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Scale coordinates back to original frame size if resized
                    if resized_frame.shape != frame.shape:
                        scale_x = frame.shape[1] / resized_frame.shape[1]
                        scale_y = frame.shape[0] / resized_frame.shape[0]
                        x1, x2 = x1 * scale_x, x2 * scale_x
                        y1, y2 = y1 * scale_y, y2 * scale_y
                    
                    people.append(PersonInfo(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(confidence)
                    ))
            
            return people
        except Exception as e:
            logger.error(f"Error in human detection: {e}")
            return []
    
    def get_movement_phase(self, chunk_index: int, total_chunks: int) -> str:
        """Determine the movement phase based on chunk position."""
        position = chunk_index / total_chunks
        
        if position < 0.2:
            return "starting position"
        elif position < 0.4:
            return "early movement"
        elif position < 0.6:
            return "mid-movement"
        elif position < 0.8:
            return "late movement"
        else:
            return "finishing position"
    
    def generate_contextual_advice(self, chunk: np.ndarray, chunk_index: int, total_chunks: int) -> str:
        """Generate context-aware coaching advice that varies throughout the video."""
        # Sample frames from the chunk
        frames = self.sample_frames_from_chunk(chunk, self.frames_per_analysis)
        
        # Get movement phase
        phase = self.get_movement_phase(chunk_index, total_chunks)
        
        # Create coaching history context
        history_context = ""
        if self.coaching_history:
            recent_advice = self.coaching_history[-3:]  # Last 3 pieces of advice
            history_context = f"\nPrevious advice given: {'; '.join(recent_advice)}"
        
        # Create exercise-specific prompt
        exercise_context = f"Exercise: {self.exercise_type}" if self.exercise_type else ""
        gender_context = f"Person: {self.person_gender}" if self.person_gender else ""
        
        prompt = f"""You are MyoCouch, analyzing a {phase} of a workout video.
        {exercise_context}
        {gender_context}
        
        Provide specific, actionable coaching for THIS phase of the movement.
        Focus on what's happening RIGHT NOW in these frames.
        {history_context}
        
        IMPORTANT: Give NEW advice different from previous segments. Be specific to this movement phase.
        Keep response under 50 words. Be encouraging but precise."""
        
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Coach this {phase}:"},
                ] + [{"type": "image"} for _ in frames]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process with half precision
        with torch.cuda.amp.autocast():
            inputs = self.processor(
                text=text,
                images=frames,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9
                )
            
            # Move outputs to CPU immediately
            outputs = outputs.cpu()
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Clear GPU memory after inference
        del inputs, outputs
        self.clear_gpu_memory()
        
        advice = response.split(f"Coach this {phase}:")[-1].strip()
        advice = advice.replace("Assistant:", "").strip()
        
        # Store in history
        self.coaching_history.append(advice)
        
        return advice
    
    def create_annotated_frame(self, frame: np.ndarray, people: List[PersonInfo]) -> np.ndarray:
        """Create an annotated frame with arrows pointing to detected people."""
        annotated = frame.copy()
        
        # Convert to PIL for easier drawing
        pil_image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        for i, person in enumerate(people):
            x1, y1, x2, y2 = person.bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Draw arrow pointing to person
            arrow_start_x = cx - 50
            arrow_start_y = y1 - 50
            
            # Draw arrow line
            draw.line([(arrow_start_x, arrow_start_y), (cx, y1)], fill="yellow", width=3)
            
            # Draw arrow head
            arrow_points = [
                (cx, y1),
                (cx - 10, y1 - 10),
                (cx + 10, y1 - 10)
            ]
            draw.polygon(arrow_points, fill="yellow")
            
            # Add label
            label = f"Person {i+1}"
            if person.gender:
                label += f" ({person.gender})"
            
            # Draw text background
            text_bbox = draw.textbbox((arrow_start_x - 20, arrow_start_y - 20), label)
            draw.rectangle(text_bbox, fill="yellow")
            draw.text((arrow_start_x - 20, arrow_start_y - 20), label, fill="black")
        
        # Convert back to numpy
        annotated = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return annotated
    
    def process_video_with_intelligence(self, video_path: str) -> Dict:
        """Process video with enhanced intelligence and visual annotations."""
        logger.info(f"Processing video with enhanced intelligence: {video_path}")
        
        # Reset coaching memory
        self.coaching_history = []
        self.exercise_type = None
        self.person_gender = None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Downsample video
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
            logger.info("Extracting video chunks...")
            chunks = self.extract_video_chunks(downsampled_path)
            logger.info(f"Extracted {len(chunks)} chunks")
            
            # Step 3: Initial analysis - identify exercise and gender
            if chunks:
                logger.info("Identifying exercise type...")
                first_chunk_frames = self.sample_frames_from_chunk(chunks[0], 4)
                self.exercise_type = self.identify_exercise(first_chunk_frames)
                logger.info(f"Exercise identified: {self.exercise_type}")
                
                # Detect person in first frame and identify gender
                first_frame = chunks[0][0]
                people = self.detect_humans(first_frame)
                if people:
                    logger.info("Identifying person's gender...")
                    self.person_gender = self.identify_gender(first_chunk_frames)
                    logger.info(f"Gender identified: {self.person_gender}")
                    # Update person info
                    people[0].gender = self.person_gender
            
            # Step 4: Process each chunk with enhanced analysis
            chunk_clips = []
            all_advice = []
            
            for i, chunk in enumerate(tqdm(chunks, desc="Analyzing chunks")):
                # Get contextual coaching advice
                logger.info(f"Analyzing chunk {i+1}/{len(chunks)}...")
                advice = self.generate_contextual_advice(chunk, i, len(chunks))
                phase = self.get_movement_phase(i, len(chunks))
                all_advice.append(f"{phase.title()}: {advice}")
                
                # Create annotated frames with human detection
                annotated_frames = []
                
                # Process every 10th frame for efficiency
                for j in range(0, len(chunk), 10):
                    frame = chunk[j]
                    people = self.detect_humans(frame)
                    
                    # Update gender for first detected person
                    if people and self.person_gender:
                        people[0].gender = self.person_gender
                    
                    # Create annotated frame
                    annotated = self.create_annotated_frame(frame, people)
                    
                    # Repeat frame 10 times to maintain timing
                    for _ in range(10):
                        if len(annotated_frames) < len(chunk):
                            annotated_frames.append(annotated)
                
                # Ensure we have the right number of frames
                annotated_frames = annotated_frames[:len(chunk)]
                
                # Save annotated chunk as video
                chunk_path = os.path.join(temp_dir, f"chunk_{i}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(chunk_path, fourcc, self.target_fps, (width, height))
                
                for frame in annotated_frames:
                    out.write(frame)
                out.release()
                
                # Create video clip with overlay
                video_clip = VideoFileClip(chunk_path)
                
                # Create enhanced text overlay with exercise info
                overlay_text = f"{self.exercise_type.upper()} - {phase.upper()}\n{advice}"
                text_overlay = self.create_enhanced_overlay(overlay_text, (width, height), len(chunk) / self.target_fps)
                
                # Composite video with text overlay
                composite = CompositeVideoClip([video_clip, text_overlay])
                chunk_clips.append(composite)
            
            # Step 5: Concatenate all chunks
            logger.info("Concatenating coached video segments...")
            final_video = concatenate_videoclips(chunk_clips, method="compose")
            
            # Save final coached video with better error handling
            output_path = video_path.replace('.mp4', '_coached_enhanced.mp4')
            
            try:
                # Check if the original video has audio
                original_clip = VideoFileClip(video_path)
                has_audio = original_clip.audio is not None
                
                # If original has audio, add it to the final video
                if has_audio:
                    final_video = final_video.set_audio(original_clip.audio)
                
                original_clip.close()
                
                logger.info(f"Writing final video (audio: {has_audio})...")
                
                # Write video with high quality settings
                final_video.write_videofile(
                    output_path,
                    fps=self.target_fps,
                    codec='libx264',
                    audio_codec='aac' if has_audio else None,
                    audio_bitrate='192k' if has_audio else None,
                    bitrate='5000k',  # Higher bitrate for better quality
                    preset='slow',  # Better compression quality
                    ffmpeg_params=['-crf', '18'],  # High quality constant rate factor
                    temp_audiofile=os.path.join(temp_dir, 'temp-audio.m4a') if has_audio else None,
                    remove_temp=True,
                    threads=4,
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
            
            # Prepare results - remove specific model references
            results = {
                'status': 'success',
                'video_info': {
                    'duration_seconds': total_frames / fps,
                    'fps': fps,
                    'resolution': (width, height),
                    'total_frames': total_frames,
                    'chunks_processed': len(chunks)
                },
                'exercise_analysis': {
                    'exercise_type': self.exercise_type,
                    'person_gender': self.person_gender,
                    'total_segments': len(chunks)
                },
                'coaching_segments': all_advice,
                'output_video_path': output_path,
                'model_used': 'Advanced AI Vision Model'  # Generic name instead of Qwen2-VL
            }
            
            logger.info(f"Enhanced video processing complete. Output saved to: {output_path}")
            return results
    
    def create_enhanced_overlay(self, text: str, video_size: Tuple[int, int], duration: float) -> TextClip:
        """Create an enhanced text overlay with better styling."""
        width, height = video_size
        
        # Calculate font size based on video resolution
        base_font_size = max(20, int(height * 0.04))  # Minimum 20px, 4% of height
        
        # Create text with better quality settings
        txt_clip = TextClip(
            text,
            fontsize=base_font_size,
            font='Arial-Bold',
            color='white',
            bg_color='rgba(0,0,0,200)',  # Slightly more opaque background
            size=(int(width * 0.85), None),  # 85% of video width
            method='caption',
            align='center',
            stroke_color='black',
            stroke_width=2,  # Thicker stroke for better readability
            kerning=-1  # Tighter letter spacing
        ).set_duration(duration)
        
        # Position at bottom with padding
        txt_clip = txt_clip.set_position(('center', int(height * 0.75)))
        
        # Add fade effects
        txt_clip = txt_clip.crossfadein(0.3).crossfadeout(0.3)
        
        return txt_clip
    
    def downsample_video(self, input_path: str, output_path: str, target_fps: int = 24) -> str:
        """Downsample video to target FPS and optionally resize."""
        try:
            # Get input video info
            probe = ffmpeg.probe(input_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])
            
            # Calculate if resizing is needed
            max_width, max_height = self.max_resolution
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                # Ensure even dimensions for video encoding
                new_width = new_width - (new_width % 2)
                new_height = new_height - (new_height % 2)
                
                logger.info(f"Resizing video from {width}x{height} to {new_width}x{new_height}")
                
                (
                    ffmpeg
                    .input(input_path)
                    .filter('fps', fps=target_fps)
                    .filter('scale', new_width, new_height)
                    .output(output_path, 
                           video_bitrate='3M',  # Increased bitrate for better quality
                           vcodec='libx264',
                           preset='medium',
                           crf=23)  # Constant Rate Factor for quality
                    .overwrite_output()
                    .run(quiet=True)
                )
            else:
                (
                    ffmpeg
                    .input(input_path)
                    .filter('fps', fps=target_fps)
                    .output(output_path, 
                           video_bitrate='3M',
                           vcodec='libx264',
                           preset='medium',
                           crf=23)
                    .overwrite_output()
                    .run(quiet=True)
                )
            
            return output_path
        except Exception as e:
            logger.error(f"Error downsampling video: {e}")
            raise
    
    def extract_video_chunks(self, video_path: str) -> List[np.ndarray]:
        """Extract video chunks of specified frame count."""
        cap = cv2.VideoCapture(video_path)
        chunks = []
        current_chunk = []
        
        # Process video in smaller batches to save memory
        batch_size = 5  # Process 5 chunks at a time
        
        while True:
            ret, frame = cap.read()
            if not ret:
                if current_chunk:
                    chunks.append(np.array(current_chunk))
                break
            
            # Keep original frame quality - no resizing here
            current_chunk.append(frame)
            
            if len(current_chunk) >= self.chunk_size:
                chunks.append(np.array(current_chunk))
                current_chunk = []
                
                # Clear memory periodically
                if len(chunks) % batch_size == 0:
                    gc.collect()
        
        cap.release()
        return chunks
    
    def sample_frames_from_chunk(self, chunk: np.ndarray, num_frames: int = 4) -> List[Image.Image]:
        """Sample frames uniformly from a video chunk."""
        total_frames = len(chunk)
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            frame = chunk[idx]
            # Note: chunk frames are already in BGR format from OpenCV
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame for AI analysis only
            analysis_frame = self.resize_frame_if_needed(frame_rgb, self.analysis_resolution)
            pil_image = Image.fromarray(analysis_frame)
            
            frames.append(pil_image)
        
        return frames 