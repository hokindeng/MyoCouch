# MyoCouch - Discord bot for video coaching with AI vision model
import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import logging
from pathlib import Path
import asyncio
from typing import Optional
from video_processor import VideoCoachingProcessor
from tempfile import NamedTemporaryFile

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DEBUG_GUILD_ID = os.getenv("DEBUG_GUILD_ID")
MODEL_SIZE = os.getenv("MODEL_SIZE", "2B")  # Default to 2B

if not TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN missing from environment or .env file.")

# Configure intents
intents = discord.Intents.default()
intents.message_content = True

# Initialize bot
bot = discord.Bot(intents=intents)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MyoCouch')

# Initialize video processor
logger.info(f"Initializing AI vision model with size: {MODEL_SIZE}")
video_processor = VideoCoachingProcessor(model_size=MODEL_SIZE)

# Video coaching slash command
@bot.slash_command(
    name="couch",
    description="Upload a video to receive AI-powered coaching with advice overlaid on the video"
)
async def couch_command(
    ctx: discord.ApplicationContext,
    video: discord.Option(
        discord.Attachment,
        description="Upload a workout video for AI coaching analysis (max 25MB)",
        required=True
    )
):
    """Process uploaded video and provide coaching advice overlaid on the video."""
    global video_processor  # Declare global at the beginning of the function
    
    # Defer the response since video processing will take time
    await ctx.defer()
    
    # Validate file type
    if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv')):
        await ctx.followup.send("❌ Please upload a valid video file (MP4, MOV, AVI, WEBM, or MKV)")
        return
    
    # Check file size (Discord limit is 25MB for free servers)
    if video.size > 25 * 1024 * 1024:
        await ctx.followup.send("❌ Video file is too large. Please upload a video under 25MB.")
        return
    
    try:
        # Send initial processing message
        embed = discord.Embed(
            title="🎬 Processing Your Video with AI",
            description=f"**Step 1/4:** Downloading video...\n"
                       f"Model: AI Vision-{MODEL_SIZE}",
            color=discord.Color.blue()
        )
        embed.set_footer(text="This may take a few minutes depending on video length")
        processing_msg = await ctx.followup.send(embed=embed)
        
        # Download the video
        video_data = await video.read()
        
        # Save video temporarily
        with NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_data)
            tmp_path = tmp_file.name
        
        # Check video duration
        try:
            import cv2
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            if duration > 24:  # More than 24 seconds (6 chunks * 4 seconds each)
                await ctx.followup.send(
                    f"⚠️ Your video is {duration:.1f} seconds long. Only the first 24 seconds will be analyzed "
                    f"(6 segments × 4 seconds each). For complete analysis, use videos under 24 seconds."
                )
        except:
            pass  # Continue even if duration check fails
        
        # Update status
        embed.description = f"**Step 2/4:** Analyzing video with AI...\n" \
                          f"Model: AI Vision-{MODEL_SIZE}"
        await processing_msg.edit(embed=embed)
        
        # Process the video in a separate thread
        loop = asyncio.get_event_loop()
        
        # Create a task for video processing
        import time
        start_time = time.time()
        last_update = start_time
        
        async def process_with_updates():
            """Process video while sending periodic updates to Discord."""
            nonlocal last_update  # Access the outer scope variable
            
            # Run processing in executor
            future = loop.run_in_executor(
                None,
                video_processor.process_video,
                tmp_path
            )
            
            # Send updates while processing
            update_messages = [
                "🔄 AI is analyzing your movements...",
                "🤖 Processing video segments...",
                "✨ Creating coaching overlays...",
                "📝 Generating personalized advice...",
                "🎬 Finalizing your coached video..."
            ]
            
            message_index = 0
            while not future.done():
                await asyncio.sleep(10)  # Check every 10 seconds
                
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Update every 30 seconds to prevent timeout
                if current_time - last_update > 30:
                    update_msg = update_messages[message_index % len(update_messages)]
                    embed.description = f"**Step 2/4:** {update_msg}\n" \
                                      f"Time elapsed: {elapsed:.0f} seconds\n" \
                                      f"Model: AI Vision-{MODEL_SIZE}"
                    
                    try:
                        await processing_msg.edit(embed=embed)
                        last_update = current_time
                        message_index += 1
                    except discord.HTTPException:
                        logger.warning("Failed to update progress message")
                
                # Timeout after 5 minutes (reduced from 10 since max 6 chunks)
                if elapsed > 300:
                    raise Exception("Processing timeout - video too long or complex")
            
            # Get the result
            return await future
        
        try:
            coaching_result = await process_with_updates()
        except asyncio.TimeoutError:
            raise Exception("Processing timeout - Discord connection lost")
        
        # Check if processing was successful
        if coaching_result['status'] != 'success':
            raise Exception("Video processing failed")
        
        # Update status
        embed.description = f"**Step 3/4:** Creating coached video with overlays...\n" \
                          f"Model: {coaching_result.get('model_used', 'AI Vision')}"
        await processing_msg.edit(embed=embed)
        
        # Print full coaching segments to console
        logger.info("=== Full Coaching Analysis ===")
        for segment in coaching_result['coaching_segments']:
            logger.info(segment)
        logger.info("=============================")
        
        # Create local directory for coached videos if it doesn't exist
        output_dir = Path("coached_videos")
        output_dir.mkdir(exist_ok=True)
        
        # Copy the video to local directory with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_filename = f"{timestamp}_coached_{video.filename}"
        local_path = output_dir / local_filename
        
        # Copy the temporary coached video to local directory
        import shutil
        temp_output_path = coaching_result['output_video_path']
        shutil.copy2(temp_output_path, local_path)
        logger.info(f"Coached video saved locally: {local_path}")
        
        # Copy the memory file to local directory
        if 'memory_file_path' in coaching_result:
            memory_filename = f"{timestamp}_memory_{video.filename.replace('.mp4', '.json')}"
            local_memory_path = output_dir / memory_filename
            shutil.copy2(coaching_result['memory_file_path'], local_memory_path)
            logger.info(f"Coaching memory saved locally: {local_memory_path}")
        
        # Create success embed
        success_embed = discord.Embed(
            title="✅ MyoCouch Analysis Complete!",
            description=f"Your video has been analyzed and coaching advice has been overlaid on the video.",
            color=discord.Color.green()
        )
        
        # Add video stats
        video_info = coaching_result['video_info']
        duration_text = f"• Duration: {video_info['duration_seconds']:.1f} seconds"
        if video_info.get('video_was_cut', False):
            duration_text += f" (cut from {video_info.get('original_duration_seconds', 0):.1f}s)"
        
        success_embed.add_field(
            name="📊 Video Statistics",
            value=duration_text + f"\n"
                  f"• Resolution: {video_info['resolution'][0]}x{video_info['resolution'][1]}\n"
                  f"• Segments analyzed: {video_info['chunks_processed']}",
            inline=False
        )
        
        # Add summary of coaching segments
        segments = coaching_result['coaching_segments']
        if segments:
            # Clean and format segments for display
            cleaned_segments = []
            for seg in segments:
                # Remove "Segment X: assistant" prefix and clean up
                if ": assistant" in seg:
                    seg = seg.split(": assistant", 1)[1].strip()
                elif "assistant\n" in seg:
                    seg = seg.split("assistant\n", 1)[1].strip()
                elif "assistant" in seg and seg.startswith("Segment"):
                    parts = seg.split(":", 1)
                    if len(parts) > 1:
                        seg = parts[1].replace("assistant", "").strip()
                
                # Truncate to first sentence for preview
                first_sentence = seg.split('.')[0] + '.' if '.' in seg else seg[:100] + '...'
                cleaned_segments.append(first_sentence)
            
            # Show first 3 segments as preview
            preview_segments = cleaned_segments[:3]
            preview_text = '\n'.join([f"• {seg}" for seg in preview_segments])
            if len(segments) > 3:
                preview_text += f"\n... and {len(segments) - 3} more segments"
            
            success_embed.add_field(
                name="💪 Coaching Preview",
                value=preview_text[:1024],  # Discord field limit
                inline=False
            )
        
        success_embed.add_field(
            name="🤖 AI Model Used",
            value=coaching_result.get('model_used', 'AI Vision'),
            inline=True
        )
        
        success_embed.set_footer(text="Keep training! Upload another video to track your progress.")
        
        # Update the message
        await processing_msg.edit(embed=success_embed)
        
        # Send the coached video
        # Check if the output video is within Discord's file size limit
        output_size = os.path.getsize(local_path)
        if output_size <= 25 * 1024 * 1024:  # 25MB limit
            # Send the coached video
            with open(local_path, 'rb') as f:
                coached_video = discord.File(f, filename=f"coached_{video.filename}")
                await ctx.followup.send(
                    "Here's your video with AI coaching advice overlaid! 🎥",
                    file=coached_video
                )
        else:
            # Video too large, provide alternative
            await ctx.followup.send(
                "⚠️ The coached video is too large to upload to Discord. "
                "Consider using a shorter video or reducing quality."
            )
        
        # Clean up temporary files
        try:
            os.remove(tmp_path)
            # Remove only the temporary output, not the local copy
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            # Remove temporary memory file
            if 'memory_file_path' in coaching_result and os.path.exists(coaching_result['memory_file_path']):
                os.remove(coaching_result['memory_file_path'])
        except:
            pass
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        error_embed = discord.Embed(
            title="❌ Processing Error",
            description=f"Sorry, I couldn't process your video.\n\n"
                       f"**Error:** {str(e)}\n\n"
                       f"**Tips:**\n"
                       f"• Try a shorter video (under 30 seconds works best)\n"
                       f"• Ensure good lighting and clear visibility\n"
                       f"• Make sure the exercise is clearly visible",
            color=discord.Color.red()
        )
        await processing_msg.edit(embed=error_embed)
        
        # Clean up temporary file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


# Motion comparison slash command
@bot.slash_command(
    name="motiondiff",
    description="Compare two workout videos side-by-side to see which has better form"
)
async def motiondiff_command(
    ctx: discord.ApplicationContext,
    video1: discord.Option(
        discord.Attachment,
        description="First workout video for comparison (max 25MB)",
        required=True
    ),
    video2: discord.Option(
        discord.Attachment,
        description="Second workout video for comparison (max 25MB)",
        required=True
    )
):
    """Compare two videos side by side and analyze which has better form."""
    global video_processor
    
    # Defer the response
    await ctx.defer()
    
    # Validate file types
    for video in [video1, video2]:
        if not video.filename.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv')):
            await ctx.followup.send(f"❌ {video.filename} is not a valid video file. Please upload MP4, MOV, AVI, WEBM, or MKV files.")
            return
    
    # Check file sizes
    for video in [video1, video2]:
        if video.size > 25 * 1024 * 1024:
            await ctx.followup.send(f"❌ {video.filename} is too large. Please upload videos under 25MB.")
            return
    
    try:
        # Send initial processing message
        embed = discord.Embed(
            title="🎬 Comparing Your Videos",
            description=f"**Step 1/3:** Downloading videos...\n"
                       f"Model: AI Vision-{MODEL_SIZE}",
            color=discord.Color.blue()
        )
        embed.set_footer(text="This will analyze the first 5 seconds of each video")
        processing_msg = await ctx.followup.send(embed=embed)
        
        # Download both videos
        video1_data = await video1.read()
        video2_data = await video2.read()
        
        # Save videos temporarily
        with NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file1:
            tmp_file1.write(video1_data)
            tmp_path1 = tmp_file1.name
            
        with NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file2:
            tmp_file2.write(video2_data)
            tmp_path2 = tmp_file2.name
        
        # Update status
        embed.description = f"**Step 2/3:** Creating side-by-side comparison...\n" \
                          f"Model: AI Vision-{MODEL_SIZE}"
        await processing_msg.edit(embed=embed)
        
        # Process the motion comparison
        loop = asyncio.get_event_loop()
        comparison_result = await loop.run_in_executor(
            None,
            video_processor.compare_motions,
            tmp_path1,
            tmp_path2
        )
        
        # Check if processing was successful
        if comparison_result['status'] != 'success':
            raise Exception("Motion comparison failed")
        
        # Update status
        embed.description = f"**Step 3/3:** AI is analyzing the differences...\n" \
                          f"Model: {comparison_result.get('model_used', 'AI Vision')}"
        await processing_msg.edit(embed=embed)
        
        # Create success embed
        success_embed = discord.Embed(
            title="✅ Motion Comparison Complete!",
            description="Your videos have been compared side-by-side with AI analysis.",
            color=discord.Color.green()
        )
        
        # Add comparison results
        success_embed.add_field(
            name="🏆 AI Verdict",
            value=comparison_result['verdict'],
            inline=False
        )
        
        success_embed.add_field(
            name="📊 Video 1 Analysis",
            value=comparison_result['video1_analysis'][:1024],
            inline=False
        )
        
        success_embed.add_field(
            name="📊 Video 2 Analysis", 
            value=comparison_result['video2_analysis'][:1024],
            inline=False
        )
        
        success_embed.add_field(
            name="🤖 AI Model Used",
            value=comparison_result.get('model_used', 'AI Vision'),
            inline=True
        )
        
        success_embed.set_footer(text="Upload more videos to compare different techniques!")
        
        # Update the message
        await processing_msg.edit(embed=success_embed)
        
        # Send the comparison video
        output_path = comparison_result['output_video_path']
        
        # Check file size
        output_size = os.path.getsize(output_path)
        if output_size <= 25 * 1024 * 1024:
            with open(output_path, 'rb') as f:
                comparison_video = discord.File(f, filename="motion_comparison.mp4")
                await ctx.followup.send(
                    "Here's your side-by-side motion comparison! 🎥",
                    file=comparison_video
                )
        else:
            await ctx.followup.send(
                "⚠️ The comparison video is too large to upload to Discord."
            )
        
        # Clean up temporary files
        try:
            os.remove(tmp_path1)
            os.remove(tmp_path2)
            if os.path.exists(output_path):
                os.remove(output_path)
        except:
            pass
        
    except Exception as e:
        logger.error(f"Error comparing videos: {str(e)}")
        error_embed = discord.Embed(
            title="❌ Comparison Error",
            description=f"Sorry, I couldn't compare your videos.\n\n"
                       f"**Error:** {str(e)}\n\n"
                       f"**Tips:**\n"
                       f"• Use videos under 10 seconds for best results\n"
                       f"• Ensure both videos show similar exercises\n"
                       f"• Make sure the movement is clearly visible",
            color=discord.Color.red()
        )
        await processing_msg.edit(embed=error_embed)
        
        # Clean up temporary files
        if 'tmp_path1' in locals() and os.path.exists(tmp_path1):
            os.remove(tmp_path1)
        if 'tmp_path2' in locals() and os.path.exists(tmp_path2):
            os.remove(tmp_path2)


@bot.event
async def on_ready():
    """Called when the bot is ready and connected to Discord."""
    logger.info(f'MyoCouch is online as {bot.user} (ID: {bot.user.id})')
    logger.info(f'Using AI Vision-{MODEL_SIZE} for video analysis')
    
    # Show memory usage if using GPU
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except:
        pass
    
    # Sync commands
    try:
        if DEBUG_GUILD_ID:
            guild_id = int(DEBUG_GUILD_ID)
            await bot.sync_commands(guild_ids=[guild_id])
            logger.info(f"Commands synced to guild {guild_id}")
        else:
            await bot.sync_commands()
            logger.info("Commands synced globally")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")


def main():
    """Start the MyoCouch bot."""
    logger.info("Starting MyoCouch Discord Bot with AI Vision...")
    logger.info("Use /couch command to upload a video for AI coaching analysis")
    logger.info(f"Configured model size: {MODEL_SIZE}")
    bot.run(TOKEN)


if __name__ == "__main__":
    main() 