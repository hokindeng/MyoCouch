# MyoCouch - Discord bot for video coaching with Qwen2-VL
import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import logging
from pathlib import Path
import asyncio
from typing import Optional
from video_processor_enhanced import EnhancedVideoCoachingProcessor  # Use enhanced processor
from tempfile import NamedTemporaryFile

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DEBUG_GUILD_ID = os.getenv("DEBUG_GUILD_ID")
MODEL_SIZE = os.getenv("QWEN2VL_MODEL_SIZE", "7B")  # Default to 7B
MEMORY_OPTIMIZATION = os.getenv("MEMORY_OPTIMIZATION", "high").lower()  # high, medium, low

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

# Configure memory optimization settings
if MEMORY_OPTIMIZATION == "high":
    # Maximum memory savings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6'
    logger.info("Memory optimization: HIGH - Using aggressive memory management")
elif MEMORY_OPTIMIZATION == "medium":
    # Balanced approach
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,garbage_collection_threshold:0.7'
    logger.info("Memory optimization: MEDIUM - Using balanced memory management")
else:
    # Low optimization (default PyTorch behavior)
    logger.info("Memory optimization: LOW - Using default memory management")

# Initialize enhanced video processor
logger.info(f"Initializing Enhanced AI Vision with model size: {MODEL_SIZE}")
video_processor = EnhancedVideoCoachingProcessor(model_size=MODEL_SIZE)  # Use enhanced processor

# Configure quality settings based on memory optimization
if MEMORY_OPTIMIZATION == "high":
    # Override settings for maximum memory savings
    video_processor.target_fps = 15
    video_processor.max_resolution = (854, 480)  # 480p
    video_processor.chunk_size = 20
    logger.info("Video quality: Optimized for memory (480p, 15fps)")
elif MEMORY_OPTIMIZATION == "medium":
    # Keep balanced defaults
    logger.info("Video quality: Balanced (720p, 24fps)")
else:
    # High quality mode
    video_processor.target_fps = 30
    video_processor.max_resolution = (1920, 1080)  # 1080p
    video_processor.chunk_size = 45
    logger.info("Video quality: High (1080p, 30fps)")

# Video coaching slash command
@bot.slash_command(
    name="couch",
    description="Upload a video to receive AI-powered coaching with intelligent analysis and visual annotations"
)
async def couch_command(
    ctx: discord.ApplicationContext,
    video: discord.Option(
        discord.Attachment,
        description="Upload a workout video for intelligent AI coaching analysis (max 25MB)",
        required=True
    )
):
    """Process uploaded video with enhanced intelligence and provide coaching advice."""
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
            title="🎬 Processing Your Video with Enhanced AI Intelligence",
            description=f"**Step 1/5:** Downloading video...\n"
                       f"AI-powered analysis with human detection in progress",
            color=discord.Color.blue()
        )
        embed.set_footer(text="Enhanced analysis includes exercise recognition and human tracking")
        processing_msg = await ctx.followup.send(embed=embed)
        
        # Download the video
        video_data = await video.read()
        
        # Save video temporarily
        with NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_data)
            tmp_path = tmp_file.name
        
        # Update status
        embed.description = f"**Step 2/5:** Identifying exercise and analyzing person...\n" \
                          f"Advanced AI vision analysis in progress"
        await processing_msg.edit(embed=embed)
        
        # Process the video in a separate thread
        loop = asyncio.get_event_loop()
        try:
            coaching_result = await loop.run_in_executor(
                None,
                video_processor.process_video_with_intelligence,  # Use enhanced method
                tmp_path
            )
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            # Try with smaller model if we ran out of memory
            if "CUDA out of memory" in str(e) and MODEL_SIZE == "7B":
                embed.description = "**Optimizing AI model for your system...**"
                await processing_msg.edit(embed=embed)
                
                # Reinitialize with smaller model
                video_processor = EnhancedVideoCoachingProcessor(model_size="2B")  # Use enhanced processor
                
                coaching_result = await loop.run_in_executor(
                    None,
                    video_processor.process_video_with_intelligence,  # Use enhanced method
                    tmp_path
                )
            else:
                raise
        
        # Check if processing was successful
        if coaching_result['status'] != 'success':
            raise Exception("Video processing failed")
        
        # Update status
        embed.description = f"**Step 3/5:** Adding visual annotations and coaching overlays...\n" \
                          f"Finalizing your personalized coaching video"
        await processing_msg.edit(embed=embed)
        
        # Create success embed with enhanced information
        success_embed = discord.Embed(
            title="✅ MyoCouch Enhanced Analysis Complete!",
            description=f"Your video has been analyzed with intelligent coaching and visual annotations.",
            color=discord.Color.green()
        )
        
        # Add exercise analysis
        exercise_info = coaching_result.get('exercise_analysis', {})
        if exercise_info:
            success_embed.add_field(
                name="🏋️ Exercise Analysis",
                value=f"• **Exercise:** {exercise_info.get('exercise_type', 'Unknown')}\n"
                      f"• **Person:** {exercise_info.get('person_gender', 'Unknown').title()}\n"
                      f"• **Total Segments:** {exercise_info.get('total_segments', 0)}",
                inline=True
            )
        
        # Add video stats
        video_info = coaching_result['video_info']
        success_embed.add_field(
            name="📊 Video Statistics",
            value=f"• Duration: {video_info['duration_seconds']:.1f} seconds\n"
                  f"• Resolution: {video_info['resolution'][0]}x{video_info['resolution'][1]}\n"
                  f"• FPS: {video_info['fps']:.0f}",
            inline=True
        )
        
        # Add dynamic coaching preview
        segments = coaching_result['coaching_segments']
        if segments:
            # Show first 3 segments as preview
            preview_segments = segments[:3]
            preview_text = '\n'.join([f"• {seg}" for seg in preview_segments])
            if len(segments) > 3:
                preview_text += f"\n... and {len(segments) - 3} more segments with unique advice"
            
            success_embed.add_field(
                name="💪 Intelligent Coaching Preview",
                value=preview_text[:1024],  # Discord field limit
                inline=False
            )
        
        success_embed.add_field(
            name="🤖 AI Features Used",
            value=f"• Vision Analysis: Advanced AI Model\n"
                  f"• Human Detection: Computer Vision\n"
                  f"• Visual Annotations: Arrows & Labels",
            inline=False
        )
        
        success_embed.set_footer(text="Each segment contains unique, phase-specific coaching advice!")
        
        # Update the message
        await processing_msg.edit(embed=success_embed)
        
        # Send the enhanced coached video
        output_path = coaching_result['output_video_path']
        
        # Check if the output video is within Discord's file size limit
        output_size = os.path.getsize(output_path)
        if output_size <= 25 * 1024 * 1024:  # 25MB limit
            # Send the coached video
            with open(output_path, 'rb') as f:
                coached_video = discord.File(f, filename=f"enhanced_coached_{video.filename}")
                await ctx.followup.send(
                    "Here's your intelligently coached video with visual annotations! 🎥✨",
                    file=coached_video
                )
        else:
            # Video too large, provide alternative
            await ctx.followup.send(
                "⚠️ The enhanced coached video is too large to upload to Discord. "
                "Consider using a shorter video or reducing quality."
            )
        
        # Clean up temporary files
        try:
            os.remove(tmp_path)
            if os.path.exists(output_path):
                os.remove(output_path)
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
                       f"• Make sure the person is clearly visible\n"
                       f"• Check that the exercise is recognizable",
            color=discord.Color.red()
        )
        await processing_msg.edit(embed=error_embed)
        
        # Clean up temporary file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


@bot.event
async def on_ready():
    """Called when the bot is ready and connected to Discord."""
    logger.info(f'MyoCouch is online as {bot.user} (ID: {bot.user.id})')
    logger.info(f'Using Advanced AI Vision Model')
    
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
    logger.info("Starting MyoCouch Discord Bot with Advanced AI Vision...")
    logger.info("Use /couch command to upload a video for AI coaching analysis")
    logger.info(f"AI Model initialized successfully")
    bot.run(TOKEN)


if __name__ == "__main__":
    main() 