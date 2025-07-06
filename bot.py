# MyoCouch - Discord bot for video coaching with Qwen2-VL
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
MODEL_SIZE = os.getenv("QWEN2VL_MODEL_SIZE", "7B")  # Default to 7B

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
logger.info(f"Initializing Qwen2-VL with model size: {MODEL_SIZE}")
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
            title="🎬 Processing Your Video with Qwen2-VL",
            description=f"**Step 1/4:** Downloading video...\n"
                       f"Model: Qwen2-VL-{MODEL_SIZE}",
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
        
        # Update status
        embed.description = f"**Step 2/4:** Analyzing video with AI...\n" \
                          f"Model: Qwen2-VL-{MODEL_SIZE}"
        await processing_msg.edit(embed=embed)
        
        # Process the video in a separate thread
        loop = asyncio.get_event_loop()
        try:
            coaching_result = await loop.run_in_executor(
                None,
                video_processor.process_video,
                tmp_path
            )
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            # Try with smaller model if we ran out of memory
            if "CUDA out of memory" in str(e) and MODEL_SIZE == "7B":
                embed.description = "**Switching to smaller model due to memory constraints...**"
                await processing_msg.edit(embed=embed)
                
                # Reinitialize with smaller model
                video_processor = VideoCoachingProcessor(model_size="2B")
                
                coaching_result = await loop.run_in_executor(
                    None,
                    video_processor.process_video,
                    tmp_path
                )
            else:
                raise
        
        # Check if processing was successful
        if coaching_result['status'] != 'success':
            raise Exception("Video processing failed")
        
        # Update status
        embed.description = f"**Step 3/4:** Creating coached video with overlays...\n" \
                          f"Model: {coaching_result.get('model_used', 'Qwen2-VL')}"
        await processing_msg.edit(embed=embed)
        
        # Create success embed
        success_embed = discord.Embed(
            title="✅ MyoCouch Analysis Complete!",
            description=f"Your video has been analyzed and coaching advice has been overlaid on the video.",
            color=discord.Color.green()
        )
        
        # Add video stats
        video_info = coaching_result['video_info']
        success_embed.add_field(
            name="📊 Video Statistics",
            value=f"• Duration: {video_info['duration_seconds']:.1f} seconds\n"
                  f"• Resolution: {video_info['resolution'][0]}x{video_info['resolution'][1]}\n"
                  f"• Segments analyzed: {video_info['chunks_processed']}",
            inline=False
        )
        
        # Add summary of coaching segments
        segments = coaching_result['coaching_segments']
        if segments:
            # Show first 3 segments as preview
            preview_segments = segments[:3]
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
            value=coaching_result.get('model_used', 'Qwen2-VL'),
            inline=True
        )
        
        success_embed.set_footer(text="Keep training! Upload another video to track your progress.")
        
        # Update the message
        await processing_msg.edit(embed=success_embed)
        
        # Send the coached video
        output_path = coaching_result['output_video_path']
        
        # Check if the output video is within Discord's file size limit
        output_size = os.path.getsize(output_path)
        if output_size <= 25 * 1024 * 1024:  # 25MB limit
            # Send the coached video
            with open(output_path, 'rb') as f:
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
                       f"• Make sure the exercise is clearly visible",
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
    logger.info(f'Using Qwen2-VL-{MODEL_SIZE} for video analysis')
    
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
    logger.info("Starting MyoCouch Discord Bot with Qwen2-VL...")
    logger.info("Use /couch command to upload a video for AI coaching analysis")
    logger.info(f"Configured model size: {MODEL_SIZE}")
    bot.run(TOKEN)


if __name__ == "__main__":
    main() 