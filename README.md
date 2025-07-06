# MyoCouch - AI-Powered Video Coaching Discord Bot 🏃‍♂️

MyoCouch is a Discord bot that provides personalized coaching advice by analyzing workout videos using Qwen2-VL, a state-of-the-art Video Language Model by Alibaba. The bot processes your videos and overlays coaching advice directly on the video, making it easy to see exactly what to improve.

## Features 🌟

- **Advanced Video Understanding**: Uses Qwen2-VL Video Language Model for comprehensive movement analysis
- **Visual Coaching Overlays**: Coaching advice is overlaid directly on your video segments
- **Intelligent Video Processing**: Automatically downsamples to 30 FPS and processes in 60-frame chunks
- **Multi-Model Support**: Choose between 2B or 7B parameter models based on your hardware
- **Memory-Efficient**: Automatically switches to smaller models if needed

## What's New in v2.0 🚀

- **Qwen2-VL Integration**: Using Alibaba's powerful vision-language model for video understanding
- **Video Overlays**: Coaching advice now appears directly on the video
- **Chunk-Based Analysis**: Videos are analyzed in 2-second segments for detailed feedback
- **Better Understanding**: AI provides context-aware coaching based on actual exercise understanding

## Setup Instructions 🛠️

### 1. Prerequisites

- Python 3.8 or higher
- Discord Bot Token ([Create a bot here](https://discord.com/developers/applications))
- CUDA-capable GPU (recommended) or CPU (slower)
- 4GB+ RAM (8GB+ recommended)
- ffmpeg installed on your system

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd MyoCouch

# Install ffmpeg (if not already installed)
# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg
# macOS:
brew install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the project root:

```env
DISCORD_BOT_TOKEN=your_bot_token_here
DEBUG_GUILD_ID=your_test_server_id_here  # Optional, for faster testing
QWEN2VL_MODEL_SIZE=7B  # Options: 2B, 7B (default)
```

### 4. Run the Bot

```bash
python bot.py
```

The bot will automatically download the Qwen2-VL model on first run.

## Usage 💪

### The `/couch` Command

1. In any Discord channel where the bot has permissions, type `/couch`
2. Upload a workout video (MP4, MOV, AVI, WEBM, or MKV)
3. Wait for MyoCouch to analyze your video (this may take a few minutes)
4. Receive your coached video with AI advice overlaid!

### How It Works:

1. **Video Processing**: Your video is downsampled to 30 FPS for consistent analysis
2. **Chunking**: The video is split into 2-second (60-frame) segments
3. **AI Analysis**: Each segment is analyzed by Qwen2-VL for movement quality
4. **Overlay Creation**: Coaching advice is overlaid on each video segment
5. **Final Output**: All segments are concatenated into a single coached video

## Example Output 📊

Your video will include overlays like:

```
Segment 1: Keep your back straight during the squat. Focus on pushing through your heels. Control the descent speed.

Segment 2: Good depth achieved! Maintain knee alignment over toes. Engage your core throughout.

Segment 3: Watch your forward lean. Keep chest up and proud. Breathe out on the way up.
```

## Model Selection Guide 🤖

Choose the right model size for your hardware:

- **7B (Default)**: Best quality, requires 8GB+ VRAM
- **2B**: Good balance, requires 4GB+ VRAM

The bot will automatically downgrade to a smaller model if it runs out of memory.

## Technical Details 🔧

### Video Processing Pipeline

1. **FPS Normalization**: Videos are resampled to 30 FPS
2. **Chunk Extraction**: 60-frame segments (2 seconds each)
3. **Frame Sampling**: 8 frames selected from each chunk for analysis
4. **AI Analysis**: Qwen2-VL analyzes frames for coaching insights
5. **Overlay Generation**: Text overlays with fade effects
6. **Video Reconstruction**: Chunks are concatenated with overlays

### System Requirements

- **Minimum**: 4GB RAM, 4GB VRAM (for 2B model)
- **Recommended**: 8GB RAM, 8GB+ VRAM (for 7B model)
- **Optimal**: 16GB RAM, 12GB+ VRAM, CUDA-capable GPU

## Troubleshooting 🔍

### Common Issues

1. **"DISCORD_BOT_TOKEN missing"**: Create the `.env` file with your token
2. **"CUDA out of memory"**: The bot will automatically try the 2B model
3. **Video too large**: Keep videos under 25MB and preferably under 1 minute
4. **ffmpeg not found**: Install ffmpeg following the installation instructions
5. **Slow processing**: Normal for CPU-only systems; consider using a GPU

### Performance Tips

- Use shorter videos (30-60 seconds) for faster processing
- Ensure good lighting and clear visibility of movements
- Record in landscape orientation
- Avoid excessive camera movement

## Development 🚀

### Project Structure

```
MyoCouch/
├── bot.py              # Main Discord bot logic
├── video_processor.py  # Qwen2-VL integration and video processing
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
└── README.md          # This file
```

### Future Enhancements

- [ ] Support for longer videos with adaptive chunking
- [ ] Multi-person coaching in the same video
- [ ] Exercise-specific coaching models
- [ ] Real-time streaming analysis
- [ ] Progress tracking across multiple videos

## License 📄

MIT License - Feel free to modify and distribute!

## Acknowledgments 🙏

- Qwen Team at Alibaba for the amazing Qwen2-VL model
- Discord.py community for the bot framework
- FFmpeg for video processing capabilities

---

Built with ❤️ for fitness enthusiasts who want AI-powered coaching directly in their videos!