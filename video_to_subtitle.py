import sys
import os
import subprocess
import argparse
from datetime import timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import whisper
except ImportError:
    print("❌ Whisper not installed. Run: pip install openai-whisper")
    sys.exit(1)

from src.config import Config as cfg

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = int(td.total_seconds() % 60)
    millis = int((td.total_seconds() % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def extract_audio(video_path, audio_path):
    print(f"🎵 Extracting audio from: {video_path}")
    cmd = f"ffmpeg -i \"{video_path}\" -ar {cfg.SAMPLE_RATE} -ac 1 \"{audio_path}\" -y -loglevel error"
    result = subprocess.call(cmd, shell=True)
    if result != 0:
        print("❌ ffmpeg error. Is ffmpeg installed?")
        sys.exit(1)
    print("✅ Audio extracted.")

def generate_subtitles(video_path, output_srt, model_size="base"):
    os.makedirs(os.path.dirname(output_srt) or ".", exist_ok=True)
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    print(f"🧠 Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)
    print("📝 Transcribing...")
    result = model.transcribe(audio_path, word_timestamps=True)
    print(f"💾 Writing SRT to {output_srt}")
    with open(output_srt, 'w', encoding='utf-8') as f:
        idx = 1
        for seg in result['segments']:
            for word in seg['words']:
                text = word['word'].strip()
                if not text:
                    continue
                f.write(f"{idx}\n{format_timestamp(word['start'])} --> {format_timestamp(word['end'])}\n{text}\n\n")
                idx += 1
    os.remove(audio_path)
    print(f"✅ Done! Subtitles saved to {output_srt}")
    return output_srt

def preview(srt_path):
    if not os.path.exists(srt_path):
        return
    print("\n" + "="*50)
    print("📖 PREVIEW (first 5 entries)")
    print("="*50)
    with open(srt_path, 'r') as f:
        lines = f.readlines()[:15]
        print(''.join(lines))
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, default="outputs/subtitles.srt", help="Output .srt path")
    parser.add_argument("--model", type=str, default="base", choices=["tiny","base","small","medium","large"])
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        sys.exit(1)

    print("="*60)
    print("🎬 VIDEO TO SUBTITLE GENERATOR")
    print("="*60)
    srt_path = generate_subtitles(args.video, args.output, args.model)
    preview(srt_path)
    print(f"\n🎯 To check subtitles: open '{srt_path}' in any text editor or load in VLC.")