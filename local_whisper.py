import os
import subprocess
from faster_whisper import WhisperModel

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = r"C:\Users\STAR PC\Desktop\fyp_text\test_video"  # Change if needed
FFMPEG_PATH = os.path.join(BASE_DIR, "ffmpeg.exe")

# =========================
# INITIALIZE WHISPER
# =========================
device = "cuda" if os.path.exists("C:\\Windows\\System32\\nvidia-smi.exe") else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

model = WhisperModel(
    "large-v3",
    device=device,
    compute_type=compute_type
)

print(f"Using device: {device}")

# =========================
# FUNCTION TO TRANSCRIBE AUDIO
# =========================
def transcribe_audio(audio_path):
    folder = os.path.dirname(audio_path)
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    transcript_path = os.path.join(folder, f"{audio_name}.txt")

    print(f"\nProcessing audio: {audio_path}")

    # Quick detection
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        initial_prompt="PQNK, Emmer Wheat, subsoiler, beds, جنتر، کاشت، کلرٹھی"
    )
    detected_lang = info.language
    if detected_lang not in ["en", "ur"]:
        detected_lang = "ur"

    # Final transcription
    segments, _ = model.transcribe(
        audio_path,
        language=detected_lang,
        beam_size=10,
        vad_filter=True,
        initial_prompt="PQNK, Emmer Wheat, subsoiler, beds, جنتر، کاشت، کلرٹھی"
    )

    raw_text = " ".join(segment.text.strip() for segment in segments)

    # Save transcript
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    print(f"✅ Transcribed → saved to '{transcript_path}'")

# =========================
# WALK THROUGH SUBFOLDERS
# =========================
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:

        file_path = os.path.join(root, file)

        # =========================
        # 🔴 NEW: CONVERT MP4 → MP3
        # =========================
        if file.lower().endswith(".mp4"):
            audio_name = os.path.splitext(file)[0]
            mp3_path = os.path.join(root, f"{audio_name}.mp3")

            if not os.path.exists(mp3_path):
                print(f"🎬 Converting video to audio: {file_path}")
                subprocess.run([
                    FFMPEG_PATH,
                    "-i", file_path,
                    "-vn",
                    "-acodec", "libmp3lame",
                    "-ab", "192k",
                    mp3_path
                ])
                print(f"✅ Converted → {mp3_path}")

        # =========================
        # EXISTING LOGIC (UNCHANGED)
        # =========================
        if file.lower().endswith((".wav", ".mp3")):

            audio_path = os.path.join(root, file)

            # Check if transcript already exists
            audio_name = os.path.splitext(file)[0]
            transcript_path = os.path.join(root, f"{audio_name}.txt")

            if os.path.exists(transcript_path):
                print(f"⏩ Skipping (already transcribed): {audio_path}")
                continue

            transcribe_audio(audio_path)

        # =========================
        # 🔴 ALSO TRANSCRIBE NEWLY CREATED MP3
        # =========================
        if file.lower().endswith(".mp4"):
            audio_name = os.path.splitext(file)[0]
            mp3_path = os.path.join(root, f"{audio_name}.mp3")

            if os.path.exists(mp3_path):

                transcript_path = os.path.join(root, f"{audio_name}.txt")

                if os.path.exists(transcript_path):
                    print(f"⏩ Skipping (already transcribed): {mp3_path}")
                    continue

                transcribe_audio(mp3_path)