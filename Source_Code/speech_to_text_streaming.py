"""
Arabic Speech Corpus - Speech to Text using Deepgram Nova-3
Processes WAV files from the arabic-speech-corpus dataset and
transcribes them using Deepgram's pre-recorded REST API.

Usage:
    # Activate venv first:
    #   .\\venv\\Scripts\\activate   (Windows)

    python speech_to_text_streaming.py --limit 3      # Test with 3 files
    python speech_to_text_streaming.py                 # Process all
    python speech_to_text_streaming.py --test-set      # Process test set
"""

import os
import csv
import argparse
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

# Paths (relative to this script's location)
SCRIPT_DIR = Path(__file__).resolve().parent

# Load .env from the script's directory (not CWD)
load_dotenv(SCRIPT_DIR / ".env")

CORPUS_DIR = SCRIPT_DIR.parent / "arabic-speech-corpus"
WAV_DIR = CORPUS_DIR / "wav"
TRANSCRIPT_FILE = CORPUS_DIR / "orthographic-transcript.txt"
TEST_WAV_DIR = CORPUS_DIR / "test set" / "wav"
TEST_TRANSCRIPT_FILE = CORPUS_DIR / "test set" / "orthographic-transcript.txt"

# Deepgram API endpoint
DEEPGRAM_API_URL = "https://api.deepgram.com/v1/listen"


def parse_transcript_file(transcript_path: Path) -> dict:
    """
    Parse the orthographic-transcript.txt file.
    Each line has the format: "filename.wav" "transcript text"
    Returns a dict mapping filename -> transcript.
    """
    transcripts = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('"')
            if len(parts) >= 4:
                filename = parts[1].strip()
                ground_truth = parts[3].strip()
                transcripts[filename] = ground_truth
    return transcripts


def transcribe_file(api_key: str, wav_path: Path) -> str:
    """
    Send a WAV file to Deepgram's pre-recorded REST API and return the transcript.
    Uses direct HTTP call for reliable authentication.
    """
    with open(wav_path, "rb") as audio_file:
        buffer_data = audio_file.read()

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }

    params = {
        "model": "nova-3",
        "language": "ar",
        "smart_format": "true",
    }

    response = httpx.post(
        DEEPGRAM_API_URL,
        headers=headers,
        params=params,
        content=buffer_data,
        timeout=60.0,
    )
    response.raise_for_status()

    data = response.json()
    transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
    return transcript


def process_dataset(
    api_key: str,
    wav_dir: Path,
    transcripts: dict,
    output_csv: Path,
    limit: int = None,
    dataset_name: str = "main",
):
    """
    Process all WAV files in the dataset directory.
    Writes results to a CSV file.
    """
    wav_files = []
    for filename, ground_truth in transcripts.items():
        wav_path = wav_dir / filename
        if wav_path.exists():
            wav_files.append((filename, wav_path, ground_truth))

    wav_files.sort(key=lambda x: x[0])

    if limit:
        wav_files = wav_files[:limit]

    total = len(wav_files)
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} dataset: {total} files")
    print(f"Output: {output_csv}")
    print(f"{'='*60}\n")

    results = []

    for i, (filename, wav_path, ground_truth) in enumerate(wav_files, 1):
        print(f"[{i}/{total}] Transcribing: {filename}")

        try:
            deepgram_transcript = transcribe_file(api_key, wav_path)
            gt_preview = ground_truth[:80] + ("..." if len(ground_truth) > 80 else "")
            dg_preview = deepgram_transcript[:80] + ("..." if len(deepgram_transcript) > 80 else "")
            print(f"  Ground Truth : {gt_preview}")
            print(f"  Deepgram     : {dg_preview}")
            print()

            results.append({
                "filename": filename,
                "ground_truth": ground_truth,
                "deepgram_transcript": deepgram_transcript,
            })

        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append({
                "filename": filename,
                "ground_truth": ground_truth,
                "deepgram_transcript": f"ERROR: {e}",
            })

        time.sleep(0.2)

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["filename", "ground_truth", "deepgram_transcript"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"Done! {len(results)} files processed.")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Arabic Speech Corpus using Deepgram Nova-3"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N files (useful for testing)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV file path (default: results.csv in API folder)",
    )
    parser.add_argument(
        "--test-set", action="store_true",
        help="Process the test set instead of the main corpus",
    )
    args = parser.parse_args()

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("ERROR: Please set your DEEPGRAM_API_KEY in the .env file")
        print(f"  .env location: {SCRIPT_DIR / '.env'}")
        return

    if args.test_set:
        wav_dir = TEST_WAV_DIR
        transcript_file = TEST_TRANSCRIPT_FILE
        dataset_name = "test set"
        default_output = SCRIPT_DIR / "results_test.csv"
    else:
        wav_dir = WAV_DIR
        transcript_file = TRANSCRIPT_FILE
        dataset_name = "main corpus"
        default_output = SCRIPT_DIR / "results.csv"

    output_csv = Path(args.output) if args.output else default_output

    print(f"Loading transcripts from: {transcript_file}")
    transcripts = parse_transcript_file(transcript_file)
    print(f"Found {len(transcripts)} transcript entries")

    process_dataset(
        api_key=api_key,
        wav_dir=wav_dir,
        transcripts=transcripts,
        output_csv=output_csv,
        limit=args.limit,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    main()