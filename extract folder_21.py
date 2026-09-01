"""
Downloads the ASVspoof 2021 LA evaluation dataset from Hugging Face and
saves the audio as .flac files plus a labels file locally, in the same
style as your ASVspoof 2019 data.

Requires:
  pip install datasets soundfile --break-system-packages
"""

import os
import io
import soundfile as sf
from datasets import load_dataset, Audio

OUTPUT_DIR = "data/ASVspoof2021_LA_eval"
FLAC_DIR = os.path.join(OUTPUT_DIR, "flac")
LABELS_FILE = os.path.join(OUTPUT_DIR, "labels.txt")

def main():
    os.makedirs(FLAC_DIR, exist_ok=True)

    print("Downloading dataset from Hugging Face (this will take a while, ~7.6 GB)...")
    ds = load_dataset("SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA", split="test")

    # Skip auto-decoding (which needs torchcodec + FFmpeg) -- get raw
    # bytes instead and decode them ourselves with soundfile.
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"Loaded {len(ds)} samples.")

    print("Saving audio files and labels locally...")
    with open(LABELS_FILE, "w") as labels_out:
        for i, example in enumerate(ds):
            filename = example["path"]
            audio_bytes = example["audio"]["bytes"]
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            label = example["label"]  # 0 = bonafide, 1 = spoof (per the dataset's ClassLabel schema)

            out_path = os.path.join(FLAC_DIR, filename)
            sf.write(out_path, audio, sr)

            # Convert to the same 1/0 convention used in your earlier scripts:
            # 1 = bonafide, 0 = spoof
            binary_label = 1 if label == 0 else 0
            file_id = filename.replace(".flac", "")
            labels_out.write(f"{file_id} {binary_label}\n")

            if (i + 1) % 1000 == 0:
                print(f"  saved {i + 1}/{len(ds)}")

    print(f"Done. Audio saved to {FLAC_DIR}")
    print(f"Labels saved to {LABELS_FILE}")

if __name__ == "__main__":
    main()