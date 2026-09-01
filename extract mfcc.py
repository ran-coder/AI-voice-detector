"""
Extracts MFCCs from every .flac file in the ASVspoof2019_LA_train/flac
folder, and pairs each one with its bonafide/spoof label from the
protocol file.

Requires:
  pip install librosa numpy --break-system-packages
"""

import os
import numpy as np
import librosa

FLAC_DIR = "data/LA/ASVspoof2019_LA_dev/flac"
PROTOCOL_FILE = "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
OUTPUT_DIR = "features"

N_MFCC = 40          # number of MFCC coefficients per frame -- 40 is a common choice
MAX_LEN = 200        # fixed number of time-frames per clip (pad/truncate so all clips match)

def load_labels(protocol_path):
    """
    Each record in the protocol file looks like:
    LA_0079 LA_T_1138215 - - bonafide
    (speaker_id  filename  -  -  label)

    We split the WHOLE file by whitespace and group every 5 tokens into
    one record, instead of reading line by line -- this works whether
    the file has proper line breaks or was flattened into one long line.
    """
    labels = {}
    with open(protocol_path, "r") as f:
        tokens = f.read().split()

    if len(tokens) % 5 != 0:
        print(f"Warning: token count ({len(tokens)}) isn't a multiple of 5 -- "
              f"the file format may be different than expected. Check the last "
              f"few tokens: {tokens[-10:]}")

    for i in range(0, len(tokens) - 4, 5):
        record = tokens[i:i + 5]
        filename = record[1]         # e.g. "LA_T_1138215"
        label = record[4]            # "bonafide" or "spoof"
        labels[filename] = 1 if label == "bonafide" else 0

    return labels

def extract_mfcc(filepath, n_mfcc=N_MFCC, max_len=MAX_LEN):
    y, sr = librosa.load(filepath, sr=None)  # sr=None keeps original sample rate
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate so every clip produces the same shape array
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading labels...")
    labels = load_labels(PROTOCOL_FILE)
    print(f"Loaded {len(labels)} labels.")

    features = []
    targets = []
    skipped = 0

    flac_files = [f for f in os.listdir(FLAC_DIR) if f.endswith(".flac")]
    print(f"Found {len(flac_files)} audio files. Extracting MFCCs...")

    for i, fname in enumerate(flac_files, 1):
        file_id = fname.replace(".flac", "")
        if file_id not in labels:
            skipped += 1
            continue

        filepath = os.path.join(FLAC_DIR, fname)
        mfcc = extract_mfcc(filepath)

        features.append(mfcc)
        targets.append(labels[file_id])

        if i % 500 == 0:
            print(f"  processed {i}/{len(flac_files)}")

    features = np.array(features)   # shape: (num_samples, N_MFCC, MAX_LEN)
    targets = np.array(targets)     # shape: (num_samples,) -- 1 = bonafide, 0 = spoof

    print(f"Done. Features shape: {features.shape}, Labels shape: {targets.shape}")
    print(f"Skipped {skipped} files with no matching label.")

    np.save(os.path.join(OUTPUT_DIR, "X_dev.npy"), features)
    np.save(os.path.join(OUTPUT_DIR, "y_dev.npy"), targets)
    print(f"Saved to {OUTPUT_DIR}/X_dev.npy and {OUTPUT_DIR}/y_dev.npy")

if __name__ == "__main__":
    main()