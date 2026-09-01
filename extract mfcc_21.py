import os
import random
import numpy as np
import librosa

FLAC_DIR = "data/ASVspoof2021_LA_eval/flac"
PROTOCOL_FILE = "data/ASVspoof2021_LA_eval/labels.txt"
OUTPUT_DIR = "features"
OUTPUT_X_NAME = "X_2021eval.npy"
OUTPUT_Y_NAME = "y_2021eval.npy"
USE_SIMPLE_LABELS = True   # True for the 2021 eval labels.txt format, False for 2019-style protocol files
SAMPLE_SIZE = 10000        # only extract MFCCs for this many randomly chosen files (set to None to process all)
RANDOM_SEED = 42           # keeps the same random subset across reruns, for reproducibility

N_MFCC = 40          # number of MFCC coefficients per frame
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

def load_labels_simple(labels_path):
    """
    For the ASVspoof2021_LA eval labels.txt file, where each line is just:
    file_id label
    (e.g. "LA_E_9332881 1"), already using the 1=bonafide, 0=spoof convention.
    """
    labels = {}
    with open(labels_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            file_id, label = parts
            labels[file_id] = int(label)
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
    labels = load_labels_simple(PROTOCOL_FILE) if USE_SIMPLE_LABELS else load_labels(PROTOCOL_FILE)
    print(f"Loaded {len(labels)} labels.")

    features = []
    targets = []
    skipped = 0

    flac_files = [f for f in os.listdir(FLAC_DIR) if f.endswith(".flac")]
    print(f"Found {len(flac_files)} audio files total.")

    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(flac_files):
        random.seed(RANDOM_SEED)
        flac_files = random.sample(flac_files, SAMPLE_SIZE)
        print(f"Randomly sampled down to {len(flac_files)} files for extraction.")

    print(f"Extracting MFCCs for {len(flac_files)} files...")

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

    np.save(os.path.join(OUTPUT_DIR, OUTPUT_X_NAME), features)
    np.save(os.path.join(OUTPUT_DIR, OUTPUT_Y_NAME), targets)
    print(f"Saved to {OUTPUT_DIR}/{OUTPUT_X_NAME} and {OUTPUT_DIR}/{OUTPUT_Y_NAME}")

if __name__ == "__main__":
    main()