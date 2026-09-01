"""
Use this if you already manually downloaded the ASVspoof zip from Kaggle
and just need to pull out the ASVspoof2019_LA_train/flac folder from it,
without extracting all 6 folders.
"""

import zipfile
import os

# CHANGE THIS to wherever your downloaded zip actually is, e.g.:
# "C:/Users/kasih/Downloads/asvpoof-2019-dataset.zip"
ZIP_PATH = r"C:\Users\kasih\AI voice detector\data\LA.zip"

TARGET_PREFIX = "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt" 
EXTRACT_DIR = "data"

def main():
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    print(f"Opening {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        matches = [n for n in z.namelist() if TARGET_PREFIX in n]

        if not matches:
            print("No matching files found. Here are the first 20 entries in the zip"
                  " so you can check the real folder structure:")
            for n in z.namelist()[:20]:
                print(" ", n)
            return

        print(f"Found {len(matches)} files under '{TARGET_PREFIX}'. Extracting...")
        for i, member in enumerate(matches, 1):
            z.extract(member, EXTRACT_DIR)
            if i % 500 == 0:
                print(f"  extracted {i}/{len(matches)}")

    print(f"Done. Files are under: {os.path.join(EXTRACT_DIR, TARGET_PREFIX)}")

if __name__ == "__main__":
    main()