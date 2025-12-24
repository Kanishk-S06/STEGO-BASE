import subprocess
import os
import sys

# --------------------------------------------------
# Project root
PROJECT_DIR = r"C:\Users\KANISHK\Desktop\STEGO-BASE"
os.chdir(PROJECT_DIR)

# --------------------------------------------------
# Two-Tier Steganography Pipeline
steps = [
    ("Tier-0: Encrypt Payload", "Encryption.py"),
    ("Tier-1: Frame Extraction + Watermarking", "extract_frames.py"),
    ("Tier-2: Payload Embedding", "embed_msg.py"),
    ("Tier-2: Payload Extraction", "extract_modified_frames.py"),
    ("Tier-0: Decryption", "Decryption.py")
]

# --------------------------------------------------
def run_script(name, script):
    print(f"\n🔹 Running step: {name}")
    try:
        subprocess.run([sys.executable, script], check=True)
        print(f"✅ {name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error while running {script}")
        sys.exit(1)

# --------------------------------------------------
def main():
    print("🚀 Starting Two-Tier Video Steganography Pipeline...\n")
    for name, script in steps:
        run_script(name, script)
    print("\n🎉 Pipeline completed successfully!")

# --------------------------------------------------
if __name__ == "__main__":
    main()
