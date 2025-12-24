"""
evaluate_performance.py — Evaluates visual fidelity of stego video vs. original video.
Computes MSE, PSNR, SSIM, NCC, and Entropy Difference across all frames.
"""

import os
import cv2
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.measure import shannon_entropy

# --- Paths ---
PROJECT_DIR = r"C:\Users\KANISHK\Desktop\STEGO-BASE"
ORIGINAL_FRAMES_DIR = os.path.join(PROJECT_DIR, "output_frames")
STEGO_FRAMES_DIR = os.path.join(PROJECT_DIR, "output_frames_with_message")

# --- Helper function ---
def calculate_frame_metrics(orig_frame, stego_frame):
    """Compute MSE, PSNR, SSIM, NCC, and Entropy Difference between two frames."""
    if orig_frame.shape != stego_frame.shape:
        stego_frame = cv2.resize(stego_frame, (orig_frame.shape[1], orig_frame.shape[0]))

    orig_float = orig_frame.astype("float")
    stego_float = stego_frame.astype("float")

    # 1️⃣ Mean Squared Error (MSE)
    mse = np.mean((orig_float - stego_float) ** 2)

    # 2️⃣ Peak Signal-to-Noise Ratio (PSNR)
    psnr = float('inf') if mse == 0 else peak_signal_noise_ratio(orig_frame, stego_frame, data_range=255)

    # 3️⃣ Structural Similarity Index (SSIM)
    ssim = structural_similarity(
        cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(stego_frame, cv2.COLOR_BGR2GRAY)
    )

    # 4️⃣ Normalized Cross-Correlation (NCC)
    numerator = np.sum(orig_float * stego_float)
    denominator = np.sqrt(np.sum(orig_float ** 2) * np.sum(stego_float ** 2))
    ncc = numerator / denominator if denominator != 0 else 0

    # 5️⃣ Entropy Difference
    entropy_orig = shannon_entropy(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY))
    entropy_stego = shannon_entropy(cv2.cvtColor(stego_frame, cv2.COLOR_BGR2GRAY))
    entropy_diff = abs(entropy_orig - entropy_stego)

    return mse, psnr, ssim, ncc, entropy_diff


def evaluate_video_quality():
    """Compare original and stego frames using MSE, PSNR, SSIM, NCC, and Entropy Difference."""
    print("\n🎥 Evaluating Visual Quality (Original vs. Stego Frames)...")

    orig_files = sorted([f for f in os.listdir(ORIGINAL_FRAMES_DIR) if f.lower().endswith(('.jpg', '.png'))])
    stego_files = sorted([f for f in os.listdir(STEGO_FRAMES_DIR) if f.lower().endswith(('.jpg', '.png'))])

    if not orig_files or not stego_files:
        raise ValueError("Frames missing! Ensure both 'output_frames' and 'output_frames_with_message' exist.")

    mse_list, psnr_list, ssim_list, ncc_list, entropy_diff_list = [], [], [], [], []
    count = min(len(orig_files), len(stego_files))

    for i in range(count):
        orig_path = os.path.join(ORIGINAL_FRAMES_DIR, orig_files[i])
        stego_path = os.path.join(STEGO_FRAMES_DIR, stego_files[i])

        orig = cv2.imread(orig_path)
        stego = cv2.imread(stego_path)
        if orig is None or stego is None:
            print(f"⚠️ Skipping unreadable frame pair: {orig_files[i]}")
            continue

        mse, psnr, ssim, ncc, entropy_diff = calculate_frame_metrics(orig, stego)
        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        ncc_list.append(ncc)
        entropy_diff_list.append(entropy_diff)

        if i % 20 == 0:
            print(f"Processed {i}/{count} frames...")

    return {
        "frame_count": count,
        "avg_mse": np.mean(mse_list),
        "avg_psnr": np.mean(psnr_list),
        "avg_ssim": np.mean(ssim_list),
        "avg_ncc": np.mean(ncc_list),
        "avg_entropy_diff": np.mean(entropy_diff_list)
    }


def main():
    print("🚀 Starting Steganographic Video Quality Evaluation...\n")

    start = time.time()
    metrics = evaluate_video_quality()
    end = time.time()

    # Handle infinite PSNR gracefully for display
    psnr_display = metrics['avg_psnr'] if np.isfinite(metrics['avg_psnr']) else 90.0

    print("\n📊 ===== VISUAL QUALITY REPORT =====")
    print(f"🔹 Frames Compared:   {metrics['frame_count']}")
    print(f"🔹 Avg MSE:            {metrics['avg_mse']:.6f}")
    print(f"🔹 Avg PSNR:           {'>90.00' if not np.isfinite(metrics['avg_psnr']) else f'{psnr_display:.2f}'} dB")
    print(f"🔹 Avg SSIM:           {metrics['avg_ssim']:.6f}")
    print(f"🔹 Avg NCC:            {metrics['avg_ncc']:.6f}")
    print(f"🔹 Avg Entropy Diff:   {metrics['avg_entropy_diff']:.6f}")
    print("───────────────────────────────────────────────")
    print(f"⚡ Evaluation Time:    {end - start:.2f} seconds")
    print("===============================================\n")


if __name__ == "__main__":
    main()
