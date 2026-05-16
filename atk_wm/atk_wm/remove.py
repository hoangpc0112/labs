import sys, os, cv2, numpy as np, subprocess

if len(sys.argv) < 4:
    print("Dung: python3 remove.py <video_co_wm.mp4> <watermark_original.png> <dau_ra.mp4> [alpha]")
    print("  Watermark_original phai la anh watermark THAT (nhu wm_original.png)")
    sys.exit(1)

src = sys.argv[1]
wm_path = sys.argv[2]
out = sys.argv[3]
alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.04

# Kiem tra: wm_original co hop le khong?
wm_check = cv2.imread(wm_path)
if wm_check is None:
    print("Loi: khong doc duoc watermark"); sys.exit(1)
mean_val = np.mean(wm_check)
print(f"Watermark: {wm_path} (mean pixel = {mean_val:.1f})")

d = "_tmp_rm"; os.makedirs(d, exist_ok=True)
subprocess.run(f"ffmpeg -y -i {src} {d}/f_%04d.png 2>nul", shell=True)

wm = wm_check.astype(np.float64)
fs = sorted(os.listdir(d))
print(f"Dang xoa watermark khoi {len(fs)} frames...")
for i, fn in enumerate(fs):
    img = cv2.imread(f"{d}/{fn}").astype(np.float64)
    # Cong thuc: frame_goc = (frame_wm - alpha * wm) / (1 - alpha)
    restored = (img - alpha * wm) / (1 - alpha + 1e-10)
    restored = np.clip(restored, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{d}/{fn}", restored)
    if (i+1) % 30 == 0:
        print(f"   Xu ly {i+1}/{len(fs)} frame...")

subprocess.run(f"ffmpeg -y -framerate 15 -i {d}/f_%04d.png "
               f"-c:v libx264 -crf 10 {out} 2>nul", shell=True)
print(f"Da tao: {out}")

# So sanh voi clean goc
if os.path.exists("clean.mp4"):
    d2 = "_tmp_cln"; os.makedirs(d2, exist_ok=True)
    subprocess.run(f"ffmpeg -y -i clean.mp4 {d2}/f_%04d.png 2>nul", shell=True)
    fs2 = sorted(os.listdir(d2))
    diffs = []
    for fn, fn2 in zip(fs, fs2):
        i1 = cv2.imread(f"{d}/{fn}").astype(np.float64)
        i2 = cv2.imread(f"{d2}/{fn2}").astype(np.float64)
        diffs.append(np.mean(np.abs(i1 - i2)))
    print(f"   Sai so TB so voi clean: {np.mean(diffs):.4f}")
    subprocess.run(f"rm -rf {d2}", shell=True)

subprocess.run(f"rm -rf {d}", shell=True)
