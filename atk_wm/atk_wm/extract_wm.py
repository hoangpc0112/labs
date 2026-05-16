import sys, os, cv2, numpy as np, subprocess

if len(sys.argv) < 4:
    print("Dung: python3 extract_wm.py <secret.mp4> <clean.mp4> <dau_ra.png> [alpha]")
    sys.exit(1)

secret = sys.argv[1]
clean = sys.argv[2]
out = sys.argv[3]
alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.04

def avg_frames(video):
    d = "_tmp_ef"; os.makedirs(d, exist_ok=True)
    subprocess.run(f"ffmpeg -y -i {video} {d}/f_%04d.png 2>nul", shell=True)
    fs = sorted(os.listdir(d))
    avg = None
    for fn in fs:
        img = cv2.imread(f"{d}/{fn}").astype(np.float64)
        avg = img if avg is None else avg + img
    avg /= len(fs)
    subprocess.run(f"rm -rf {d}", shell=True)
    return avg

print(">> Trung binh secret frames...")
avg_secret = avg_frames(secret)
print(">> Trung binh clean frames...")
avg_clean = avg_frames(clean)

# Lay watermark thuan:
# avg_secret = (1-alpha)*avg_clean + alpha*W
# => W = (avg_secret - (1-alpha)*avg_clean) / alpha
W = (avg_secret - (1-alpha) * avg_clean) / alpha
W = np.clip(W, 0, 255).astype(np.uint8)
cv2.imwrite(out, W)
print(f"Da luu {out} (watermark thuan khiet, mean={np.mean(W):.1f})")
