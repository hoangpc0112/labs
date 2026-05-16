import sys, os, cv2, numpy as np, subprocess

if len(sys.argv) < 2:
    print("Dung: python3 recover.py <video.mp4> [so_frame]")
    sys.exit(1)

video = sys.argv[1]
nf = int(sys.argv[2]) if len(sys.argv) > 2 else None

d = "_tmp_rec"; os.makedirs(d, exist_ok=True)
subprocess.run(f"ffmpeg -y -i {video} {d}/f_%04d.png 2>nul", shell=True)
fs = sorted(os.listdir(d))[:nf] if nf else sorted(os.listdir(d))

avg = None
for fn in fs:
    img = cv2.imread(f"{d}/{fn}").astype(np.float64)
    avg = img if avg is None else avg + img
avg /= len(fs)

# Xuat anh thong thuong (pixel ~200)
out = np.clip(avg, 0, 255).astype(np.uint8)
cv2.imwrite("recovered_raw.png", out)

# Xuat anh da duoc normalize de nhin ro watermark
out_eq = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("recovered_wm.png", out_eq)

print(f"Da luu recovered_wm.png (normalize, {len(fs)} frames)")
print(f"Da luu recovered_raw.png (raw, pixel goc ~200)")
subprocess.run(f"rm -rf {d}", shell=True)
