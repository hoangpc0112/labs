import sys, os, cv2, numpy as np, subprocess

if len(sys.argv) < 4:
    print("Dung: python3 embed.py <video_goc.mp4> <watermark.png> <dau_ra.mp4> [alpha]")
    sys.exit(1)

src = sys.argv[1]; wm_path = sys.argv[2]; out = sys.argv[3]
alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.04

wm = cv2.imread(wm_path)
if wm is None:
    print("Loi: khong doc duoc watermark"); sys.exit(1)

print(f"Chen watermark (alpha={alpha}) vao {src}...")
d = "_tmp_emb2"; os.makedirs(d, exist_ok=True)
subprocess.run(f"ffmpeg -y -i {src} {d}/f_%04d.png 2>nul", shell=True)
for fn in sorted(os.listdir(d)):
    img = cv2.imread(f"{d}/{fn}")
    if img is None: continue
    blended = cv2.addWeighted(img, 1-alpha, wm, alpha, 0)
    cv2.imwrite(f"{d}/{fn}", blended)

subprocess.run(f"ffmpeg -y -framerate 15 -i {d}/f_%04d.png -i {src} "
               f"-c:v libx264 -crf 10 -c:a copy -shortest {out} 2>nul", shell=True)
print(f"Da tao: {out}")
subprocess.run(f"rm -rf {d}", shell=True)
