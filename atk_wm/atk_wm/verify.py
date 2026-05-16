import sys, os, cv2, numpy as np, subprocess

if len(sys.argv) < 3:
    print("Dung: python3 verify.py <video_goc.mp4> <video_da_xu_ly.mp4>")
    sys.exit(1)

v1, v2 = sys.argv[1], sys.argv[2]

d1 = "_tmp_v1"; os.makedirs(d1, exist_ok=True)
subprocess.run(f"ffmpeg -y -i {v1} {d1}/f_%04d.png 2>nul", shell=True)
d2 = "_tmp_v2"; os.makedirs(d2, exist_ok=True)
subprocess.run(f"ffmpeg -y -i {v2} {d2}/f_%04d.png 2>nul", shell=True)

fs1, fs2 = sorted(os.listdir(d1)), sorted(os.listdir(d2))
n = min(len(fs1), len(fs2))

print(f"So sanh {v1} vs {v2} ({n} frame):")
psnr_list = []
for i in range(n):
    i1 = cv2.imread(f"{d1}/{fs1[i]}").astype(np.float64)
    i2 = cv2.imread(f"{d2}/{fs2[i]}").astype(np.float64)
    mse = np.mean((i1 - i2)**2)
    psnr = 10 * np.log10(255**2 / (mse + 1e-10))
    psnr_list.append(psnr)

print(f"   PSNR trung binh: {np.mean(psnr_list):.2f} dB")
print(f"   PSNR thap nhat: {min(psnr_list):.2f} dB")

# Kiem tra watermark con sot khong
d_avg = "_tmp_avg"; os.makedirs(d_avg, exist_ok=True)
subprocess.run(f"ffmpeg -y -i {v2} {d_avg}/f_%04d.png 2>nul", shell=True)
fs_avg = sorted(os.listdir(d_avg))
avg_v2 = None
for fn in fs_avg:
    img = cv2.imread(f"{d_avg}/{fn}").astype(np.float64)
    avg_v2 = img if avg_v2 is None else avg_v2 + img
avg_v2 /= len(fs_avg)
avg_v2 = avg_v2.astype(np.uint8)
cv2.imwrite("verify_avg.png", avg_v2)
print("   Da luu verify_avg.png (trung binh frame de kiem tra watermark sot)")

# Do tuong quan voi cac watermark da biet
for wm_name, wm_path in [("WM_original", "wm_original.png"), ("WM_fake", "wm_fake.png")]:
    if os.path.exists(wm_path):
        wm = cv2.imread(wm_path, cv2.IMREAD_GRAYSCALE)
        hw, ww = wm.shape
        avg_gray = cv2.cvtColor(avg_v2, cv2.COLOR_BGR2GRAY)
        avg_gray = cv2.resize(avg_gray, (ww, hw))
        corr = cv2.matchTemplate(avg_gray, wm, cv2.TM_CCOEFF_NORMED)[0][0]
        print(f"   Tuong quan voi {wm_name}: {corr:.4f}")

psnr_avg = np.mean(psnr_list)
print(f"\n>>> PSNR {psnr_avg:.1f}dB: ", end="")
if psnr_avg > 35: print("Chat luong TOT")
elif psnr_avg > 28: print("Chat luong KHA")
else: print("Chat luong KEM")

subprocess.run(f"rm -rf {d1} {d2} {d_avg}", shell=True)
