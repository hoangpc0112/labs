import sys, cv2, numpy as np

if len(sys.argv) < 3:
    print("Dung: python3 compare_wm.py <watermark_goc.png> <watermark_khoi_phuc.png>")
    sys.exit(1)

goc = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
phuc = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
if goc is None or phuc is None:
    print("Loi doc anh"); sys.exit(1)

h, w = goc.shape
phuc = cv2.resize(phuc, (w, h))
corr = cv2.matchTemplate(phuc, goc, cv2.TM_CCOEFF_NORMED)[0][0]
mse = np.mean((goc.astype(np.float64) - phuc.astype(np.float64))**2)
psnr = 10 * np.log10(255**2/(mse+1e-10))

print(f"Tuong quan: {corr:.4f}")
print(f"PSNR: {psnr:.2f} dB")
if corr > 0.4:
    print(">> Watermark phuc hoi thanh cong.")
