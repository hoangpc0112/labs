import wave
import numpy as np

BLOCK = 512
SR = 44100

PILOT_START = 200
BAND_A = (900, 1500)
BAND_B = (1800, 2400)
MAX_BITS = 2048

def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sr != SR:
        raise SystemExit(f"Sai sample rate: {sr}")

    if sw != 2:
        raise SystemExit("Script này kỳ vọng PCM 16-bit.")

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0

    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)

    return data

def band_bins(freq_range):
    freqs = np.fft.rfftfreq(BLOCK, d=1.0 / SR)
    lo, hi = freq_range
    return np.where((freqs >= lo) & (freqs <= hi))[0]

def read_dew_bit(audio, block_idx):
    start = block_idx * BLOCK
    end = start + BLOCK

    block = audio[start:end]
    spec = np.fft.rfft(block)

    bins_a = band_bins(BAND_A)
    bins_b = band_bins(BAND_B)

    ea = np.sum(np.abs(spec[bins_a]) ** 2)
    eb = np.sum(np.abs(spec[bins_b]) ** 2)

    return 1 if ea > eb else 0

def bits_to_bytes(bits):
    out = bytearray()

    for i in range(0, len(bits) - 7, 8):
        val = 0
        for bit in bits[i:i+8]:
            val = (val << 1) | bit
        out.append(val)

    return bytes(out)

audio = read_wav("audio.wav")

bits = []
for i in range(MAX_BITS):
    bits.append(read_dew_bit(audio, PILOT_START + i))

data = bits_to_bytes(bits)
text = data.decode("ascii", errors="ignore")

end_idx = text.find("|END")
if end_idx != -1:
    text = text[:end_idx + 4]

print(text)
