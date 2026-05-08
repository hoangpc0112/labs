import wave
import numpy as np
import random
import struct
import hashlib

def ask_int(prompt, default=None):
    raw = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    if raw == "" and default is not None:
        return default
    return int(raw)

def ask_text(prompt, default=None):
    raw = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    if raw == "" and default is not None:
        return default
    return raw

def ask_band(prompt):
    raw = input(f"{prompt}, dạng low-high: ").strip().replace(" ", "")
    if "-" not in raw:
        raise SystemExit("Định dạng band không hợp lệ. Ví dụ đúng: 2600-3400")

    lo, hi = raw.split("-", 1)
    return (int(lo), int(hi))

def read_wav(path, expected_sr):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sr != expected_sr:
        raise SystemExit(f"Sample rate không khớp. File đang là {sr} Hz, script đang dùng {expected_sr} Hz.")

    if sw != 2:
        raise SystemExit("Script này kỳ vọng WAV PCM 16-bit.")

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0

    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)

    return data

def band_bins(freq_range, block_size, sr):
    freqs = np.fft.rfftfreq(block_size, d=1.0 / sr)
    lo, hi = freq_range
    return np.where((freqs >= lo) & (freqs <= hi))[0]

def read_dew_bit(audio, block_idx, block_size, sr, band_a, band_b):
    start = block_idx * block_size
    end = start + block_size

    if end > len(audio):
        raise SystemExit("Vượt quá độ dài audio khi đọc watermark.")

    block = audio[start:end]
    spec = np.fft.rfft(block)

    bins_a = band_bins(band_a, block_size, sr)
    bins_b = band_bins(band_b, block_size, sr)

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

print("=== DEW MAIN PAYLOAD DECODER ===")
print("Nhập các tham số lấy được từ manifest DEW0.\n")

audio_path = input("Đường dẫn audio WAV [audio.wav]: ").strip() or "audio.wav"
sr = ask_int("Sample rate", 44100)
block_size = ask_int("BLOCK")
main_start = ask_int("MAIN_START")
span = ask_int("SPAN")
band_a = ask_band("MAIN Band A")
band_b = ask_band("MAIN Band B")
repeat = ask_int("R, số lần lặp mỗi bit")
seed = ask_text("SEED")
expected_magic = ask_text("MAGIC", "DWE2").encode("ascii")
max_bytes = ask_int("MAX, số byte tối đa cần đọc")

audio = read_wav(audio_path, sr)

max_bits = max_bytes * 8
needed_positions = max_bits * repeat

if needed_positions > span:
    raise SystemExit(
        f"Không đủ SPAN. Cần {needed_positions} block nhưng SPAN chỉ có {span}."
    )

rng = random.Random(seed)
positions = rng.sample(range(main_start, main_start + span), needed_positions)

bits = []

for i in range(0, len(positions), repeat):
    group = positions[i:i+repeat]
    votes = [read_dew_bit(audio, pos, block_size, sr, band_a, band_b) for pos in group]

    bit = 1 if sum(votes) > (repeat // 2) else 0
    bits.append(bit)

data = bits_to_bytes(bits)

magic = data[:len(expected_magic)]
print("\n=== KẾT QUẢ MAIN DEW ===")
print("Magic đọc được:", magic)

if magic != expected_magic:
    raise SystemExit(
        "Sai magic. Kiểm tra lại BLOCK, MAIN_START, SPAN, MAIN_BANDS, R, SEED hoặc MAGIC."
    )

cipher_len = struct.unpack(">I", data[4:8])[0]
print("Cipher length:", cipher_len)

cipher = data[8:8+cipher_len]
stored_hash = data[8+cipher_len:8+cipher_len+32]
actual_hash = hashlib.sha256(cipher).digest()

if stored_hash != actual_hash:
    raise SystemExit("SHA-256 không khớp. Payload bị đọc sai.")

open("recovered_payload.enc", "wb").write(cipher)

print("SHA256:", hashlib.sha256(cipher).hexdigest())
print("Created recovered_payload.enc")
