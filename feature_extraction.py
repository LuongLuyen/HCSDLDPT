"""
feature_extraction.py
======================
Trích rút đặc trưng âm thanh tiếng động cơ, bám sát công thức trong
Lecture 10 - Indexing and Retrieval for Audio (slide 8-27).

Pipeline (dùng slide "Audio Feature Extraction Pipeline", trang 8):
  1. Preprocessing : decode -> mono PCM -> resample chuẩn (ở đây dùng 22050 Hz,
     đủ cho dải tần động cơ ~ 20kHz theo Nyquist)
  2. Framing       : chia frame 25ms, hop 10ms (đúng như slide)
  3. Time-domain   : E, ZCR, silence ratio  -> tính trên từng frame
  4. Frequency-domain (qua STFT): spectral centroid, bandwidth,
     harmonicity, f0 (pitch/RPM proxy), MFCC(13)
  5. Aggregation   : mean + std của mỗi đặc trưng frame-level
     -> vector đặc trưng mức segment (cả file)

Vì tiếng động cơ có đủ 4 kênh (Channel 1-2 = audio, Channel 3 = RPM,
Channel 4 = Torque) theo dataset rdoerfler/procedural-engine-sounds,
hàm load_engine_audio() sẽ tách riêng audio (dùng để trích đặc trưng)
và annotation (dùng làm ground-truth đánh giá, KHÔNG đưa vào vector).
"""

import numpy as np
import librosa

SR = 22050          # sample rate chuẩn hóa (slide: 16kHz cho speech, 44.1kHz cho music;
                     # động cơ là "noise" nên chọn mức trung gian, đủ băng thông cho hai)
FRAME_MS = 25        # độ dài frame (giống slide)
HOP_MS = 10           # bước nhảy frame (giống slide)
N_FFT = 2048
N_MFCC = 13           # dùng 13 hệ số MFCC như slide nêu


# ---------------------------------------------------------------------
# 0. Đọc file WAV nhiều kênh của dataset và tách audio / annotation
# ---------------------------------------------------------------------
def load_engine_audio(path, sr=SR):
    """
    Đọc file WAV 4 kênh (rdoerfler/procedural-engine-sounds):
      kênh 1-2 = stereo engine audio
      kênh 3   = RPM * 0.0001  (nhân 10000 để ra RPM thật)
      kênh 4   = Torque * 0.001 (nhân 1000 để ra Nm thật)
    Trả về: (audio_mono, rpm_series, torque_series)
    Nếu file chỉ có 1-2 kênh (đã tách sẵn), trả về rpm=torque=None.
    """
    y, orig_sr = librosa.load(path, sr=None, mono=False)
    if y.ndim == 1:
        audio = librosa.resample(y, orig_sr=orig_sr, target_sr=sr) if orig_sr != sr else y
        return audio, None, None

    n_ch = y.shape[0]
    audio_stereo = y[:2, :]
    audio_mono = librosa.to_mono(audio_stereo)
    if orig_sr != sr:
        audio_mono = librosa.resample(audio_mono, orig_sr=orig_sr, target_sr=sr)

    rpm = torque = None
    if n_ch >= 3:
        rpm = y[2, :] * 10000.0
    if n_ch >= 4:
        torque = y[3, :] * 1000.0

    return audio_mono, rpm, torque


# ---------------------------------------------------------------------
# 1. Đặc trưng miền thời gian  (slide 11-14)
# ---------------------------------------------------------------------
def frame_signal(x, sr=SR, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    frames = librosa.util.frame(x, frame_length=frame_len, hop_length=hop_len).T
    return frames, frame_len, hop_len


def average_energy(frames):
    """ E = (1/N) * sum(x(n)^2)   -- slide 11 """
    return np.mean(frames ** 2, axis=1)


def zero_crossing_rate(frames):
    """ ZCR = (1/2N) * sum |sgn(x(n)) - sgn(x(n-1))|   -- slide 11 """
    sgn = np.sign(frames)
    sgn[sgn == 0] = 0
    diff = np.abs(np.diff(sgn, axis=1))
    N = frames.shape[1]
    return np.sum(diff, axis=1) / (2 * N)


def silence_ratio(x, sr=SR, amp_thresh=None, min_silence_ms=50):
    """
    Silence: |x[n]| < A_thresh liên tục >= T_min mẫu (slide 12,14).
    A_thresh: ngưỡng tương đối = 5% biên độ RMS toàn file nếu không truyền vào.
    Trả về: tỷ lệ thời gian im lặng / tổng thời gian file (slide ví dụ: 8/60 = 0.133)
    """
    if amp_thresh is None:
        amp_thresh = 0.05 * np.sqrt(np.mean(x ** 2))
    is_silent = np.abs(x) < amp_thresh
    min_len = int(sr * min_silence_ms / 1000)

    # đếm các đoạn liên tục silent >= min_len
    silent_samples = 0
    run = 0
    for v in is_silent:
        if v:
            run += 1
        else:
            if run >= min_len:
                silent_samples += run
            run = 0
    if run >= min_len:
        silent_samples += run

    return silent_samples / len(x)


# ---------------------------------------------------------------------
# 2. Đặc trưng miền tần số  (slide 16-23)
# ---------------------------------------------------------------------
def stft_magnitude(x, sr=SR, n_fft=N_FFT, hop_ms=HOP_MS):
    hop_len = int(sr * hop_ms / 1000)
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_len, window="hann"))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return S, freqs


def spectral_centroid(S, freqs):
    """ SC = sum(f_k * |X(k)|) / sum(|X(k)|)   -- slide 20 (độ 'sáng' / brightness) """
    num = np.sum(freqs[:, None] * S, axis=0)
    den = np.sum(S, axis=0) + 1e-10
    return num / den


def spectral_bandwidth(S, freqs, db_thresh=3.0):
    """
    Bandwidth = hiệu giữa tần số cao nhất và thấp nhất của thành phần phổ dương,
    'dương' được định nghĩa là lớn hơn ngưỡng khoảng 3dB trên mức im lặng -- slide 19.
    Tính trên phổ trung bình toàn file cho đơn giản, trả về 1 giá trị/frame.
    """
    ref = S.max(axis=0, keepdims=True) + 1e-10
    db = 20 * np.log10(S / ref + 1e-10)
    mask = db > -db_thresh
    bw = np.zeros(S.shape[1])
    for i in range(S.shape[1]):
        active = freqs[mask[:, i]]
        bw[i] = (active.max() - active.min()) if len(active) > 0 else 0.0
    return bw


def harmonicity_and_f0(x, sr=SR, hop_ms=HOP_MS, fmin=30, fmax=2000):
    """
    Ước lượng f0 (tần số cơ bản, liên hệ trực tiếp RPM/60 x số xy-lanh) bằng thuật toán
    YIN (slide 21: 'Pitch (f0): fundamental frequency from autocorrelation, cepstrum, or
    YIN algorithm'). Harmonicity = tỷ lệ năng lượng tập trung ở các bội số f,2f,3f...
    (slide 22: 'spectral components mostly consist of the coefficients of the lowest
    frequency ... f, 2f, 3f, 4f').
    """
    hop_len = int(sr * hop_ms / 1000)
    f0 = librosa.yin(x, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_len)

    S, freqs = stft_magnitude(x, sr=sr, hop_ms=hop_ms)
    harmonicity = np.zeros(len(f0))
    n_frames = min(len(f0), S.shape[1])
    for i in range(n_frames):
        f = f0[i]
        if f <= 0 or np.isnan(f):
            harmonicity[i] = 0.0
            continue
        harmonics = [k * f for k in range(1, 6) if k * f < freqs[-1]]
        idx = [np.argmin(np.abs(freqs - h)) for h in harmonics]
        harm_energy = np.sum(S[idx, i])
        total_energy = np.sum(S[:, i]) + 1e-10
        harmonicity[i] = harm_energy / total_energy

    return f0[:n_frames], harmonicity[:n_frames]


def mfcc_features(x, sr=SR, n_mfcc=N_MFCC, hop_ms=HOP_MS):
    """
    MFCC theo đúng quy trình slide 21: STFT -> Mel filter bank -> log -> DCT -> 13 hệ số/frame
    """
    hop_len = int(sr * hop_ms / 1000)
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc, hop_length=hop_len)
    return mfcc.T  # (n_frames, n_mfcc)


# ---------------------------------------------------------------------
# 3. Aggregation -> vector đặc trưng mức segment (slide 8: "Aggregation")
# ---------------------------------------------------------------------
FEATURE_NAMES = (
    ["E_mean", "E_std", "ZCR_mean", "ZCR_std", "silence_ratio"] +
    ["centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std"] +
    ["f0_mean", "f0_std", "harmonicity_mean", "harmonicity_std"] +
    [f"mfcc{i+1}_mean" for i in range(N_MFCC)] +
    [f"mfcc{i+1}_std" for i in range(N_MFCC)]
)  # tổng = 5 + 4 + 4 + 26 = 39 chiều


def extract_feature_vector(path):
    """
    Hàm chính: đọc 1 file WAV -> trả về vector đặc trưng N=39 chiều (numpy array)
    theo đúng thứ tự trong FEATURE_NAMES.
    """
    audio, rpm, torque = load_engine_audio(path)
    frames, frame_len, hop_len = frame_signal(audio)

    E = average_energy(frames)
    ZCR = zero_crossing_rate(frames)
    sil = silence_ratio(audio)

    S, freqs = stft_magnitude(audio)
    centroid = spectral_centroid(S, freqs)
    bandwidth = spectral_bandwidth(S, freqs)
    f0, harmonicity = harmonicity_and_f0(audio)
    mfcc = mfcc_features(audio)

    n = min(len(E), len(ZCR), S.shape[1], len(f0), mfcc.shape[0])

    vec = np.concatenate([
        [E[:n].mean(), E[:n].std()],
        [ZCR[:n].mean(), ZCR[:n].std()],
        [sil],
        [centroid[:n].mean(), centroid[:n].std()],
        [bandwidth[:n].mean(), bandwidth[:n].std()],
        [f0[:n].mean(), f0[:n].std()],
        [harmonicity[:n].mean(), harmonicity[:n].std()],
        mfcc[:n].mean(axis=0),
        mfcc[:n].std(axis=0),
    ])

    extra = {
        "rpm_mean_from_channel": float(np.mean(rpm)) if rpm is not None else None,
        "torque_mean_from_channel": float(np.mean(torque)) if torque is not None else None,
    }
    return vec, extra


if __name__ == "__main__":
    # smoke test với tín hiệu tổng hợp (không cần file WAV thật)
    import soundfile as sf
    t = np.linspace(0, 3, SR * 3)
    f0_test = 80  # ~ động cơ 2 xy-lanh 4 kỳ ở tốc độ thấp
    synth = 0.4 * sum(np.sin(2 * np.pi * k * f0_test * t) / k for k in range(1, 6))
    synth += 0.02 * np.random.randn(len(t))
    sf.write("/tmp/test_engine.wav", synth.astype(np.float32), SR)

    vec, extra = extract_feature_vector("/tmp/test_engine.wav")
    print("Vector dim:", len(vec), "== len(FEATURE_NAMES):", len(FEATURE_NAMES))
    for name, val in zip(FEATURE_NAMES, vec):
        print(f"  {name:20s} = {val:.4f}")
