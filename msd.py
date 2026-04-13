"""
===============================================================================
  StSy Audio Analysis Pipeline — 统一音频分析模块
  
  将音频从原始波形转化为 LLM 可直接推理的结构化语义表示。
  
  Pipeline 概览:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Step 1  加载音频 (16kHz for NN, 44.1kHz for DSP)             │
  │  Step 2  Essentia NN 语义标签 (mood/genre/instrument/A-V)     │
  │  Step 3  底层信号特征 (spectral/MFCC/HPCP/RMS/ZCR)           │
  │  Step 4  节奏时序分析 (BPM curve/beat grid/tempo evolution)   │
  │  Step 5  调性时序分析 (key trajectory/modulation detection)   │
  │  Step 6  音色时序分析 (brightness/texture/energy evolution)   │
  │  Step 7  结构段落检测 (MFCC novelty → section boundaries)    │
  │  Step 8  三层语义翻译 + 打包 + sanitize → JSON for LLM        │
  └─────────────────────────────────────────────────────────────────┘
  
  使用: 修改下方 CONFIG 区域 → python stsy_audio_pipeline.py
  
  输出: 每个音频文件 → 一个 JSON，可直接作为 LLM Judge 的输入
===============================================================================
"""

import os
import sys
import json
import math
import warnings
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

warnings.filterwarnings("ignore")

# =================================================================
#  CONFIG
# =================================================================

AUDIO_DIRS = [
    "/Volumes/T7shield/0-thesis-project/empathybgm/project",
    # 可以添加多个路径，也支持单个文件路径
    # "/path/to/another/folder",
    # "/path/to/single_file.wav",
]

OUTPUT_DIR  = "./stsy_output_project"
MODEL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OVERWRITE   = False          # True=已有 JSON 时覆盖
RECURSIVE   = True           # True=递归扫描子文件夹

# ─── 采样率────────────────────────────────────────

SR_NN   = 16000    # 神经网络模型采样率
SR_DSP  = 44100    # 信号处理采样率
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# =================================================================


# =================================================================
#  工具函数
# =================================================================

def hz_to_midi(f: float) -> Optional[float]:
    if f <= 0:
        return None
    return 69.0 + 12.0 * math.log2(f / 440.0)


def robust_stats(x: np.ndarray) -> dict:
    """计算稳健统计量，避免极端值干扰。"""
    if x.size == 0:
        return {"mean": None, "std": None, "p05": None, "p50": None, "p95": None}
    return {
        "mean": round(float(np.mean(x)), 4),
        "std":  round(float(np.std(x)), 4),
        "p05":  round(float(np.quantile(x, 0.05)), 4),
        "p50":  round(float(np.quantile(x, 0.50)), 4),
        "p95":  round(float(np.quantile(x, 0.95)), 4),
    }


def linear_slope(t: np.ndarray, y: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    tt = t - t.mean()
    denom = float(np.dot(tt, tt))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(tt, (y - y.mean())) / denom)


def moving_average(x: np.ndarray, k: int = 11) -> np.ndarray:
    if x.size < k:
        return x
    k = int(k) | 1  # 确保奇数
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def slice_by_time(t, y, start, end):
    mask = (t >= start) & (t < end)
    return t[mask], y[mask]


def compute_trend(values: np.ndarray) -> str:
    """计算一段数据的趋势标签。"""
    if len(values) < 3:
        return "stable"
    x = np.arange(len(values), dtype=float)
    x = x - x.mean()
    slope = float(np.dot(x, values - values.mean()) / (np.dot(x, x) + 1e-12))
    value_range = float(np.ptp(values)) + 1e-12
    relative_slope = slope * len(values) / value_range
    if relative_slope > 0.3:
        return "rising"
    elif relative_slope < -0.3:
        return "falling"
    return "stable"


def detect_key_events(times: np.ndarray, values: np.ndarray,
                      event_type: str, sigma: float = 2.0) -> list:
    """Z-score 突变点检测。"""
    if len(values) < 10:
        return []
    diff = np.diff(values)
    z = (diff - np.mean(diff)) / (np.std(diff) + 1e-12)
    events = []
    for i, zs in enumerate(z):
        if abs(zs) > sigma:
            direction = "increase" if zs > 0 else "decrease"
            events.append({
                "time_sec": round(float(times[i]), 1),
                "type": f"{event_type}_{direction}",
                "from": round(float(values[i]), 1),
                "to": round(float(values[i + 1]), 1),
                "magnitude": round(min(abs(float(zs)) / 4.0, 1.0), 2),
            })
    events.sort(key=lambda e: e["magnitude"], reverse=True)
    return events[:5]


def dedup_events(events: list, window_sec: float = 3.0) -> list:
    """同 domain 在 window_sec 内只保留 magnitude 最高的事件。

    输入必须已按 time 排序。贪心策略：按 magnitude 降序处理，
    高 magnitude 事件锁定其时间窗口，后续同 domain 事件若落入
    已锁定窗口则被丢弃。
    """
    if not events:
        return events
    # 按 magnitude 降序，优先保留强事件
    ranked = sorted(enumerate(events), key=lambda x: x[1]["magnitude"], reverse=True)
    kept_by_domain = {}   # domain -> list of kept times
    kept_indices = set()
    for orig_idx, evt in ranked:
        d = evt["domain"]
        t = evt["time"]
        # 检查该 domain 是否已有事件占据了附近的时间窗口
        if d in kept_by_domain:
            if any(abs(t - kt) < window_sec for kt in kept_by_domain[d]):
                continue  # 被更强的事件覆盖，丢弃
        kept_by_domain.setdefault(d, []).append(t)
        kept_indices.add(orig_idx)
    # 保持原始时间顺序
    return [evt for i, evt in enumerate(events) if i in kept_indices]


def segment_time_series(times: np.ndarray, values: np.ndarray,
                        n: int = 4, labels: list = None) -> list:
    """将时间序列等分为 n 段，返回每段统计量。"""
    if labels is None:
        labels = (["Opening", "Build-up", "Peak", "Resolution"] if n == 4
                  else [f"Part {i+1}" for i in range(n)])
    if len(times) == 0:
        return []
    dur = times[-1] - times[0]
    seg_dur = dur / n
    segments = []
    for i in range(n):
        s = times[0] + i * seg_dur
        e = s + seg_dur
        mask = (times >= s) & (times < e)
        v = values[mask]
        if len(v) == 0:
            continue
        segments.append({
            "label": labels[i] if i < len(labels) else f"Part {i+1}",
            "start_sec": round(float(s), 1),
            "end_sec": round(float(e), 1),
            "mean": round(float(np.mean(v)), 1),
            "std": round(float(np.std(v)), 1),
            "trend": compute_trend(v),
        })
    return segments


def segment_time_series_by_sections(times: np.ndarray, values: np.ndarray,
                                     sections: list) -> list:
    """按真实段落边界切分时间序列，返回每段统计量。

    与 segment_time_series (等时间 n 等分) 不同，这里使用 structure
    检测到的实际段落边界，使所有维度的分段描述与音乐结构对齐。
    """
    results = []
    for sec in sections:
        s, e = sec["start"], sec["end"]
        mask = (times >= s) & (times < e)
        v = values[mask]
        if len(v) == 0:
            continue
        results.append({
            "label": sec["label"],
            "start_sec": round(float(s), 1),
            "end_sec": round(float(e), 1),
            "mean": round(float(np.mean(v)), 1),
            "std": round(float(np.std(v)), 1),
            "trend": compute_trend(v),
        })
    return results


def sanitize_for_json(obj):
    """递归清除 NaN/Inf，替换为 None，确保 json.dump 不报错。"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.floating):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


# =================================================================
#  Step 1: 音频加载
# =================================================================

def load_audio(filepath: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    加载音频为两种采样率：
    - 16kHz  用于 Essentia NN 模型推理
    - 44.1kHz 用于信号处理 (节拍/调性/频谱)
    
    Returns: (audio_16k, audio_44k, duration_sec)
    """
    import essentia.standard as es
    
    audio_16k = es.MonoLoader(filename=filepath, sampleRate=SR_NN, resampleQuality=4)()
    audio_44k = es.MonoLoader(filename=filepath, sampleRate=SR_DSP, resampleQuality=4)()
    duration = float(len(audio_44k) / SR_DSP)
    
    return audio_16k, audio_44k, duration


# =================================================================
#  Step 2: Essentia NN 语义标签
# =================================================================

class EssentiaModels:
    """
    Essentia 预训练模型管理器。
    加载一次，复用于所有音频文件。
    """
    
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}
        self.available = False
        
    def load(self):
        """加载所有可用模型，缺失的跳过不报错。"""
        try:
            from essentia.standard import (
                TensorflowPredictEffnetDiscogs as EffNet,
                TensorflowPredict2D as P2D,
                TensorflowPredictMusiCNN as MusiCNN,
            )
        except ImportError:
            print("[Step 2] Essentia TF 模型不可用，将使用 fallback")
            return
        
        mp = lambda name: os.path.join(self.model_dir, name)
        
        # 骨干网络
        try:
            self.models["effnet"] = EffNet(
                graphFilename=mp("discogs-effnet-bs64-1.pb"),
                output="PartitionedCall:1"
            )
            self.models["musicnn"] = MusiCNN(
                graphFilename=mp("msd-musicnn-1.pb"),
                output="model/dense/BiasAdd"
            )
            self.available = True
        except Exception as e:
            print(f"[Step 2] 骨干模型加载失败: {e}")
            return
        
        # EffNet 分类头
        effnet_heads = {
            "mood_theme":  "mtg_jamendo_moodtheme-discogs-effnet-1",
            "instrument":  "mtg_jamendo_instrument-discogs-effnet-1",
            "genre":       "mtg_jamendo_genre-discogs-effnet-1",
        }
        for key, base in effnet_heads.items():
            pb = mp(f"{base}.pb")
            js_path = os.path.join(self.model_dir, f"{base}.json")
            if os.path.exists(pb):
                self.models[key] = P2D(graphFilename=pb)
                self.models[f"{key}_labels"] = self._load_labels(js_path)
        
        # MusiCNN 分类头
        musicnn_heads = {
            "av":              ("deam-msd-musicnn-2",              "model/Identity",  None),
            "mood_happy":      ("mood_happy-msd-musicnn-1",        "model/Softmax",   None),
            "mood_sad":        ("mood_sad-msd-musicnn-1",          "model/Softmax",   None),
            "mood_aggressive": ("mood_aggressive-msd-musicnn-1",   "model/Softmax",   None),
            "mood_relaxed":    ("mood_relaxed-msd-musicnn-1",      "model/Softmax",   None),
            "dance":           ("danceability-msd-musicnn-1",      "model/Softmax",   None),
            "vi":              ("voice_instrumental-msd-musicnn-1", "model/Softmax",  None),
        }
        for key, (base, output, inp) in musicnn_heads.items():
            pb = mp(f"{base}.pb")
            if os.path.exists(pb):
                kwargs = {"graphFilename": pb, "output": output}
                if inp:
                    kwargs["input"] = inp
                self.models[key] = P2D(**kwargs)
        
        loaded = [k for k in self.models if not k.endswith("_labels")]
        print(f"[Step 2] 已加载 {len(loaded)} 个模型: {', '.join(loaded)}")
    
    def _load_labels(self, path: str) -> list:
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return json.load(f).get("classes", [])
    
    def predict_multilabel(self, key: str, embeddings: np.ndarray) -> dict:
        """多标签预测，返回 {label: score} 字典。"""
        if key not in self.models:
            return {}
        preds = np.mean(self.models[key](embeddings), axis=0)
        labels = self.models.get(f"{key}_labels", [])
        if labels:
            d = {l: round(float(s), 4) for l, s in zip(labels, preds)}
            return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
        return {}
    
    def predict_scalar(self, key: str, embeddings: np.ndarray) -> Optional[float]:
        """标量预测 (如 mood_happy)。"""
        if key not in self.models:
            return None
        preds = np.mean(self.models[key](embeddings), axis=0)
        return round(float(preds[0]), 4)
    
    def predict_scalar_temporal(self, key: str, embeddings: np.ndarray) -> Optional[np.ndarray]:
        """逐 patch 标量预测，返回 shape=(n_patches,) 的时间序列。"""
        if key not in self.models:
            return None
        preds = self.models[key](embeddings)  # (n_patches, n_outputs)
        return preds[:, 0].astype(float)      # 取第一列
    
    def predict_av_temporal(self, embeddings: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """逐 patch 的 Arousal-Valence 预测，返回 (arousal_ts, valence_ts)。"""
        if "av" not in self.models:
            return None
        preds = self.models["av"](embeddings)  # (n_patches, 2)
        return preds[:, 0].astype(float), preds[:, 1].astype(float)


def extract_nn_tags(audio_16k: np.ndarray, models: EssentiaModels) -> dict:
    """
    Step 2: 用 Essentia NN 模型提取高层语义标签。
    
    输出:
    - 全局标签 (概率分布): mood_theme, genre, instrument, arousal, valence, ...
    - 时序标签 (逐 patch): arousal, valence, mood_happy/sad/aggressive/relaxed
      → 供下游 translate_nn_temporal_for_llm 做分段情感分析
    """
    if not models.available:
        return {"_method": "unavailable", "_note": "NN models not loaded"}
    
    duration = float(len(audio_16k)) / SR_NN
    
    # EffNet embeddings → mood/genre/instrument
    emb_effnet = models.models["effnet"](audio_16k)
    
    mood_theme = models.predict_multilabel("mood_theme", emb_effnet)
    genre      = models.predict_multilabel("genre", emb_effnet)
    instrument = models.predict_multilabel("instrument", emb_effnet)
    
    # MusiCNN embeddings → mood dimensions / arousal-valence
    emb_musicnn = models.models["musicnn"](audio_16k)
    n_patches = emb_musicnn.shape[0]
    
    # ── 全局标签 (向后兼容，保持原有行为) ──
    
    # Arousal-Valence (连续维度)
    arousal, valence = None, None
    if "av" in models.models:
        av = np.mean(models.models["av"](emb_musicnn), axis=0)
        arousal = round(float(av[0]), 4)
        valence = round(float(av[1]), 4)
    
    # 独立 mood 二分类
    mood_scores = {}
    for mood in ["happy", "sad", "aggressive", "relaxed"]:
        score = models.predict_scalar(f"mood_{mood}", emb_musicnn)
        if score is not None:
            mood_scores[mood] = score
    
    # Danceability
    danceability = models.predict_scalar("dance", emb_musicnn)
    
    # Voice / Instrumental
    voice_inst = {}
    if "vi" in models.models:
        vi = np.mean(models.models["vi"](emb_musicnn), axis=0)
        voice_inst = {
            "voice": round(float(vi[0]), 4),
            "instrumental": round(float(vi[1]), 4),
        }
    
    # ── 时序标签 (逐 patch，不做均值) ──
    # patch 时间轴: 将 n_patches 均匀映射到 [0, duration]
    patch_times = np.linspace(0, duration, n_patches, endpoint=False)
    
    temporal = {"times_sec": patch_times.tolist()}
    
    # Arousal-Valence 时序
    av_result = models.predict_av_temporal(emb_musicnn)
    if av_result is not None:
        a_ts, v_ts = av_result
        temporal["arousal"] = a_ts.tolist()
        temporal["valence"] = v_ts.tolist()
    
    # Mood 时序
    for mood in ["happy", "sad", "aggressive", "relaxed"]:
        mood_ts = models.predict_scalar_temporal(f"mood_{mood}", emb_musicnn)
        if mood_ts is not None:
            temporal[f"mood_{mood}"] = mood_ts.tolist()
    
    return {
        "_method": "essentia-nn",
        "mood_theme":    {k: v for k, v in list(mood_theme.items())[:8]},  # Top 8
        "genre":         {k: v for k, v in list(genre.items())[:6]},
        "instrument":    {k: v for k, v in list(instrument.items())[:8]},
        "mood_scores":   mood_scores,
        "arousal":       arousal,
        "valence":       valence,
        "danceability":  danceability,
        "voice_instrumental": voice_inst,
        "temporal":      temporal,
    }


# =================================================================
#  Step 3: 底层信号特征
# =================================================================

def extract_low_level(audio: np.ndarray, sr: int = SR_DSP,
                      frame_size: int = 4096, hop_size: int = 2048) -> dict:
    """
    Step 3: 一次帧遍历，提取全部底层特征时间序列。
    
    输出: 时间对齐的特征矩阵，供 Step 4-6 使用。
    """
    import essentia.standard as es
    
    window = es.Windowing(type="blackmanharris92")
    spectrum_algo = es.Spectrum()
    mfcc_algo = es.MFCC(numberCoefficients=13, inputSize=frame_size // 2 + 1)
    spectral_peaks = es.SpectralPeaks(
        sampleRate=sr, minFrequency=40, maxFrequency=5000,
        maxPeaks=60, magnitudeThreshold=1e-4, orderBy="magnitude"
    )
    hpcp_algo = es.HPCP(
        sampleRate=sr, size=12, referenceFrequency=440,
        minFrequency=40, maxFrequency=5000,
        bandPreset=True, normalized="unitMax"
    )
    
    t_frames, rms, zcr = [], [], []
    mfccs, hpcps = [], []
    centroid, flatness, rolloff, flux = [], [], [], []
    prev_spec = None
    
    def _centroid(spec):
        if spec.size == 0: return 0.0
        freqs = np.arange(spec.size, dtype=float) * (sr / 2.0) / (spec.size - 1)
        return float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
    
    def _flatness(spec):
        s = spec.astype(float) + 1e-12
        return float(np.exp(np.mean(np.log(s))) / (np.mean(s) + 1e-12))
    
    def _rolloff(spec, ratio=0.85):
        cumsum = np.cumsum(spec.astype(float))
        idx = int(np.searchsorted(cumsum, ratio * (cumsum[-1] + 1e-12)))
        idx = min(idx, spec.size - 1)
        return float(idx * (sr / 2.0) / (spec.size - 1))
    
    for i, frame in enumerate(es.FrameGenerator(audio, frameSize=frame_size,
                                                 hopSize=hop_size, startFromZero=True)):
        t = (i * hop_size) / sr
        t_frames.append(t)
        
        frame_f = frame.astype(np.float32)
        rms.append(float(np.sqrt(np.mean(frame_f ** 2) + 1e-12)))
        zcr.append(float(np.mean(frame_f[1:] * frame_f[:-1] < 0)))
        
        spec = spectrum_algo(window(frame_f))
        _, mfcc = mfcc_algo(spec)
        mfccs.append(mfcc)
        
        freqs, mags = spectral_peaks(spec)
        hpcps.append(hpcp_algo(freqs, mags))
        
        c = _centroid(spec)
        centroid.append(c)
        flatness.append(_flatness(spec))
        rolloff.append(_rolloff(spec))
        
        if prev_spec is not None:
            ds = spec.astype(float) - prev_spec
            flux.append(float(np.sqrt(np.mean(ds * ds))))
        else:
            flux.append(0.0)
        prev_spec = spec.astype(float)
    
    return {
        "t": np.array(t_frames),
        "rms": np.array(rms),
        "zcr": np.array(zcr),
        "mfcc": np.vstack(mfccs) if mfccs else np.zeros((0, 13)),
        "hpcp": np.vstack(hpcps) if hpcps else np.zeros((0, 12)),
        "centroid": np.array(centroid),
        "flatness": np.array(flatness),
        "rolloff": np.array(rolloff),
        "flux": np.array(flux),
        "frame_size": frame_size,
        "hop_size": hop_size,
    }


# =================================================================
#  Step 4: 节奏时序分析
# =================================================================

def analyze_rhythm(audio: np.ndarray, sr: int = SR_DSP) -> dict:
    """
    Step 4: 节奏分析 — 全局 + 时序。
    
    输出:
    - 全局 BPM、beat confidence
    - 逐 beat 的瞬时 BPM 曲线 (smoothed)
    - beat 时间戳序列
    """
    import essentia.standard as es
    
    bpm, beats, beat_conf, _, _ = es.RhythmExtractor2013(method="multifeature")(audio)
    beats = np.array(beats, dtype=float)
    
    # 瞬时 BPM
    if beats.size > 1:
        inst_bpm = 60.0 / np.clip(np.diff(beats), 1e-3, None)
        smoothed_bpm = moving_average(inst_bpm, k=11)
        beat_times = beats[1:].tolist()  # 与 inst_bpm 对齐
    else:
        inst_bpm = np.array([])
        smoothed_bpm = np.array([])
        beat_times = []
    
    # Essentia EBUR128 响度 (可选，用于能量包络)
    try:
        loud = es.LoudnessEBUR128()(audio)
        loudness_integrated = round(float(loud[0]), 2)
        loudness_range = round(float(loud[1]), 2)
    except Exception:
        loudness_integrated, loudness_range = None, None
    
    return {
        "bpm_global": round(float(bpm), 1),
        "beat_confidence": round(float(beat_conf), 3),
        "beats_sec": beats.tolist(),
        "beat_count": int(beats.size),
        "tempo_curve": {
            "times_sec": beat_times,
            "inst_bpm": inst_bpm.tolist(),
            "smoothed_bpm": smoothed_bpm.tolist(),
        },
        "loudness_integrated": loudness_integrated,
        "loudness_range": loudness_range,
    }


# =================================================================
#  Step 5: 调性时序分析
# =================================================================

def analyze_tonality(audio: np.ndarray, sr: int = SR_DSP, low: dict = None,
                     window_sec: float = 12.0, hop_sec: float = 6.0) -> dict:
    """
    Step 5: 调性分析 — 全局 key + 滑动窗口 key 轨迹。
    """
    import essentia.standard as es
    
    key, scale, strength = es.KeyExtractor(sampleRate=sr)(audio)
    
    # 滑动窗口调性检测
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    keys_t, keys, scales, strengths = [], [], [], []
    
    for start in range(0, max(0, len(audio) - win + 1), hop):
        seg = audio[start:start + win]
        k, sc, st = es.KeyExtractor(sampleRate=sr)(seg)
        keys_t.append(round(start / sr, 1))
        keys.append(str(k))
        scales.append(str(sc))
        strengths.append(round(float(st), 3))
    
    # HPCP 均值 (色度分布)
    hpcp_mean = []
    if low is not None and low["hpcp"].shape[0] > 0:
        hpcp_mean = np.mean(low["hpcp"], axis=0).tolist()
    
    return {
        "global_key": str(key),
        "global_scale": str(scale),
        "global_strength": round(float(strength), 3),
        "key_over_time": {
            "times_sec": keys_t,
            "keys": keys,
            "scales": scales,
            "strengths": strengths,
            "window_sec": window_sec,
            "hop_sec": hop_sec,
        },
        "hpcp_mean_12": [round(x, 4) for x in hpcp_mean],
    }


# =================================================================
#  Step 6: 音色时序分析
# =================================================================

def analyze_timbre(low: dict) -> dict:
    """
    Step 6: 音色分析 — 统计量 + 降采样时间序列。
    
    时间序列降采样到 ~1秒分辨率，供 Step 7 做分段分析。
    """
    centroid = low["centroid"]
    flatness = low["flatness"]
    rolloff = low["rolloff"]
    flux = low["flux"]
    zcr = low["zcr"]
    t = low["t"]
    
    # MFCC 统计
    mfcc = low["mfcc"]
    mfcc_stats = []
    if mfcc.shape[0] > 0:
        for i in range(mfcc.shape[1]):
            mfcc_stats.append(robust_stats(mfcc[:, i]))
    
    # 降采样时间序列 (~1s 分辨率)
    step = max(1, int(1.0 / (low["hop_size"] / SR_DSP)))
    
    # 动态/结构
    rms_db = 20.0 * np.log10(low["rms"] + 1e-12)
    
    time_series = {
        "times": t[::step].tolist(),
        "centroid": centroid[::step].tolist(),
        "flatness": flatness[::step].tolist(),
        "rms_db": rms_db[::step].tolist(),
    }
    dyn_range = float(np.quantile(rms_db, 0.95) - np.quantile(rms_db, 0.05)) if rms_db.size else 0.0
    energy_slope = linear_slope(t, rms_db)
    
    return {
        "spectral_centroid_hz": robust_stats(centroid),
        "spectral_flatness": robust_stats(flatness),
        "spectral_rolloff_hz": robust_stats(rolloff),
        "spectral_flux": robust_stats(flux),
        "zero_crossing_rate": robust_stats(zcr),
        "mfcc_stats": mfcc_stats,
        "time_series": time_series,
        "dynamic_range_db": round(dyn_range, 2),
        "energy_slope_db_per_sec": round(energy_slope, 4),
        "rms_db_stats": robust_stats(rms_db),
    }


# =================================================================
#  Step 6b: 基于内容的段落检测 (Structure Segmentation)
# =================================================================

def detect_structure(low: dict, rhythm: dict, tonal: dict,
                     timbre: dict, duration: float,
                     min_seg_sec: float = 8.0) -> dict:
    """
    Step 6b: 综合多维特征检测音乐段落边界。

    方法:
    - 用 MFCC 自相似矩阵 + checkerboard kernel 计算 novelty curve
    - 在 novelty 曲线上取峰值作为段落边界
    - 结合 rhythm/tonal/timbre 的突变事件交叉验证

    输出:
    - boundaries: 段落边界时间戳
    - sections: 每个段落的起止时间 + 综合描述
    """
    t = low["t"]
    mfcc = low["mfcc"]

    if mfcc.shape[0] < 20:
        return {
            "boundaries": [],
            "sections": [{"start": 0.0, "end": round(duration, 1),
                          "label": "Full track", "description": "Too short for segmentation"}],
        }

    # ── MFCC 自相似 novelty ──
    # 标准化 MFCC
    mfcc_norm = mfcc - np.mean(mfcc, axis=0, keepdims=True)
    norms = np.linalg.norm(mfcc_norm, axis=1, keepdims=True) + 1e-12
    mfcc_norm = mfcc_norm / norms

    # Checkerboard kernel novelty (Foote 2000)
    kernel_size = max(4, min(32, mfcc_norm.shape[0] // 8))
    half = kernel_size // 2
    # checkerboard: +1 in top-left/bottom-right, -1 in top-right/bottom-left
    kernel = np.ones((kernel_size, kernel_size))
    kernel[:half, half:] = -1
    kernel[half:, :half] = -1

    n_frames = mfcc_norm.shape[0]
    novelty = np.zeros(n_frames)
    for i in range(half, n_frames - half):
        r_start = max(0, i - half)
        r_end = min(n_frames, i + half)
        block = mfcc_norm[r_start:r_end] @ mfcc_norm[r_start:r_end].T
        k_size = r_end - r_start
        k_slice = kernel[:k_size, :k_size]
        novelty[i] = float(np.sum(block * k_slice))

    # 平滑 + 归一化
    novelty = moving_average(novelty, k=max(3, kernel_size // 2) | 1)
    if novelty.max() > 1e-8:
        novelty = novelty / novelty.max()

    # ── 峰值检测 → 段落边界 ──
    min_gap_frames = max(1, int(min_seg_sec / (low["hop_size"] / SR_DSP)))
    threshold = 0.3

    # 段落数上限: 防止长音频过度切分, 同时不限制短音频
    # min_seg_sec 已经约束了短音频, 这里主要约束长音频
    max_sections = min(8, max(6, int(duration / 20)))
    max_boundaries = max_sections - 1  # n 个边界 → n+1 个段落

    peaks = []
    for i in range(1, len(novelty) - 1):
        if novelty[i] > novelty[i-1] and novelty[i] > novelty[i+1]:
            if novelty[i] > threshold:
                peaks.append((i, novelty[i]))

    # 按强度排序，贪心去除过近的峰
    peaks.sort(key=lambda x: x[1], reverse=True)
    selected = []
    for idx, strength in peaks:
        if len(selected) >= max_boundaries:
            break
        if all(abs(idx - s) >= min_gap_frames for s, _ in selected):
            selected.append((idx, strength))
    selected.sort(key=lambda x: x[0])

    boundary_times = [round(float(t[idx]), 1) for idx, _ in selected]

    # ── 构建 sections ──
    starts = [0.0] + boundary_times
    ends = boundary_times + [round(duration, 1)]

    sections = []
    for si, (s, e) in enumerate(zip(starts, ends)):
        seg_dur_sec = e - s
        label = _infer_section_label(si, len(starts), seg_dur_sec, duration)
        sections.append({
            "start": s,
            "end": e,
            "duration": round(seg_dur_sec, 1),
            "label": label,
        })

    # ── 标签去重：相同标签加数字后缀区分 ──
    label_counts = Counter(sec["label"] for sec in sections)
    label_seen = {}
    for sec in sections:
        lbl = sec["label"]
        if label_counts[lbl] > 1:
            idx = label_seen.get(lbl, 0) + 1
            label_seen[lbl] = idx
            sec["label"] = f"{lbl} {idx}"

    return {
        "boundaries": boundary_times,
        "sections": sections,
    }


def _infer_section_label(index: int, total: int, seg_dur: float,
                         full_dur: float) -> str:
    """基于位置和时长启发式推断段落标签。"""
    ratio = seg_dur / full_dur if full_dur > 0 else 0
    rel_pos = index / max(1, total - 1) if total > 1 else 0.5

    if total == 1:
        return "Full track"
    if index == 0 and ratio < 0.2:
        return "Intro"
    if index == total - 1 and ratio < 0.2:
        return "Outro"
    if 0.3 <= rel_pos <= 0.7 and ratio >= 0.15:
        return "Core/Chorus"
    if rel_pos < 0.3:
        return "Verse A"
    if rel_pos > 0.7:
        return "Verse B / Coda"
    return f"Section {index + 1}"


# =================================================================
#  Step 7: 三层语义翻译 — 数值 → LLM 可推理的结构化文本
# =================================================================

def translate_rhythm_for_llm(rhythm: dict, duration: float,
                             sections: Optional[list] = None) -> dict:
    """
    将节奏数据转化为三层表示:
      Layer 1 — 一句话概览
      Layer 2 — 分段摘要 (按真实段落边界，若无则 fallback 到四等分)
      Layer 3 — 关键事件 (突变点)
    
    Args:
        sections: structure 检测到的段落列表 [{start, end, label}, ...]
                  若为 None，fallback 到等时间四等分。
    """
    bpm = rhythm["bpm_global"]
    conf = rhythm["beat_confidence"]
    smoothed = np.array(rhythm["tempo_curve"]["smoothed_bpm"], dtype=float)
    times = np.array(rhythm["tempo_curve"]["times_sec"], dtype=float)
    
    if len(smoothed) < 5:
        return {
            "overview": f"Tempo: {bpm:.0f} BPM (confidence: {conf:.2f}). Insufficient temporal data.",
            "segments": [],
            "events": [],
            "stats": {"bpm": round(bpm), "beat_confidence": round(conf, 2)},
        }
    
    min_len = min(len(times), len(smoothed))
    times, smoothed = times[:min_len], smoothed[:min_len]
    
    # 关键统计
    n_edge = max(1, len(smoothed) // 8)
    start_bpm = float(np.mean(smoothed[:n_edge]))
    end_bpm   = float(np.mean(smoothed[-n_edge:]))
    peak_bpm  = float(np.max(smoothed))
    low_bpm   = float(np.min(smoothed))
    
    # Layer 1: Overview
    parts = [f"Tempo: {bpm:.0f} BPM (confidence: {conf:.2f})"]
    
    if end_bpm > start_bpm * 1.15:
        parts.append(f"accelerates from ~{start_bpm:.0f} to ~{end_bpm:.0f}")
    elif start_bpm > end_bpm * 1.15:
        parts.append(f"decelerates from ~{start_bpm:.0f} to ~{end_bpm:.0f}")
    else:
        parts.append("tempo remains stable")
    
    # 中段峰值/低谷检测 — 以中位数为基准，捕捉 start≈end 时的中段波动
    # 双重条件: 偏离中位数 ×10% 且偏离首尾两端 ×5%，排除单纯加/减速的端点
    median_bpm = float(np.median(smoothed))
    
    if (peak_bpm > median_bpm * 1.10
            and peak_bpm > max(start_bpm, end_bpm) * 1.05):
        peak_t = float(times[np.argmax(smoothed)])
        parts.append(f"peaks at ~{peak_bpm:.0f} BPM around {peak_t:.0f}s")
    
    if (low_bpm < median_bpm * 0.90
            and low_bpm < min(start_bpm, end_bpm) * 0.95):
        valley_t = float(times[np.argmin(smoothed)])
        parts.append(f"dips to ~{low_bpm:.0f} BPM around {valley_t:.0f}s")
    
    if peak_bpm - low_bpm > 40:
        parts.append(f"wide range ({low_bpm:.0f}–{peak_bpm:.0f})")
    
    # Layer 2: Segments — 优先使用真实段落边界
    if sections and len(sections) > 0:
        raw_segments = segment_time_series_by_sections(times, smoothed, sections)
    else:
        raw_segments = segment_time_series(times, smoothed, n=4)
    
    seg_descs = []
    for seg in raw_segments:
        desc = f"{seg['label']} ({seg['start_sec']:.0f}–{seg['end_sec']:.0f}s): ~{seg['mean']:.0f} BPM, {seg['trend']}"
        if seg['std'] > 15:
            desc += f", variable (±{seg['std']:.0f})"
        seg_descs.append(desc)
    
    # Layer 3: Events
    events = detect_key_events(times, smoothed, "tempo", sigma=2.0)
    
    return {
        "overview": ". ".join(parts),
        "segments": seg_descs,
        "events": events,
        "stats": {
            "bpm": round(bpm),
            "beat_confidence": round(conf, 2),
            "bpm_range": [round(low_bpm), round(peak_bpm)],
            "start_bpm": round(start_bpm),
            "end_bpm": round(end_bpm),
        },
    }


def translate_tonality_for_llm(tonal: dict, duration: float,
                               sections: Optional[list] = None) -> dict:
    """
    将调性数据转化为三层表示:
      Layer 1 — 主调 + 整体调性行为
      Layer 2 — 调性路径 (合并去噪)
      Layer 2b — 按段落的调性摘要 (使用真实段落边界)
      Layer 3 — 转调事件
    
    Args:
        sections: structure 检测到的段落列表 [{start, end, label}, ...]
                  若为 None，fallback 到等时间四等分。
    """
    keys_list = tonal["key_over_time"]["keys"]
    scales_list = tonal["key_over_time"]["scales"]
    times_list = tonal["key_over_time"]["times_sec"]
    strengths_list = tonal["key_over_time"]["strengths"]
    
    dominant = f"{tonal['global_key']} {tonal['global_scale']}"
    
    if not keys_list:
        return {
            "overview": f"Key: {dominant} (strength: {tonal['global_strength']:.2f}). No temporal data.",
            "path": [],
            "segments": [],
            "events": [],
            "stats": {"dominant_key": dominant, "strength": tonal["global_strength"]},
        }
    
    # 构建调性路径 (合并连续相同段)
    path = []
    cur_key, cur_start, cur_strengths = None, None, []
    
    for key, scale, t, st in zip(keys_list, scales_list, times_list, strengths_list):
        full = f"{key} {scale}"
        if full != cur_key:
            if cur_key is not None:
                path.append({
                    "key": cur_key,
                    "start": round(cur_start, 1),
                    "end": round(t, 1),
                    "duration": round(t - cur_start, 1),
                    "strength": round(float(np.mean(cur_strengths)), 2),
                })
            cur_key, cur_start, cur_strengths = full, t, [st]
        else:
            cur_strengths.append(st)
    
    if cur_key is not None:
        path.append({
            "key": cur_key,
            "start": round(cur_start, 1),
            "end": round(duration, 1),
            "duration": round(duration - cur_start, 1),
            "strength": round(float(np.mean(cur_strengths)), 2),
        })
    
    # 去噪: 过滤 < 5秒的短暂调性
    stable = [s for s in path if s["duration"] >= 5 or len(path) == 1]
    noise_count = len(path) - len(stable)
    
    # Layer 1: Overview
    parts = [f"Key: {dominant} (strength: {tonal['global_strength']:.2f})"]
    n_mod = len(stable) - 1
    if n_mod == 0:
        parts.append("tonally static throughout")
    elif n_mod <= 3:
        parts.append(f"modulates: {' → '.join(s['key'] for s in stable)}")
    else:
        parts.append(f"complex movement with {n_mod} modulations")
    if noise_count > 0:
        parts.append(f"({noise_count} brief fluctuations filtered)")
    
    # Layer 2: Path descriptions
    path_descs = []
    for s in stable:
        stability = "stable" if s["strength"] > 0.5 else "weak"
        path_descs.append(
            f"{s['key']} ({s['start']:.0f}–{s['end']:.0f}s, {s['duration']:.0f}s, {stability})"
        )
    
    # Layer 3: Modulation events
    events = []
    for i in range(1, len(stable)):
        prev, curr = stable[i - 1], stable[i]
        from_mode = prev["key"].split()[-1] if " " in prev["key"] else ""
        to_mode = curr["key"].split()[-1] if " " in curr["key"] else ""
        
        if from_mode != to_mode:
            mod_type = "darkening" if to_mode == "minor" else "brightening"
        else:
            mod_type = "parallel shift"
        
        events.append({
            "time_sec": curr["start"],
            "from": prev["key"],
            "to": curr["key"],
            "type": mod_type,
        })
    
    # Layer 2b: 按段落的调性摘要 — 优先使用真实段落边界
    if sections and len(sections) > 0:
        seg_boundaries = sections
    else:
        # fallback: 四等分
        seg_dur = duration / 4
        seg_boundaries = [
            {"start": i * seg_dur, "end": (i + 1) * seg_dur,
             "label": lbl}
            for i, lbl in enumerate(["Opening", "Build-up", "Peak", "Resolution"])
        ]
    
    tonal_segments = []
    for sec in seg_boundaries:
        seg_start, seg_end = sec["start"], sec["end"]
        seg_label = sec["label"]
        # 找出落在该段落时间窗口内的所有 key 窗口
        seg_keys = []
        seg_strs = []
        for k, sc, t, st in zip(keys_list, scales_list, times_list, strengths_list):
            if seg_start <= t < seg_end:
                seg_keys.append(f"{k} {sc}")
                seg_strs.append(st)
        if not seg_keys:
            tonal_segments.append(
                f"{seg_label} ({seg_start:.0f}–{seg_end:.0f}s): {dominant}, inferred"
            )
        else:
            counter = Counter(seg_keys)
            dom_key = counter.most_common(1)[0][0]
            avg_str = float(np.mean(seg_strs))
            stability = "stable" if avg_str > 0.5 else "weak"
            n_unique = len(counter)
            extra = f", {n_unique} key(s) detected" if n_unique > 1 else ""
            tonal_segments.append(
                f"{seg_label} ({seg_start:.0f}–{seg_end:.0f}s): {dom_key}, {stability}{extra}"
            )
    
    return {
        "overview": ". ".join(parts),
        "path": path_descs,
        "segments": tonal_segments,
        "events": events,
        "stats": {
            "dominant_key": dominant,
            "strength": tonal["global_strength"],
            "n_modulations": n_mod,
            "path_keys": [s["key"] for s in stable],
        },
    }


def translate_timbre_for_llm(timbre: dict, duration: float,
                             sections: Optional[list] = None) -> dict:
    """
    将音色数据转化为三层表示:
      Layer 1 — 整体音色特性
      Layer 2 — 分段音色演变 (按真实段落边界)
      Layer 3 — 音色突变事件
    
    Args:
        sections: structure 检测到的段落列表 [{start, end, label}, ...]
                  若为 None，fallback 到等时间四等分。
    """
    centroid_mean = timbre["spectral_centroid_hz"]["mean"]
    flatness_mean = timbre["spectral_flatness"]["mean"]
    dyn_range = timbre["dynamic_range_db"]
    slope = timbre["energy_slope_db_per_sec"]
    
    # 质感描述 (基于 centroid + flatness 联合)
    if centroid_mean is None:
        brightness = "unknown"
    elif centroid_mean > 3000:
        brightness = "bright/metallic"
    elif centroid_mean > 1800:
        brightness = "balanced"
    else:
        brightness = "dark/warm"
    
    if flatness_mean is None:
        texture = "unknown"
    elif flatness_mean > 0.15:
        texture = "noisy/distorted"
    elif flatness_mean > 0.06:
        texture = "textured"
    else:
        texture = "clean/harmonic"
    
    # 动态描述
    if dyn_range > 25:
        dynamics = "massive contrast (quiet→loud)"
    elif dyn_range > 15:
        dynamics = "moderate contrast"
    else:
        dynamics = "compressed/flat"
    
    if slope > 0.3:
        energy_trend = "building up"
    elif slope < -0.1:
        energy_trend = "decaying"
    else:
        energy_trend = "steady"
    
    # Layer 1
    parts = [
        f"Timbre: {brightness}, {texture} texture",
        f"dynamics: {dynamics} ({dyn_range:.0f}dB range)",
        f"energy: {energy_trend}",
    ]
    
    # Layer 2: 分段 centroid 演变 — 优先使用真实段落边界
    ts = timbre.get("time_series", {})
    centroid_ts = ts.get("centroid", [])
    times_ts = ts.get("times", [])
    
    seg_descs = []
    if centroid_ts and times_ts and len(centroid_ts) > 8:
        t_arr = np.array(times_ts)
        c_arr = np.array(centroid_ts)
        if sections and len(sections) > 0:
            raw_segments = segment_time_series_by_sections(t_arr, c_arr, sections)
        else:
            raw_segments = segment_time_series(t_arr, c_arr, n=4)
        for seg in raw_segments:
            b = "bright" if seg["mean"] > 2500 else "balanced" if seg["mean"] > 1500 else "dark"
            seg_descs.append(
                f"{seg['label']} ({seg['start_sec']:.0f}–{seg['end_sec']:.0f}s): {b}, {seg['trend']}"
            )
    
    # Layer 2b: 分段 RMS dB 能量描述
    rms_db_ts = ts.get("rms_db", [])
    energy_seg_descs = []
    if rms_db_ts and times_ts and len(rms_db_ts) > 8:
        t_arr = np.array(times_ts)
        rms_arr = np.array(rms_db_ts)
        if sections and len(sections) > 0:
            energy_segments = segment_time_series_by_sections(t_arr, rms_arr, sections)
        else:
            energy_segments = segment_time_series(t_arr, rms_arr, n=4)
        for seg in energy_segments:
            mean_db = seg["mean"]
            if mean_db > -10:
                level = "loud"
            elif mean_db > -25:
                level = "moderate"
            else:
                level = "quiet"
            energy_seg_descs.append(
                f"{seg['label']} ({seg['start_sec']:.0f}–{seg['end_sec']:.0f}s): "
                f"{level} ({mean_db:.0f}dB), {seg['trend']}"
            )
    
    # Layer 3: 音色突变
    events = []
    if centroid_ts and times_ts and len(centroid_ts) > 10:
        events = detect_key_events(
            np.array(times_ts), np.array(centroid_ts), "timbre", sigma=2.0
        )
    
    return {
        "overview": ". ".join(parts),
        "segments": seg_descs,
        "energy_segments": energy_seg_descs,
        "events": events,
        "stats": {
            "centroid_mean_hz": round(centroid_mean, 1) if centroid_mean else None,
            "flatness_mean": round(flatness_mean, 4) if flatness_mean else None,
            "dynamic_range_db": round(dyn_range, 1),
            "energy_slope": round(slope, 3),
        },
    }


# =================================================================
#  Step 7b: NN 情感时序翻译 — 全局静态 → 分段动态
# =================================================================

def _describe_av_point(arousal: float, valence: float) -> str:
    """将 Arousal-Valence 坐标转化为情绪象限标签。

    DEAM 模型输出范围为 [1, 9]，中性点为 5.0。
    基于 Russell (1980) 的环形情感模型将二维空间划分为象限。
    """
    # Russell's circumplex model — DEAM 9-point scale, neutral=5.0
    HIGH = 5.5   # 高于中性 + 0.5 → 高唤醒/高效价
    LOW  = 4.5   # 低于中性 − 0.5 → 低唤醒/低效价
    if arousal > HIGH:
        if valence > HIGH:
            return "excited/joyful"
        elif valence < LOW:
            return "tense/aggressive"
        else:
            return "alert/energetic"
    elif arousal < LOW:
        if valence > HIGH:
            return "calm/serene"
        elif valence < LOW:
            return "sad/depressed"
        else:
            return "relaxed/subdued"
    else:
        if valence > HIGH:
            return "content/pleasant"
        elif valence < LOW:
            return "uneasy/melancholic"
        else:
            return "neutral"


def translate_nn_temporal_for_llm(nn_tags: dict, duration: float,
                                   sections: Optional[list] = None) -> dict:
    """
    将 NN 时序标签 (arousal/valence/mood 逐 patch) 转化为三层表示:
      Layer 1 — 整体情感弧线概览
      Layer 2 — 按段落的情感摘要 (与 rhythm/tonality/timbre 对齐)
      Layer 3 — 情感突变事件

    如果 nn_tags 中没有 temporal 数据 (e.g. 模型不可用)，返回空结构。
    """
    temporal = nn_tags.get("temporal")
    if not temporal or "arousal" not in temporal:
        return {
            "overview": None,
            "segments": [],
            "events": [],
            "stats": {},
        }

    times = np.array(temporal["times_sec"], dtype=float)
    arousal = np.array(temporal["arousal"], dtype=float)
    valence = np.array(temporal["valence"], dtype=float)

    if len(times) < 5:
        return {
            "overview": "Insufficient temporal patches for emotional trajectory.",
            "segments": [],
            "events": [],
            "stats": {},
        }

    # ── Layer 1: 整体情感弧线概览 ──
    a_mean, v_mean = float(np.mean(arousal)), float(np.mean(valence))
    overall_mood = _describe_av_point(a_mean, v_mean)

    n_edge = max(1, len(arousal) // 8)
    a_start, a_end = float(np.mean(arousal[:n_edge])), float(np.mean(arousal[-n_edge:]))
    v_start, v_end = float(np.mean(valence[:n_edge])), float(np.mean(valence[-n_edge:]))

    start_mood = _describe_av_point(a_start, v_start)
    end_mood = _describe_av_point(a_end, v_end)

    parts = [f"Emotion: overall {overall_mood} (A={a_mean:.2f}, V={v_mean:.2f})"]
    if start_mood != end_mood:
        parts.append(f"arc: {start_mood} → {end_mood}")
    else:
        parts.append("emotionally consistent throughout")

    # 加入 mood 维度补充 (如果有的话)
    mood_summary = []
    for mood in ["happy", "sad", "aggressive", "relaxed"]:
        key = f"mood_{mood}"
        if key in temporal:
            m_arr = np.array(temporal[key], dtype=float)
            m_mean = float(np.mean(m_arr))
            if m_mean > 0.6:
                mood_summary.append(f"{mood} ({m_mean:.0%})")
    if mood_summary:
        parts.append(f"dominant moods: {', '.join(mood_summary)}")

    # ── Layer 2: 按段落的情感摘要 ──
    seg_descs = []
    if sections and len(sections) > 0:
        for sec in sections:
            s, e = sec["start"], sec["end"]
            mask = (times >= s) & (times < e)
            a_seg = arousal[mask]
            v_seg = valence[mask]
            if len(a_seg) == 0:
                continue
            seg_a = float(np.mean(a_seg))
            seg_v = float(np.mean(v_seg))
            seg_mood = _describe_av_point(seg_a, seg_v)
            trend_a = compute_trend(a_seg)
            trend_v = compute_trend(v_seg)

            desc = (
                f"{sec['label']} ({s:.0f}–{e:.0f}s): "
                f"{seg_mood} (A={seg_a:.2f} {trend_a}, V={seg_v:.2f} {trend_v})"
            )
            # 加入该段最突出的 mood 维度
            seg_moods = []
            for mood in ["happy", "sad", "aggressive", "relaxed"]:
                key = f"mood_{mood}"
                if key in temporal:
                    m_arr = np.array(temporal[key], dtype=float)
                    m_seg = m_arr[mask]
                    if len(m_seg) > 0 and float(np.mean(m_seg)) > 0.6:
                        seg_moods.append(mood)
            if seg_moods:
                desc += f" [{', '.join(seg_moods)}]"
            seg_descs.append(desc)

    # ── Layer 3: 情感突变事件 ──
    events = []
    # Arousal 突变
    a_events = detect_key_events(times, arousal, "arousal", sigma=2.0)
    for ev in a_events:
        a_from, a_to = ev["from"], ev["to"]
        if ev["type"] == "arousal_increase":
            desc = f"Energy/arousal surges ({a_from:.2f}→{a_to:.2f})"
        else:
            desc = f"Energy/arousal drops ({a_from:.2f}→{a_to:.2f})"
        events.append({
            "time_sec": ev["time_sec"],
            "type": ev["type"],
            "desc": desc,
            "magnitude": ev["magnitude"],
        })
    # Valence 突变
    v_events = detect_key_events(times, valence, "valence", sigma=2.0)
    for ev in v_events:
        v_from, v_to = ev["from"], ev["to"]
        if ev["type"] == "valence_increase":
            desc = f"Mood brightens ({v_from:.2f}→{v_to:.2f})"
        else:
            desc = f"Mood darkens ({v_from:.2f}→{v_to:.2f})"
        events.append({
            "time_sec": ev["time_sec"],
            "type": ev["type"],
            "desc": desc,
            "magnitude": ev["magnitude"],
        })
    events.sort(key=lambda x: x["magnitude"], reverse=True)
    events = events[:5]  # 最多 5 个情感事件

    return {
        "overview": ". ".join(parts),
        "segments": seg_descs,
        "events": events,
        "stats": {
            "arousal_mean": round(a_mean, 3),
            "valence_mean": round(v_mean, 3),
            "arousal_range": [round(float(np.min(arousal)), 3),
                              round(float(np.max(arousal)), 3)],
            "valence_range": [round(float(np.min(valence)), 3),
                              round(float(np.max(valence)), 3)],
            "overall_quadrant": overall_mood,
            "arc": f"{start_mood} → {end_mood}" if start_mood != end_mood else "stable",
        },
    }


# =================================================================
#  Step 8: 质量校验 + 打包
# =================================================================

def check_audio_quality(timbre: dict, rhythm: dict) -> dict:
    """
    多指标噪声检测 — 替代单一阈值。
    
    三个指标加权投票:
    - 频谱平坦度 (高=噪声)          权重 0.55 — 最可靠的噪声指标
    - 节拍置信度 (低=非音乐)         权重 0.15 — 对非流行音乐常误报
    - 动态范围 (低=恒定噪声)         权重 0.30 — 可靠的音乐/噪声区分

    动态范围反制: 当 DR > 20dB 时 (明确有动态起伏的真实音频),
    将 confidence 异常度折半, 避免自由节奏的配乐被误判。
    """
    flatness = timbre["spectral_flatness"]["mean"] or 0
    beat_conf = rhythm["beat_confidence"]
    dyn_range = timbre["dynamic_range_db"]
    
    # 各指标异常度 [0, 1]
    flat_anom = max(0, min(1, (flatness - 0.15) / 0.15)) if flatness > 0.15 else 0
    conf_anom = max(0, min(1, (0.3 - beat_conf) / 0.3)) if beat_conf < 0.3 else 0
    dyn_anom  = max(0, min(1, (5 - dyn_range) / 5)) if dyn_range < 5 else 0
    
    # 动态范围反制: 高 DR 说明不是恒定噪声, confidence 低更可能是节奏复杂
    if dyn_range > 20:
        conf_anom *= 0.5
    
    noise_score = 0.55 * flat_anom + 0.15 * conf_anom + 0.30 * dyn_anom
    
    if noise_score > 0.6:
        tag = "NOISE/CORRUPTED"
    elif noise_score > 0.3:
        tag = "LOW_QUALITY"
    else:
        tag = "OK"
    
    return {
        "quality_tag": tag,
        "noise_score": round(noise_score, 3),
        "is_reliable": noise_score < 0.3,
        "details": {
            "flatness_anomaly": round(flat_anom, 3),
            "confidence_anomaly": round(conf_anom, 3),
            "dynamics_anomaly": round(dyn_anom, 3),
        },
    }


def assemble_llm_input(filepath: str, duration: float,
                       nn_tags: dict, rhythm: dict, tonal: dict,
                       timbre: dict, structure: dict) -> dict:
    """
    Step 8: 打包所有分析结果为 LLM Judge 的最终输入格式。
    
    设计原则:
    - LLM 看到的第一级是自然语言概览
    - 第二级是结构化的分段描述 (基于真实段落边界，而非固定等分)
    - 第三级是精确的关键事件
    - 原始统计量放在最后作为参考
    
    *** 统一分段策略 ***:
    timeline 和 detail.*.segments 都使用 structure 检测的真实段落边界，
    确保所有维度 (rhythm/tonality/timbre/energy/emotion) 在同一套段落划分下对齐。
    如果 structure 检测失败 (e.g. 音频过短)，fallback 到等时间四等分。
    """
    # 取得真实段落边界
    real_sections = structure.get("sections", [])
    
    # 将三个维度的分段描述对齐到同一套边界
    rhythm_llm = translate_rhythm_for_llm(rhythm, duration, sections=real_sections)
    tonal_llm  = translate_tonality_for_llm(tonal, duration, sections=real_sections)
    timbre_llm = translate_timbre_for_llm(timbre, duration, sections=real_sections)
    emotion_llm = translate_nn_temporal_for_llm(nn_tags, duration, sections=real_sections)
    
    # ── 综合概览 (LLM 最先看到的内容，结构化 dict 便于按维度引用) ──
    overview = {
        "rhythm": rhythm_llm["overview"],
        "tonality": tonal_llm["overview"],
        "timbre": timbre_llm["overview"],
        "emotion": emotion_llm["overview"],  # NN 情感弧线 (时序)
        "semantic": None,                    # NN 全局标签 (静态)
    }
    
    # NN 语义标签概览
    if nn_tags.get("_method") == "essentia-nn":
        tag_parts = []
        moods = nn_tags.get("mood_theme", {})
        if moods:
            top_moods = list(moods.items())[:3]
            tag_parts.append("Mood: " + ", ".join(f"{k} ({v:.0%})" for k, v in top_moods))
        genres = nn_tags.get("genre", {})
        if genres:
            top_genres = list(genres.items())[:3]
            tag_parts.append("Genre: " + ", ".join(f"{k} ({v:.0%})" for k, v in top_genres))
        if nn_tags.get("arousal") is not None:
            tag_parts.append(f"Arousal: {nn_tags['arousal']:.2f}, Valence: {nn_tags['valence']:.2f}")
        vi = nn_tags.get("voice_instrumental", {})
        if vi:
            if vi.get("instrumental", 0) > 0.7:
                tag_parts.append("Instrumental")
            elif vi.get("voice", 0) > 0.7:
                tag_parts.append("Contains vocals")
        if tag_parts:
            overview["semantic"] = ". ".join(tag_parts)
    
    # ── 统一时间线 (基于 structure 检测的真实段落边界) ──
    # 每个 section 聚合 rhythm/tonality/timbre/energy 四个维度的描述，
    # 它们在 translate_*_for_llm 中已按相同的 sections 边界生成，
    # 这里只需按顺序一一对应拼装即可。
    
    n_sections = len(real_sections)
    timeline = []
    for i, sec in enumerate(real_sections):
        entry = {
            "label": sec["label"],
            "period": f"{sec['start']:.0f}–{sec['end']:.0f}s",
            "duration_sec": round(sec["end"] - sec["start"], 1),
        }
        if i < len(rhythm_llm["segments"]):
            entry["rhythm"] = rhythm_llm["segments"][i]
        if i < len(tonal_llm["segments"]):
            entry["tonality"] = tonal_llm["segments"][i]
        if i < len(timbre_llm["segments"]):
            entry["timbre"] = timbre_llm["segments"][i]
        if i < len(timbre_llm.get("energy_segments", [])):
            entry["energy"] = timbre_llm["energy_segments"][i]
        if i < len(emotion_llm["segments"]):
            entry["emotion"] = emotion_llm["segments"][i]
        timeline.append(entry)
    
    # ── 关键事件合并 + 语义化描述 + 按时间排序 ──
    all_events = []
    for e in rhythm_llm.get("events", []):
        bpm_from, bpm_to = e["from"], e["to"]
        if e["type"] == "tempo_increase":
            desc = f"Tempo surges from {bpm_from:.0f} to {bpm_to:.0f} BPM — energy intensifies"
        else:
            desc = f"Tempo drops from {bpm_from:.0f} to {bpm_to:.0f} BPM — energy pulls back"
        all_events.append({"time": e["time_sec"], "domain": "rhythm",
                           "magnitude": e.get("magnitude", 0),
                           "desc": desc})
    for e in tonal_llm.get("events", []):
        mod_type = e["type"]
        if mod_type == "darkening":
            mood_shift = "mood darkens"
        elif mod_type == "brightening":
            mood_shift = "mood brightens"
        else:
            mood_shift = "harmonic color shifts"
        all_events.append({"time": e["time_sec"], "domain": "tonality",
                           "magnitude": 0.5,
                           "desc": f"Key change: {e['from']} → {e['to']} — {mood_shift}"})
    for e in timbre_llm.get("events", []):
        c_from, c_to = e["from"], e["to"]
        if e["type"] == "timbre_increase":
            if c_to > 3000:
                qual = "texture becomes bright/metallic"
            else:
                qual = "texture opens up"
        else:
            if c_to < 1500:
                qual = "texture becomes dark/warm"
            else:
                qual = "texture thins out"
        all_events.append({"time": e["time_sec"], "domain": "timbre",
                           "magnitude": e.get("magnitude", 0),
                           "desc": qual})
    all_events.sort(key=lambda x: x["time"])
    
    # 加入情感突变事件
    for e in emotion_llm.get("events", []):
        all_events.append({
            "time": e["time_sec"],
            "domain": "emotion",
            "magnitude": e.get("magnitude", 0),
            "desc": e["desc"],
        })
    all_events.sort(key=lambda x: x["time"])
    
    # nn_semantic_tags: 去掉原始时序数组 (太大)，只保留全局标签
    # 时序数据已经通过 emotion_llm 翻译为语义描述
    nn_tags_clean = {k: v for k, v in nn_tags.items() if k != "temporal"}
    
    return {
        "meta": {
            "duration_sec": round(duration, 2),
        },
        "overview": overview,
        "structure": structure,
        "timeline": timeline,
        "critical_moments": dedup_events(all_events)[:12],
        "nn_semantic_tags": nn_tags_clean,
        "detail": {
            "rhythm": rhythm_llm,
            "tonality": tonal_llm,
            "timbre": timbre_llm,
            "emotion": emotion_llm,
        },
    }


# =================================================================
#  主 Pipeline
# =================================================================

def analyze_single(filepath: str, models: EssentiaModels) -> dict:
    """
    分析单个音频文件，返回完整的 LLM 输入 JSON。
    """
    print(f"  [1/8] Loading audio...", end="", flush=True)
    audio_16k, audio_44k, duration = load_audio(filepath)
    print(f" {duration:.1f}s")
    
    print(f"  [2/8] NN semantic tags...", end="", flush=True)
    nn_tags = extract_nn_tags(audio_16k, models)
    n_tags = sum(1 for v in nn_tags.values() if v and v != "essentia-nn")
    print(f" {n_tags} dimensions")
    
    print(f"  [3/8] Low-level features...", end="", flush=True)
    low = extract_low_level(audio_44k, SR_DSP)
    print(f" {low['t'].size} frames")
    
    print(f"  [4/8] Rhythm analysis...", end="", flush=True)
    rhythm = analyze_rhythm(audio_44k, SR_DSP)
    print(f" {rhythm['bpm_global']} BPM, {rhythm['beat_count']} beats")
    
    print(f"  [5/8] Tonality analysis...", end="", flush=True)
    tonal = analyze_tonality(audio_44k, SR_DSP, low)
    print(f" {tonal['global_key']} {tonal['global_scale']}")
    
    print(f"  [6/8] Timbre analysis...", end="", flush=True)
    timbre = analyze_timbre(low)
    print(f" centroid={timbre['spectral_centroid_hz']['mean']:.0f}Hz")
    
    print(f"  [7/8] Structure detection...", end="", flush=True)
    structure = detect_structure(low, rhythm, tonal, timbre, duration)
    print(f" {len(structure['sections'])} sections")
    
    print(f"  [8/8] Assembling LLM input...", end="", flush=True)
    result = assemble_llm_input(filepath, duration, nn_tags, rhythm, tonal,
                                timbre, structure)
    print(f" done")
    
    return result


# =================================================================
#  文件扫描
# =================================================================

def scan_audio_files(dirs: list) -> list:
    """扫描 AUDIO_DIRS 中的所有音频文件，支持文件夹和单文件混合。"""
    files = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            print(f"  ⚠ 路径不存在，跳过: {d}")
            continue
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(str(p))
        elif p.is_dir():
            if RECURSIVE:
                files.extend(str(f) for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
            else:
                files.extend(str(f) for f in p.glob("*") if f.suffix.lower() in AUDIO_EXTS)
    return sorted(set(files))


def main():
    # 检查模型目录
    if not os.path.isdir(MODEL_DIR):
        print(f"models/ 目录不存在: {MODEL_DIR}")
        print("请先下载 Essentia 模型文件。NN 标签将使用 fallback。")
    
    # 扫描文件
    files = scan_audio_files(AUDIO_DIRS)
    if not files:
        print("未找到音频文件，请检查 AUDIO_DIRS")
        sys.exit(1)
    
    print(f"找到 {len(files)} 个音频文件:")
    for f in files[:10]:
        print(f"  • {f}")
    if len(files) > 10:
        print(f"  ... 及其他 {len(files) - 10} 个文件")
    print()
    
    # 加载模型 (只加载一次)
    models = EssentiaModels(MODEL_DIR)
    models.load()
    print()
    
    # 输出目录
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 逐文件分析
    ok, fail = 0, 0
    for i, fp in enumerate(files, 1):
        name = os.path.basename(fp)
        out_path = out_dir / f"{Path(fp).stem}.json"
        
        if out_path.exists() and not OVERWRITE:
            print(f"[{i}/{len(files)}] {name} — 已存在，跳过")
            ok += 1
            continue
        
        print(f"[{i}/{len(files)}] {name}")
        try:
            result = analyze_single(fp, models)
            result = sanitize_for_json(result)
            
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"  → {out_path}\n")
            ok += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}\n")
            fail += 1
    
    print(f"\n{'='*60}")
    print(f"  完成: {ok} 成功, {fail} 失败")
    print(f"  输出: {out_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
