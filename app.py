"""AudioTruth: AI Voice Authenticator

GPU-first local application for binary AI-generated-vs-human voice classification.
The model architecture and preprocessing parameters intentionally match the supplied
project implementation. The model examines a fixed 3-second segment; this program
keeps the Mel-Spectrogram as an internal feature representation but does not display it.
"""

from __future__ import annotations

import html
import logging
import os
import sys
import time
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as T


# =============================================================================
# Application configuration
# =============================================================================
APP_NAME = "AudioTruth: AI Voice Authenticator"
DEVELOPER_NAME = "Abdalla"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "audio_deepfake_model.pth"

TARGET_SAMPLE_RATE = 16_000
SEGMENT_SECONDS = 3
TARGET_SAMPLES = TARGET_SAMPLE_RATE * SEGMENT_SECONDS
N_MELS = 64
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 512
MAX_UPLOAD_MB = 150
MIN_AUDIO_SECONDS = 0.25
MAX_SOURCE_SAMPLE_RATE = 192_000

# The project agreement requires GPU inference. By default, the app will NOT
# silently fall back to CPU. This environment flag exists only for local testing.
ALLOW_CPU_FOR_TESTS = os.environ.get("AUDIOTRUTH_ALLOW_CPU", "0").strip() == "1"

# The existing training/app convention is class 0 = human/real, class 1 = AI/fake.
REAL_CLASS_INDEX = 0
FAKE_CLASS_INDEX = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("AudioTruth")


# =============================================================================
# CNN architecture
# IMPORTANT: Keep layer names and shapes aligned with the supplied checkpoint.
# =============================================================================
class DeepfakeDetectorCNN(nn.Module):
    """CNN architecture compatible with the provided model checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(x))


@dataclass
class AnalysisResult:
    result_html: str
    scores: dict[str, float] | None
    details_md: str


@dataclass
class AudioTruthEngine:
    model: DeepfakeDetectorCNN
    device: torch.device
    mel_transform: T.MelSpectrogram
    amplitude_to_db: T.AmplitudeToDB
    gpu_name: str
    cuda_version: str
    gpu_memory_gb: float

    def _prepare_waveform(self, audio_path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
        """Read audio safely and reproduce the training input shape and length."""
        path = Path(audio_path)
        if not path.is_file():
            raise ValueError("لم يتم العثور على الملف الصوتي. أعد رفع الملف ثم حاول مرة أخرى.")

        try:
            file_size = path.stat().st_size
        except OSError as exc:
            raise ValueError("تعذر الوصول إلى الملف الصوتي. اختر الملف مرة أخرى.") from exc

        if file_size <= 0:
            raise ValueError("الملف الصوتي فارغ.")
        if file_size > MAX_UPLOAD_MB * 1024 * 1024:
            raise ValueError(f"حجم الملف أكبر من الحد المسموح ({MAX_UPLOAD_MB} MB). اختر ملفًا أصغر.")

        try:
            # always_2d=True makes mono/stereo handling consistent: [frames, channels].
            audio_array, source_sr = sf.read(
                str(path), dtype="float32", always_2d=True
            )
        except Exception as exc:
            LOGGER.exception("Audio file could not be decoded")
            raise ValueError(
                "تعذرت قراءة الملف الصوتي. جرّب WAV أو MP3 صالحًا. "
                "إذا استمرت المشكلة، حوّل الملف إلى WAV PCM ثم أعد المحاولة."
            ) from exc

        if audio_array.size == 0 or audio_array.shape[0] == 0:
            raise ValueError("لا يحتوي الملف على عينات صوتية قابلة للتحليل.")
        if not isinstance(source_sr, (int, np.integer)) or source_sr < 1:
            raise ValueError("معدل أخذ العينات في الملف غير صالح.")
        if source_sr > MAX_SOURCE_SAMPLE_RATE:
            raise ValueError(
                f"معدل أخذ العينات مرتفع جدًا ({source_sr} Hz). "
                "حوّل الملف إلى 16–48 kHz ثم أعد المحاولة."
            )
        if audio_array.shape[1] > 32:
            raise ValueError("عدد القنوات في هذا الملف غير مدعوم.")
        if not np.isfinite(audio_array).all():
            raise ValueError("يحتوي الملف على قيم صوتية غير صالحة (NaN أو Infinity).")

        source_frames = int(audio_array.shape[0])
        source_channels = int(audio_array.shape[1])
        source_duration = source_frames / float(source_sr)
        if source_duration < MIN_AUDIO_SECONDS:
            raise ValueError(
                f"المقطع قصير جدًا ({source_duration:.2f} ثانية). "
                f"استخدم تسجيلًا مدته {MIN_AUDIO_SECONDS:.2f} ثانية على الأقل، ويفضل 3 ثوانٍ أو أكثر."
            )

        # Average channels to mono, matching the supplied pipeline.
        mono = audio_array.mean(axis=1, dtype=np.float32)
        waveform = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)

        # Resampling is performed on CPU; feature extraction and model inference use GPU.
        if int(source_sr) != TARGET_SAMPLE_RATE:
            resampler = _get_resampler(int(source_sr), TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)

        post_resample_samples = int(waveform.shape[-1])
        if post_resample_samples < TARGET_SAMPLES:
            waveform = torch.nn.functional.pad(
                waveform, (0, TARGET_SAMPLES - post_resample_samples)
            )
            segment_note = "أُضيف صمت إلى نهاية المقطع للوصول إلى 3 ثوانٍ."
        elif post_resample_samples > TARGET_SAMPLES:
            waveform = waveform[:, :TARGET_SAMPLES]
            segment_note = "تم تحليل أول 3 ثوانٍ فقط من التسجيل."
        else:
            segment_note = "تم تحليل التسجيل كاملًا (3 ثوانٍ)."

        waveform = waveform.contiguous()
        if waveform.shape != (1, TARGET_SAMPLES):
            raise RuntimeError(f"Unexpected waveform shape after preprocessing: {tuple(waveform.shape)}")

        metadata = {
            "filename": path.name,
            "source_sr": int(source_sr),
            "source_channels": source_channels,
            "source_duration": source_duration,
            "source_frames": source_frames,
            "segment_note": segment_note,
            "resampled": int(source_sr) != TARGET_SAMPLE_RATE,
        }
        return waveform, metadata

    def analyze(self, audio_path: str | None) -> AnalysisResult:
        """Run GPU-backed feature extraction and inference with readable errors."""
        if not audio_path:
            return AnalysisResult(
                result_html=_notice_card(
                    "بانتظار الملف الصوتي",
                    "ارفع ملف WAV أو MP3 صالحًا، ثم اضغط زر التحليل.",
                    tone="neutral",
                ),
                scores=None,
                details_md="لم يبدأ التحليل بعد.",
            )

        try:
            waveform_cpu, metadata = self._prepare_waveform(audio_path)

            # CUDA synchronization makes the displayed inference timing meaningful.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start_time = time.perf_counter()

            # Compatibility path: reproduce the previous app's CPU feature extraction,
            # then send the resulting tensor to GPU for the CNN inference.
            with torch.inference_mode():
                # Shape before batch dimension: [1, 64, time]. The CNN expects [B, 1, 64, time].
                mel = self.mel_transform(waveform_cpu)
                mel_db = self.amplitude_to_db(mel)
                model_input = mel_db.unsqueeze(0).to(self.device)
                if model_input.ndim != 4 or model_input.shape[1] != 1 or model_input.shape[2] != N_MELS:
                    raise RuntimeError(f"Unexpected model input shape: {tuple(model_input.shape)}")

                logits = self.model(model_input)
                if logits.shape != (1, 2):
                    raise RuntimeError(
                        f"The model returned shape {tuple(logits.shape)}; expected (1, 2). "
                        "Check that the checkpoint matches this model architecture."
                    )
                probabilities = torch.softmax(logits.float(), dim=1)[0]

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if not torch.isfinite(probabilities).all():
                raise RuntimeError("The model returned a non-finite score. Check the input and checkpoint.")

            real_score = float(probabilities[REAL_CLASS_INDEX].item() * 100.0)
            fake_score = float(probabilities[FAKE_CLASS_INDEX].item() * 100.0)
            is_fake = fake_score >= real_score

            result_html = _prediction_card(
                filename=metadata["filename"],
                is_fake=is_fake,
                real_score=real_score,
                fake_score=fake_score,
            )
            scores = {
                "Human speech · model score": max(0.0, min(1.0, real_score / 100.0)),
                "AI-generated · model score": max(0.0, min(1.0, fake_score / 100.0)),
            }

            duration_text = f"{metadata['source_duration']:.2f} s"
            source_text = f"{metadata['source_sr']:,} Hz · {metadata['source_channels']} channel(s)"
            device_text = f"{self.gpu_name} · CUDA {self.cuda_version}"
            # Keep the main presentation clean. Implementation details about the fixed
            # three-second analysis window are explained in the collapsed technical section.
            details_md = (
                "### تفاصيل التنفيذ | Runtime details\n"
                f"- **Device:** {device_text}\n"
                f"- **Inference time:** {elapsed_ms:.1f} ms (feature extraction + model)\n"
                f"- **Input:** {source_text}\n"
                f"- **Original duration:** {duration_text}"
            )

            LOGGER.info(
                "Analysis complete | file=%s | device=%s | real=%.2f%% | fake=%.2f%% | time_ms=%.1f",
                metadata["filename"], self.device, real_score, fake_score, elapsed_ms,
            )
            return AnalysisResult(result_html, scores, details_md)

        except ValueError as exc:
            LOGGER.warning("User/input validation message: %s", exc)
            return AnalysisResult(
                _notice_card("تعذر تحليل الملف", str(exc), tone="error"),
                None,
                "لم تكتمل عملية التحليل. راجع رسالة الخطأ وارفع ملفًا صالحًا.",
            )
        except Exception as exc:
            LOGGER.exception("Audio analysis failed")
            safe_message = html.escape(str(exc)[:500])
            return AnalysisResult(
                _notice_card(
                    "حدث خطأ أثناء التحليل",
                    "لم تكتمل عملية التحليل. راجع رسالة CMD التي تحتوي على تفاصيل الخطأ، ثم أرسلها للمساعدة في التشخيص. "
                    f"<br><small>Technical message: {safe_message}</small>",
                    tone="error",
                ),
                None,
                "لم تكتمل عملية التحليل. احتفظ برسالة CMD لتشخيص المشكلة.",
            )


@lru_cache(maxsize=8)
def _get_resampler(source_sr: int, target_sr: int) -> T.Resample:
    """Cache resamplers to avoid rebuilding filter kernels for repeated sample rates."""
    return T.Resample(orig_freq=source_sr, new_freq=target_sr).cpu()


def _notice_card(title: str, message: str, tone: str = "neutral") -> str:
    themes = {
        "neutral": ("#334155", "#e2e8f0"),
        "error": ("#b91c1c", "#fee2e2"),
    }
    border, text_color = themes.get(tone, themes["neutral"])
    return (
        '<section class="result-card" style="border-color:' + border + ';">'
        f'<h3 style="color:{text_color};">{html.escape(title)}</h3>'
        f'<p>{message}</p></section>'
    )


def _prediction_card(filename: str, is_fake: bool, real_score: float, fake_score: float) -> str:
    safe_filename = html.escape(filename)
    if is_fake:
        border = "#ef4444"
        badge_bg = "#991b1b"
        headline = "MODEL PREDICTION: AI-GENERATED"
        arabic = "رجّح النموذج أن الصوت مولّد أو مُعالج اصطناعيًا."
        tone = "red"
        selected_score = fake_score
    else:
        border = "#22c55e"
        badge_bg = "#166534"
        headline = "MODEL PREDICTION: HUMAN SPEECH"
        arabic = "رجّح النموذج أن الصوت أقرب إلى الكلام البشري الحقيقي."
        tone = "green"
        selected_score = real_score

    return f"""
    <section class="result-card prediction-card" style="border-color:{border};">
      <div class="result-topline">
        <span class="result-badge" style="background:{badge_bg};">{headline}</span>
        <span class="result-percent" style="color:{border};">{selected_score:.2f}%</span>
      </div>
      <h3 class="result-ar" dir="rtl">{arabic}</h3>
      <div class="score-track" aria-label="Model score">
        <div class="score-fill" style="width:{selected_score:.2f}%;background:{border};"></div>
      </div>
      <p class="result-file">File: <strong>{safe_filename}</strong></p>
    </section>
    """


def _load_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    """Load tensor weights safely and support common checkpoint wrappers."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "ضع audio_deepfake_model.pth بجانب app.py ثم شغّل البرنامج من جديد."
        )

    try:
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility fallback for older PyTorch versions. The expected file is a local checkpoint.
        checkpoint = torch.load(str(model_path), map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            "تعذر تحميل ملف الأوزان بأمان. تأكد أن audio_deepfake_model.pth هو ملف النموذج الصحيح "
            "ولم يتلف أثناء النقل."
        ) from exc

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    if not isinstance(state, dict) or not state:
        raise RuntimeError("ملف النموذج لا يحتوي على state_dict صالح.")

    # Allow checkpoints saved from DataParallel without changing architecture keys.
    if all(isinstance(key, str) and key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}

    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError("ملف النموذج يحتوي على عناصر لا تبدو كأوزان Tensor متوقعة.")
    return state


def create_engine(model_path: Path = DEFAULT_MODEL_PATH, require_gpu: bool = True) -> AudioTruthEngine:
    """Validate CUDA, load the exact checkpoint, initialize transforms, and warm up."""
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU REQUIRED BUT NOT AVAILABLE.\n"
            "لم يجد PyTorch بطاقة CUDA متاحة. لا يمكن تشغيل هذا الإصدار من AudioTruth وفق شرط GPU.\n"
            "تحقق من NVIDIA Driver ونسخة PyTorch CUDA ثم أعد اختبار torch.cuda.is_available()."
        )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        gpu_name = props.name
        gpu_memory_gb = props.total_memory / (1024 ** 3)
        cuda_version = str(torch.version.cuda or "unknown")
        # Keep PyTorch's default floating-point settings for repeatable behavior.
        # We deliberately avoid optional TF32 tuning while validating consistency
        # with the previous application.
    else:
        # Only possible when explicitly opted into testing mode.
        device = torch.device("cpu")
        gpu_name = "CPU TEST MODE (GPU NOT ACTIVE)"
        gpu_memory_gb = 0.0
        cuda_version = "not available"
        LOGGER.warning("CPU TEST MODE is active. Do not use this mode for the agreed GPU demonstration.")

    state_dict = _load_state_dict(Path(model_path))
    model = DeepfakeDetectorCNN()
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise RuntimeError(
            "أوزان النموذج لا تتطابق مع معمارية CNN داخل app.py. "
            "لم يتم تشغيل نموذج عشوائي أو غير مكتمل. احتفظ بملف الأوزان الأصلي وأرسل رسالة الخطأ."
        ) from exc

    model = model.to(device).eval()
    # Match the original app's default torchaudio settings exactly. Keep these
    # transforms on CPU for comparison consistency; the CNN inference remains on GPU.
    mel_transform = T.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    ).cpu()
    amplitude_to_db = T.AmplitudeToDB().cpu()

    engine = AudioTruthEngine(
        model=model,
        device=device,
        mel_transform=mel_transform,
        amplitude_to_db=amplitude_to_db,
        gpu_name=gpu_name,
        cuda_version=cuda_version,
        gpu_memory_gb=gpu_memory_gb,
    )

    # Smoke test: verify the actual feature-extraction + CNN path on the selected device.
    try:
        dummy_waveform = torch.zeros((1, TARGET_SAMPLES), dtype=torch.float32, device="cpu")
        with torch.inference_mode():
            dummy_mel = amplitude_to_db(mel_transform(dummy_waveform))
            dummy_input = dummy_mel.unsqueeze(0).to(device)
            test_logits = model(dummy_input)
        if test_logits.shape != (1, 2):
            raise RuntimeError(f"Unexpected warm-up output shape: {tuple(test_logits.shape)}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            "فشل اختبار التشغيل الأولي على الجهاز المحدد. لم تبدأ الواجهة لأن GPU/model smoke test لم ينجح."
        ) from exc

    LOGGER.info("Model loaded strictly from: %s", Path(model_path).resolve())
    LOGGER.info("PyTorch version: %s", torch.__version__)
    LOGGER.info("CUDA runtime reported by PyTorch: %s", cuda_version)
    LOGGER.info("Selected device: %s", device)
    LOGGER.info("GPU memory: %.2f GB", gpu_memory_gb)
    LOGGER.info("Warm-up CPU feature extraction + selected-device CNN inference succeeded; model output shape=(1, 2).")
    return engine


CUSTOM_CSS = """
:root { color-scheme: dark; }
body, .gradio-container { background: #080b12 !important; color: #e5e7eb !important; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
#main-shell { border: 1px solid #1e293b; border-radius: 22px; padding: 22px; background: linear-gradient(145deg,#0b1220,#080b12); }
#developer-tag { display: inline-block; float: right; border: 1px solid #3b82f6; border-radius: 999px; padding: 7px 14px; background: #10203c; color: #93c5fd; font-size: 13px; font-weight: 700; }
#main-title { clear: both; padding-top: 18px; text-align: center; font-size: clamp(26px,4vw,38px); font-weight: 850; letter-spacing: -.7px; color: #f8fafc; }
#subtitle { text-align: center; color: #9ca3af; line-height: 1.9; margin: 7px auto 22px; }
#hardware-card { border: 1px solid #166534; background: linear-gradient(100deg,#0b2119,#0f172a); border-radius: 14px; padding: 13px 16px; margin: 14px 0 22px; }
#hardware-title { color: #bbf7d0; font-weight: 800; letter-spacing: .3px; }
#hardware-desc { color: #86efac; font-size: 12px; margin-top: 4px; }
#action-button { min-height: 52px; font-weight: 800; border-radius: 12px; }
#result-panel { min-height: 180px; }
.result-card { background: linear-gradient(145deg,#111827,#0b1220); border: 1px solid #334155; border-radius: 16px; padding: 20px; margin: 4px 0; box-shadow: 0 8px 26px rgba(0,0,0,.18); overflow-wrap: anywhere; }
.result-topline { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; }
.result-badge { color: white; padding: 8px 11px; border-radius: 8px; font-size: 12px; font-weight: 850; letter-spacing: .2px; }
.result-percent { font-size: clamp(25px,4vw,34px); font-weight: 900; font-variant-numeric: tabular-nums; }
.result-ar { color: #e5e7eb; font-size: 16px; line-height: 1.95; margin-top: 18px; }
.score-track { height: 8px; background: #1f2937; border-radius: 99px; margin: 16px 0; overflow: hidden; }
.score-fill { height: 100%; border-radius: 99px; }
.result-file { color: #9ca3af; font-size: 12px; margin-top: 14px; }
.result-file strong { color: #e5e7eb; }
footer { display: none !important; }
@media (max-width: 680px) { #main-shell { padding: 12px; } .result-card { padding: 15px; } #developer-tag { float: none; } }
"""


def build_demo(engine: AudioTruthEngine) -> gr.Blocks:
    gpu_status = (
        f"{engine.gpu_name} | CUDA {engine.cuda_version} | "
        f"{engine.gpu_memory_gb:.1f} GB VRAM"
        if engine.device.type == "cuda"
        else "CPU TEST MODE — GPU is NOT active"
    )
    status_color = "#4ade80" if engine.device.type == "cuda" else "#fbbf24"

    with gr.Blocks(title=APP_NAME) as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        with gr.Column(elem_id="main-shell"):
            gr.HTML(f'<div id="developer-tag">👨‍💻 Developed by {html.escape(DEVELOPER_NAME)}</div>')
            gr.HTML(
                '<div id="main-title">AudioTruth: AI Voice Authenticator</div>'
                '<div id="subtitle" dir="rtl">نظام ذكي لتحليل الصوت وتصنيف الكلام إلى بشري أو مولّد اصطناعيًا</div>'
            )
            gr.HTML(
                f'<div id="hardware-card">'
                f'<div id="hardware-title" style="color:{status_color};">● GPU ENGINE: {"ACTIVE" if engine.device.type == "cuda" else "TEST MODE"}</div>'
                f'<div id="hardware-desc">{html.escape(gpu_status)}</div>'
                '</div>'
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=280):
                    audio_input = gr.Audio(
                        sources=["upload"],
                        type="filepath",
                        label="إدراج ملف صوتي للتحليل (WAV / MP3)",
                    )
                    analyze_button = gr.Button(
                        "⚡ تشغيل التحليل الصوتي",
                        variant="primary",
                        elem_id="action-button",
                    )
                    gr.Markdown(
                        "**خطوات الاستخدام | How to use**\n\n"
                        "1. ارفع ملفًا صوتيًا صالحًا.\n"
                        "2. اضغط زر التحليل.\n"
                        "3. راجع تصنيف النموذج ودرجاته."
                    )
                with gr.Column(scale=1, min_width=300):
                    result_output = gr.HTML(
                        value=_notice_card(
                            "جاهز للتحليل",
                            "ارفع ملفًا صوتيًا من الجهة اليسرى ثم اضغط تشغيل التحليل.",
                        ),
                        elem_id="result-panel",
                        label="تقرير التحليل",
                    )
                    confidence_output = gr.Label(
                        label="Model scores · درجات النموذج",
                        num_top_classes=2,
                    )
            details_output = gr.Markdown(
                "### حالة النظام | System status\nGPU جاهز. لم يتم تحليل ملف بعد."
            )
            with gr.Accordion("⚙️ الإعدادات والملاحظات التقنية | Settings & Technical Notes", open=False):
                gr.Markdown(
                    "### ملاحظات حول الدرجات | Model Scores\n\n"
                    "ملاحظة: النسبة هي **Model Score** من **Softmax**، وليست احتمالًا مُعايرًا أو دليلًا جنائيًا قاطعًا.\n\n"
                    "تعبّر الدرجة عن مخرجات النموذج عند مقارنة الفئتين اللتين تدرّب عليهما، وقد لا تعكس احتمالًا واقعيًا مُعايرًا.\n\n"
                    "---\n"
                    "### حدود الاستخدام | Model Limitations\n\n"
                    "- AudioTruth هو **experimental classifier** لتصنيف الأنماط الصوتية إلى Human Speech أو AI-generated. وقد يخطئ في بعض التسجيلات.\n"
                    "- قد تتأثر النتيجة بجودة التسجيل، والضوضاء، والضغط الصوتي، والتعديلات، أو استخدام مولّد صوت لم يظهر في بيانات التدريب.\n"
                    "- النتيجة لا تحدد هوية المتحدث، ولا تثبت وحدها الأصالة أو الاحتيال أو مصدر التسجيل.\n\n"
                    "---\n"
                    "### المعالجة الصوتية | Audio Processing\n\n"
                    f"- يحوّل النظام الإشارة إلى **Mono** وبمعدل **{TARGET_SAMPLE_RATE:,} Hz**.\n"
                    f"- يعتمد النموذج الحالي على مقطع ثابت مدته **{SEGMENT_SECONDS} ثوانٍ** لأن هذا هو طول المقاطع المستخدم في مسار التدريب الأصلي. إذا كان التسجيل أطول، يُحلل أول مقطع فقط؛ وإذا كان أقصر، يُضاف صمت عند الحاجة للوصول إلى الطول المطلوب.\n"
                    "- لذلك فإن النتيجة الحالية تمثل المقطع المستخدم في التحليل، ولا تلخّص بالضرورة التسجيل الطويل كاملًا. تحليل التسجيل كاملًا يحتاج إلى مسار مقاطع واختبارات إضافية، أو تدريب النموذج على استراتيجية مختلفة.\n"
                    "- تُستخدم خصائص **Mel-Spectrogram** داخليًا لتمثيل الصوت للنموذج، ولا تُعرض كصورة في الواجهة.\n"
                    "- عند توفر CUDA، تعمل شبكة **CNN inference** على GPU؛ وتبقى خطوة استخراج Mel-Spectrogram على CPU في هذه النسخة لمطابقة مسار المعالجة السابق أثناء التحقق من ثبات النتائج."
                )

        analyze_button.click(
            fn=lambda path: _run_analysis(engine, path),
            inputs=[audio_input],
            outputs=[result_output, confidence_output, details_output],
            concurrency_limit=1,
            show_progress="full",
        )
    return demo


def _run_analysis(
    engine: AudioTruthEngine, audio_path: str | None
) -> tuple[str, dict[str, float] | None, str]:
    result = engine.analyze(audio_path)
    return result.result_html, result.scores, result.details_md


def main() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("AudioTruth requires Python 3.10 or newer. Python 3.12 is recommended.")

    require_gpu = not ALLOW_CPU_FOR_TESTS
    try:
        engine = create_engine(DEFAULT_MODEL_PATH, require_gpu=require_gpu)
    except Exception as exc:
        LOGGER.exception("Startup validation failed")
        print("\n" + "=" * 76, file=sys.stderr)
        print("AudioTruth could not start.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("=" * 76 + "\n", file=sys.stderr)
        raise SystemExit(1) from exc

    demo = build_demo(engine)
    LOGGER.info("Starting local-only interface at http://127.0.0.1:7860")
    demo.queue(default_concurrency_limit=1, max_size=12)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        show_error=False,
    )


if __name__ == "__main__":
    main()
