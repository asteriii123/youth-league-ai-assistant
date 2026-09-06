"""
语音转文字服务。

负责：

1. 加载Whisper模型
2. 视频提取音频
3. 音频转文字


技术：

faster-whisper
FFmpeg


流程：

mp4/mp3

↓

wav

↓

Whisper

↓

transcript文本
"""
import subprocess
from pathlib import Path

from app.core.config import settings


class TranscriptionError(Exception):
    pass


_model = None


def get_whisper_model():
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError("faster-whisper尚未安装，请重新安装后端依赖") from exc
        _model = WhisperModel(settings.whisper_model, device=settings.whisper_device, compute_type=settings.whisper_compute_type)
    return _model


def prepare_audio(source: Path) -> tuple[Path, bool]:
    if source.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return source, False
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise TranscriptionError("未检测到FFmpeg，安装后才能处理视频文件")
    target_dir = settings.converted_dir / "meetings"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{source.stem}.wav"
    try:
        subprocess.run([ffmpeg, "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", str(target)], check=True, capture_output=True, timeout=1800)
    except (subprocess.SubprocessError, OSError) as exc:
        target.unlink(missing_ok=True)
        raise TranscriptionError("FFmpeg无法提取视频音轨，请检查文件是否完整") from exc
    return target, True


def transcribe_media(source: Path) -> str:
    if not settings.whisper_enabled:
        raise TranscriptionError("本地音视频转写已在配置中关闭")
    audio_path, temporary = prepare_audio(source)
    try:
        segments, _ = get_whisper_model().transcribe(str(audio_path), language="zh", vad_filter=True, beam_size=5)
        text = "".join(segment.text.strip() for segment in segments).strip()
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError("本地转写失败，请检查文件或Whisper模型") from exc
    finally:
        if temporary:
            audio_path.unlink(missing_ok=True)
    if not text:
        raise TranscriptionError("没有识别到可用的中文语音内容")
    return text
