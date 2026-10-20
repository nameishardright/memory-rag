"""Offline speech extraction; source media and transcripts stay local."""
from pathlib import Path
import os
import threading

_MODEL = None
_LOCK = threading.Lock()


def model_directory():
    return Path(os.environ.get('MR_PRIVATE_WHISPER_MODEL') or
                Path.home() / 'tools' / 'subtitle-models' / 'whisper-large-v3-turbo')


def audio_info(path):
    import av
    with av.open(str(path)) as container:
        return {'has_audio': bool(container.streams.audio),
                'duration_seconds': float(container.duration or 0) / 1000000}


def _load_model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        path = model_directory()
        if not (path / 'model.bin').is_file():
            raise FileNotFoundError('Local speech model is unavailable')
        _MODEL = WhisperModel(str(path), device='cpu', compute_type='int8',
                             cpu_threads=4, num_workers=1, local_files_only=True)
    return _MODEL


def timestamp(seconds):
    value = max(0, int(seconds))
    return '%02d:%02d:%02d' % (value // 3600, value // 60 % 60, value % 60)


def transcribe_media(path):
    metadata = audio_info(path)
    if not metadata['has_audio']:
        return {'text': '', 'method': 'media_no_audio', 'warnings': [], **metadata}
    with _LOCK:
        segments, info = _load_model().transcribe(str(path), beam_size=3, vad_filter=True,
                                                 condition_on_previous_text=False,
                                                 word_timestamps=False)
        lines, count, weak = [], 0, 0
        for segment in segments:
            value = segment.text.strip()
            if not value:
                continue
            count += 1
            weak += float(segment.avg_logprob) < -1.0
            lines.append('[%s --> %s]\n%s' % (timestamp(segment.start), timestamp(segment.end), value))
        warnings = ['Automatic speech recognition; verify important names, dates and numbers against the source audio.']
        if weak:
            warnings.append('low_confidence_segments:' + str(weak))
        return {'text': '\n\n'.join(lines), 'method': 'local_speech_transcript' if lines else 'media_no_speech',
                'warnings': warnings if lines else [], 'language': info.language,
                'language_probability': float(info.language_probability), 'segments': count, **metadata}
