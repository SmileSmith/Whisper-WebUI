"""
use pyannote to do speaker diarization

@Example: https://github.com/Majdoddin/nlp
@pyannote: https://github.com/pyannote/pyannote-audio
@bilibili: https://www.bilibili.com/video/BV1cN411h7E9/

streaming use https://github.com/juanmc2005/diart

"""
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torch
import time

spacermilli = 200


def get_audio_file(file_name: str):
    # 如果是.wav .mp3 等音频文件直接返回
    if file_name.endswith(".wav") or file_name.endswith(".mp3"):
        return file_name
    spacer = AudioSegment.silent(duration=spacermilli)
    audio = AudioSegment.from_file(file_name)
    audio = spacer.append(audio, crossfade=0)
    # 导出.wav格式，重命名文件后缀
    new_file_name = file_name.split(".")[0] + ".wav"
    audio.export(new_file_name, format="wav")
    return new_file_name


def get_audio_diarization(file_name: str):
    start_time = time.time()
    file_name = get_audio_file(file_name)
    diarization_list = []
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token="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")

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    diarization = pipeline(file_name)

    track_id = 1
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        diarization_list.append({
            "id": track_id,
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker
        })
        track_id += 1
    diarization_time = time.time() - start_time
    return diarization_list, diarization_time
