import os
import torch
import torchaudio
from pyannote.audio import Pipeline
from qwen_asr import Qwen3ASRModel

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.getenv("HF_TOKEN"),
)

QWEN_DEVICE = "cuda:0" if torch.cuda.is_available() else "mps"

qwen_model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map=QWEN_DEVICE,
)


def diarize_audio(audio_path: str) -> list[dict]:
    """Run speaker diarization on an audio file and return speaker turns."""
    output = pipeline(audio_path)

    return [
        {
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker,
        }
        for turn, speaker in output.speaker_diarization
    ]


def transcribe_qwen(diarization_result: list[dict], audio: str) -> list[dict]:
    """Transcribe each diarized speaker turn with Qwen3-ASR, in the TRANSCRIPTION_SCHEMA format."""
    waveform, sr = torchaudio.load(audio)
    waveform = waveform.mean(dim=0).numpy()

    segments = [
        waveform[int(turn["start"] * sr):int(turn["end"] * sr)]
        for turn in diarization_result
    ]
    results = qwen_model.transcribe(
        audio=[(segment, sr) for segment in segments],
        language=["English"] * len(segments),
    )

    transcript = []
    for turn, result in zip(diarization_result, results):
        minutes, seconds = divmod(int(turn["start"]), 60)
        transcript.append({
            "timestamp": f"{minutes:02d}:{seconds:02d}",
            "speakerID": str(turn["speaker"]),
            "text": str(result.text),
        })

    return transcript


if __name__ == "__main__":
    import sys

    diarization_result = diarize_audio(sys.argv[1])
    for segment in transcribe_qwen(diarization_result, sys.argv[1]):
        print(f"[{segment['timestamp']}] {segment['speakerID']}: {segment['text']}")