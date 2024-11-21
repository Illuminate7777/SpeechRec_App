import whisperx
import gc



device = "cuda"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16"
audio_file = "Meeting.wav"
audio = whisperx.load_audio(audio_file)
model = whisperx.load_model("large-v3", device, compute_type=compute_type)
result = model.transcribe(audio, batch_size=batch_size)
#print(result["segments"]) # before alignment
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
diarize_model = whisperx.DiarizationPipeline(use_auth_token="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", device=device)
diarize_segments = diarize_model(audio, min_speakers=3, max_speakers=7)
diarize_segments.speaker.unique()
result = whisperx.assign_word_speakers(diarize_segments, result)
#print(diarize_segments)
#print(result["segments"])
text = result["segments"]
with open("output.md", "w", encoding="utf-8") as f:
  # Write the text content directly
  for segment in result["segments"]:
    text = segment["text"]
    speaker = segment["speaker"]

    f.write(f"Text: {text}\n")
    f.write(f"Speaker: {speaker}\n\n")

print("Results saved to output.md")