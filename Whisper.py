import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QVBoxLayout, QFileDialog, QCheckBox
from pydub import AudioSegment
import whisperx
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

def convert_mp4_to_mp3(mp4_path):
    mp3_path = mp4_path.replace(".mp4", ".mp3")
    audio = AudioSegment.from_file(mp4_path, format="mp4")
    audio.export(mp3_path, format="mp3")
    return mp3_path

def choose_file(diarization_enabled):
    file_path, _ = QFileDialog.getOpenFileName(None, "Select Audio File", "", "Audio Files (*.mp3 *.mp4 *.wav)")
    if file_path:
        if file_path.endswith(".mp4"):
            file_path = convert_mp4_to_mp3(file_path)

        if diarization_enabled:
            audio = whisperx.load_audio(file_path)

            # Attempt to load WhisperX models explicitly on CUDA
            try:
                whisper_model = whisperx.load_model("large-v3", device, compute_type="float16")
                align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
                diarize_model = whisperx.DiarizationPipeline(device=device)
                
                # Transcribe and align with CUDA
                result = whisper_model.transcribe(audio, batch_size=16)
                result = whisperx.align(result["segments"], align_model, metadata, audio, device)
                
                # Diarization with CUDA
                diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=5)
                result = whisperx.assign_word_speakers(diarize_segments, result)

                output_text = ""
                for segment in result["segments"]:
                    output_text += f"Speaker {segment['speaker']}: {segment['text']}\n\n"

            except ValueError as e:
                print(f"CUDA error in WhisperX components: {e}. Diarization will fall back to CPU.")
                whisper_model = whisperx.load_model("large-v3", "cpu", compute_type="float32")
                result = whisper_model.transcribe(audio)
                align_model, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")
                result = whisperx.align(result["segments"], align_model, metadata, audio, "cpu")
                diarize_model = whisperx.DiarizationPipeline(device="cpu")
                diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=5)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                
                output_text = ""
                for segment in result["segments"]:
                    output_text += f"Speaker {segment['speaker']}: {segment['text']}\n\n"
        else:
            result = pipe(file_path, return_timestamps=True, generate_kwargs={"language": "english"})
            output_text = result["text"]

        output_file_path, _ = QFileDialog.getSaveFileName(None, "Save Results", "", "Markdown Files (*.md);;Text Files (*.txt)")
        if output_file_path:
            with open(output_file_path, "w") as output_file:
                output_file.write(output_text)
            print(f"Results written to {output_file_path}")
        else:
            print("Output file not saved.")



class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Speech Recognition App")
        self.resize(600, 300)

        # Create a button, layout, and checkbox for speaker separation
        choose_button = QPushButton("Choose File")
        choose_button.clicked.connect(self.on_choose_file)
        choose_button.setStyleSheet("background-color: white; color: black; height: 30px; width: 150px;")

        self.diarization_checkbox = QCheckBox("Enable Speaker Separation")

        layout = QVBoxLayout()
        layout.addWidget(self.diarization_checkbox)
        layout.addWidget(choose_button)
        self.setLayout(layout)
        
        stylesheet = "background-color: black;"
        self.setStyleSheet(stylesheet)

        self.show()

    def on_choose_file(self):
        diarization_enabled = self.diarization_checkbox.isChecked()
        choose_file(diarization_enabled)

if __name__ == "__main__":
    app = QApplication([])
    window = AppWindow()
    app.exec_()
