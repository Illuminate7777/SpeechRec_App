from pydub import AudioSegment

def convert_mp3_to_wav(input_file, output_file):
  """
  Converts an mp3 file to a wav file using pydub.

  Args:
    input_file: Path to the input mp3 file.
    output_file: Path to the output wav file.
  """
  # Open the mp3 file
  sound = AudioSegment.from_mp3(input_file)

  # Export the audio as a wav file
  sound.export(output_file, format="wav")

# Example usage
input_file = "Speech_Rec/Voice 002.mp3"
output_file = "Meeting.wav"
convert_mp3_to_wav(input_file, output_file)

print("Conversion complete!")
