from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler_tts_mini_v0.1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

text = """ My favorite thing about building at Eder Denver, we used to call it Eder Denver with my friend that I went there with, really was the atmosphere. I think the atmosphere was like a rock concert where everyone was in the same page and everyone was passionate about the same thing called blockchain. And we were all in that same community. It really built the foundation on the project that I'm pursuing on, which is the A to B project. And I could really say last year it lit a newfound passion in me in blockchain. I wasn't really interested in blockchain, but last year after it went, I was interested enough to create a club for my own university. And for this year, as I go, not by myself as an an individual but now as a member of a club I'm excited to grow that fire and grow the hype for the year of the sport well"""

prompt = text

description = "A female speaker with a slightly low-pitched voice, Clear and precise tone."

input_ids = tokenizer(description, return_tensors="pt", max_length= None).input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt", max_length= None).input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)