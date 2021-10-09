import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer

extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
decoder = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")


librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# load audio
audio_input, sample_rate = sf.read(librispeech_samples_ds[2]["file"])

# pad input values and return pt tensor
input_values = extractor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)

transcription = decoder.decode(predicted_ids[0])
print(transcription)