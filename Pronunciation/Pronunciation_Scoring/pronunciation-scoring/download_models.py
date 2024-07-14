import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Create the models directory if it doesn't exist
os.makedirs("./models", exist_ok=True)
os.makedirs("./models/tokenizer", exist_ok=True)
os.makedirs("./models/model", exist_ok=True)

print("Downloading and saving tokenizer...")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer.save_pretrained("./models/tokenizer")
print("Tokenizer saved successfully.")

print("Downloading and saving model...")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.save_pretrained("./models/model")
print("Model saved successfully.")

print("Download and save process completed.")