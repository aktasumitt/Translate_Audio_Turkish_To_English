from transformers import VitsModel, AutoTokenizer
from src.utils import load_json
from src.exception.exception import ExceptionNetwork, sys
from src.logging.logger import logger
import torch
import torchaudio
from pathlib import Path

def model_load_and_predict(load_english_text_path: Path, translated_audio_save_path: Path, load_model_path, load_processor_path):
    try:
        logger.info(f"Loading Text-to-Speech model from {load_model_path} and tokenizer from {load_processor_path}...")
        
        tokenizer = AutoTokenizer.from_pretrained(load_processor_path)
        model = VitsModel.from_pretrained(load_model_path)

        translated_text = load_json(load_english_text_path)

        inputs = tokenizer(translated_text, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs).waveform

        logger.info(f"Saving generated speech to {translated_audio_save_path}...")
        torchaudio.save(translated_audio_save_path, sample_rate=model.config.sampling_rate, src=output)

    except Exception as e:
        raise ExceptionNetwork(e, sys)
