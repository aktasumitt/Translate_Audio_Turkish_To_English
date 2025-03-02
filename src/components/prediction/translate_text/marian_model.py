from transformers import MarianMTModel, MarianTokenizer
from src.utils import save_as_json, load_json
from src.exception.exception import ExceptionNetwork, sys
from src.logging.logger import logger
from pathlib import Path

def model_load_and_predict(load_turkish_text_path: Path, translated_text_save_path: Path, load_model_path: Path, load_processor_path: Path):
    try:
        logger.info(f"Loading translation model from {load_model_path} and tokenizer from {load_processor_path}...")

        tokenizer = MarianTokenizer.from_pretrained(load_processor_path)
        model = MarianMTModel.from_pretrained(load_model_path)

        # Transkripti JSON'dan oku
        transcription = load_json(load_turkish_text_path)

        # Tokenizer ile inputları encode et
        input_ids = tokenizer(transcription, return_tensors="pt", padding=True, truncation=True).input_ids

        # Modeli çalıştır ve çıktı al
        translated_ids = model.generate(input_ids)

        # Çıktıyı decode et
        translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

        logger.info(f"Saving translated text to {translated_text_save_path}...")
        save_as_json(translated_text, save_path=translated_text_save_path)
        
        logger.info("Translation completed successfully.")
        return translated_text

    except Exception as e:
        raise ExceptionNetwork(e, sys)
