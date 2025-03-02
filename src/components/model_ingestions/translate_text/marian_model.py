from transformers import MarianMTModel, MarianTokenizer
from src.exception.exception import ExceptionNetwork, sys
from src.logging.logger import logger

def model_ingestion(save_path_tokenizer, save_path_model):
    try:
        model_name = "Helsinki-NLP/opus-mt-tr-en"
        logger.info("Translation model (Turkish to English) downloading is starting...")

        # Tokenizer ve modeli yükle
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        # Daha güvenli bir kaydetme yöntemi
        tokenizer.save_pretrained(save_path_tokenizer)
        model.save_pretrained(save_path_model)
        logger.info("Translation model (Turkish to English) was saved successfully.")
    
    except Exception as e:
        raise ExceptionNetwork(e, sys)
