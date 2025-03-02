from transformers import VitsModel, AutoTokenizer
from src.exception.exception import ExceptionNetwork, sys
from src.logging.logger import logger

def model_ingestion(save_path_tokenizer, save_path_model):
    try:
        model_name = "facebook/mms-tts-eng"
        logger.info("Text-to-Speech model (MMS-TTS) downloading is starting...")

        model = VitsModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Daha güvenli bir kaydetme yöntemi
        tokenizer.save_pretrained(save_path_tokenizer)
        model.save_pretrained(save_path_model)
        logger.info("Text-to-Speech model (MMS-TTS) was saved successfully.")
    
    except Exception as e:
        raise ExceptionNetwork(e, sys)
