from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from src.exception.exception import ExceptionNetwork,sys
from src.logging.logger import logger

def model_ingestion(save_path_processor,save_path_model):
    try:
        model_name = "emre/whisper-medium-turkish-2"
        logger.info("Speech recognation model (whisper) downloading is starting...")
        # Model ve işlemciyi yükleme
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)

        # Daha güvenli bir kaydetme yöntemi
        processor.save_pretrained(save_path_processor)
        model.save_pretrained(save_path_model)
        logger.info("Speech recognation model (whisper) was saved")
        
    except Exception as e:
        raise ExceptionNetwork(e,sys)

    