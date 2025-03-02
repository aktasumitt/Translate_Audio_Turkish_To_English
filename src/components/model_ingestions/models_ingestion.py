from src.components.model_ingestions.speech_recognation import whisper_model
from src.components.model_ingestions.text_to_speech import mms_tts_model
from src.components.model_ingestions.translate_text import marian_model
from src.entity.config_entity import ModelIngestionEntity
from src.exception.exception import ExceptionNetwork, sys
from src.logging.logger import logger

class ModelsIngestions:
    def __init__(self, config: ModelIngestionEntity):
        self.config = config

    def speech_recognation(self):
        try:
            logger.info("Speech recognition model ingestion is starting...")
            whisper_model.model_ingestion(
                save_path_model=self.config.speech_recognation_model_path,
                save_path_processor=self.config.speech_recognation_processor_path)
            logger.info("Speech recognition model ingestion completed successfully.")
        except Exception as e:
            raise ExceptionNetwork(e, sys)

    def translate_text(self):
        try:
            logger.info("Translation model ingestion is starting...")
            marian_model.model_ingestion(
                save_path_model=self.config.translator_model_path,
                save_path_tokenizer=self.config.translator_processor_path
            )
            logger.info("Translation model ingestion completed successfully.")
        except Exception as e:
            raise ExceptionNetwork(e, sys)

    def text_to_speech(self):
        try:
            logger.info("Text-to-Speech model ingestion is starting...")
            mms_tts_model.model_ingestion(
                save_path_model=self.config.text_to_speech_model_path,
                save_path_tokenizer=self.config.text_to_speech_model_path
            )
            logger.info("Text-to-Speech model ingestion completed successfully.")
        except Exception as e:
            raise ExceptionNetwork(e, sys)
