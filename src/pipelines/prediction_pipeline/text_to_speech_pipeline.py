from src.components.prediction.predictions import Predictions
from src.configuration.configuration import Configurations
from src.exception.exception import ExceptionNetwork,sys
from src.logging.logger import logger

class TextToSpeechPipeline:
    def __init__(self):
        configuration = Configurations()
        self.config = configuration.prediction_config()

    def run_text_to_speech(self, english_text_path):
        try:
            logger.info(f"Starting text-to-speech for {english_text_path}...")
            predictions = Predictions(config=self.config)
            predictions.text_to_speech(english_text_path)
            logger.info("Text-to-Speech completed successfully.")
        except Exception as e:
            raise ExceptionNetwork(e,sys)
