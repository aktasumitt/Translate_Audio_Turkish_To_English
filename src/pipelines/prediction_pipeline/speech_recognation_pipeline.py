from src.components.prediction.predictions import Predictions
from src.configuration.configuration import Configurations
from src.exception.exception import ExceptionNetwork,sys
from src.logging.logger import logger

class SpeechRecognationPredictionPipeline:
    def __init__(self):
        
        configuration = Configurations()
        self.config = configuration.prediction_config()


    def run_speech_recognation(self, turkish_audio_path):
        try:
            logger.info(f"Starting speech recognition for {turkish_audio_path}...")
            predictions = Predictions(config=self.config)
            predictions.speech_recognation(turkish_audio_path)
            logger.info("Speech recognition completed successfully.")
        except Exception as e:
            raise ExceptionNetwork(e,sys)


