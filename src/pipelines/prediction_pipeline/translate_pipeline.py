from src.components.prediction.predictions import Predictions
from src.configuration.configuration import Configurations
from src.exception.exception import ExceptionNetwork,sys
from src.logging.logger import logger

class TranslatePredictionPipeline:
    def __init__(self):
        configuration = Configurations()
        self.config = configuration.prediction_config()

    def run_translator(self, turkish_text_path):
        try:
            logger.info(f"Starting translation for {turkish_text_path}...")
            predictions = Predictions(config=self.config)
            predictions.translate_text(turkish_text_path)
            logger.info("Translation completed successfully.")
        except Exception as e:
            raise ExceptionNetwork(e,sys)
