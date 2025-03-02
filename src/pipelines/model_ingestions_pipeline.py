from src.components.model_ingestions.models_ingestion import ModelsIngestions
from src.configuration.configuration import Configurations
from src.entity.config_entity import ModelIngestionEntity
from src.exception.exception import ExceptionNetwork,sys
from src.logging.logger import logger

class ModelIngestionPipeline:
    def __init__(self, config: ModelIngestionEntity):
        self.config = config

    def run_model_ingestion(self):
        try:
            logger.info("Starting model ingestion pipeline...")

            model_ingestions = ModelsIngestions(config=self.config)
            model_ingestions.speech_recognation()
            model_ingestions.translate_text()
            model_ingestions.text_to_speech()

            logger.info("Model ingestion completed successfully.")
        except Exception as e:
            raise ExceptionNetwork(e,sys)

if __name__ == "__main__":
    try:
        configurations = Configurations()
        model_ingestion_config = configurations.model_ingestion_config()

        model_ingestion_pipeline = ModelIngestionPipeline(model_ingestion_config)
        model_ingestion_pipeline.run_model_ingestion()

    except Exception as e:
        raise ExceptionNetwork(e,sys)
