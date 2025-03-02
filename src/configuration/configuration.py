from src.constants import CONFIG_FILE
from src.entity.config_entity import ModelIngestionEntity,PredictionEntity
from src.utils import load_yaml



class Configurations():
    def __init__(self):
        self.config=load_yaml(CONFIG_FILE)
        
    def model_ingestion_config(self):
        config=self.config.model_ingestions
        
        return ModelIngestionEntity(speech_recognation_model_path=config.whisper_model_save_path,
                            speech_recognation_processor_path=config.whisper_processor_save_path,
                            translator_model_path=config.marian_model_save_path,
                            translator_processor_path=config.marian_processor_save_path, 
                            text_to_speech_model_path=config.mms_model_save_path,
                            text_to_speech_processor_path=config.mms_processor_save_path)
        
    def prediction_config(self):
        config_model=self.config.model_ingestions
        config_predict=self.config.predictions
        
        return PredictionEntity(speech_recognation_model_path=config_model.whisper_model_save_path,
                        speech_recognation_processor_path=config_model.whisper_processor_save_path,
                        translator_model_path=config_model.marian_model_save_path,
                        translator_processor_path=config_model.marian_processor_save_path, 
                        text_to_speech_model_path=config_model.mms_model_save_path,
                        text_to_speech_processor_path=config_model.mms_processor_save_path,
                        recognated_text_save_path=config_predict.recognated_text_save_path,
                        translated_text_path=config_predict.translated_text_path,
                        translated_audio_save_path=config_predict.translated_audio_save_path,
                        main_turkish_audio_path=config_predict.main_turkish_audio_path)


