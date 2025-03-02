from src.components.prediction.speech_recognation import whisper_prediction
from src.components.prediction.text_to_speech import mms_tts_prediction
from src.components.prediction.translate_text import marian_model
from src.entity.config_entity import PredictionEntity
from src.exception.exception import ExceptionNetwork, sys
from src.logging.logger import logger
from pathlib import Path


class Predictions:
    def __init__(self, config: PredictionEntity):
        self.config = config

    def speech_recognation(self, turkish_audio_path: Path):
        try:
                
                whisper_prediction.model_load_and_predict(load_model_path=self.config.speech_recognation_model_path,
                                                        load_processor_path=self.config.speech_recognation_processor_path,
                                                        load_turkish_audio_path=turkish_audio_path,
                                                        transcription_save_path=self.config.recognated_text_save_path)
        except Exception as e:
            raise ExceptionNetwork(e, sys)

    def translate_text(self, turkish_text_path: Path):
        try:
            
            marian_model.model_load_and_predict(load_model_path=self.config.translator_model_path,
                                                load_processor_path=self.config.translator_processor_path,
                                                load_turkish_text_path=turkish_text_path,
                                                translated_text_save_path=self.config.translated_text_path)
        except Exception as e:
            raise ExceptionNetwork(e, sys)

    def text_to_speech(self, english_text_path: Path):
        try:
            
            
            mms_tts_prediction.model_load_and_predict(load_model_path=self.config.text_to_speech_model_path,
                                                    load_processor_path=self.config.text_to_speech_processor_path,
                                                    load_english_text_path=english_text_path,
                                                    translated_audio_save_path=self.config.translated_audio_save_path)
        except Exception as e:
            raise ExceptionNetwork(e, sys)
