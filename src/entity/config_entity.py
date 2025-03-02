from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelIngestionEntity:
    speech_recognation_model_path:Path
    speech_recognation_processor_path:Path
    translator_model_path:Path
    translator_processor_path:Path
    text_to_speech_model_path:Path
    text_to_speech_processor_path:Path
    

@dataclass
class PredictionEntity:
    speech_recognation_model_path:Path
    speech_recognation_processor_path:Path
    translator_model_path:Path
    translator_processor_path:Path
    text_to_speech_model_path:Path
    text_to_speech_processor_path:Path
    
    recognated_text_save_path: Path
    translated_text_path: Path
    translated_audio_save_path: Path
    
    main_turkish_audio_path: Path

    
    