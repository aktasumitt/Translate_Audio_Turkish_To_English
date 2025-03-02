from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from src.utils import save_as_json
from src.exception.exception import ExceptionNetwork, sys
from src.logging.logger import logger
import torchaudio
from pathlib import Path

def model_load_and_predict(load_turkish_audio_path: Path, load_model_path, load_processor_path, transcription_save_path):
    try:
        logger.info(f"Loading speech recognition model from {load_model_path} and processor from {load_processor_path}...")
        
        processor = AutoProcessor.from_pretrained(load_processor_path) 
        model = AutoModelForSpeechSeq2Seq.from_pretrained(load_model_path) 

        waveform, sample_rate = torchaudio.load(load_turkish_audio_path)

        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        inputs = processor(
            waveform.numpy(),
            sampling_rate=target_sample_rate,
            return_tensors="pt"
        )

        forced_decoder_ids = processor.get_decoder_prompt_ids(language="turkish", task="transcribe")

        input_features = inputs.input_features

        generated_ids = model.generate(
            input_features, 
            forced_decoder_ids=forced_decoder_ids,
            suppress_tokens=None 
        )

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        save_as_json(transcription, save_path=transcription_save_path)

        logger.info(f"Transcription saved successfully at {transcription_save_path}.")
        return transcription

    except Exception as e:
        raise ExceptionNetwork(e, sys)
