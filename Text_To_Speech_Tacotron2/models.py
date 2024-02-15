import torchaudio

def Create_Bundles(devices):
    print("Creating Models...")
    
    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
    processor = bundle.get_text_processor()
    tacotron2 = bundle.get_tacotron2().to(devices)
    vocoder = bundle.get_vocoder().to(devices)
    
    return processor,tacotron2,vocoder