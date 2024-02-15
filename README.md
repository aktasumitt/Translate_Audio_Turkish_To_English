# Translate Audio from Turkish to English 

## Introduction:

- In this project, I aimed to translate Turkish Audio to English Audio.
- In this project, I used separate models for Turkish Speech to Text, Text Translation to English, and English Text to Speech, and integrated them all into one place. When run sequentially, we achieved audio-to-audio translation.
-  For Speech to Text, I utilized the OpenAI Whisper model, the Bart Model for translation, and the Taotron model for Text to Speech.

#### For details :
 - Whisper : https://openai.com/research/whisper
 - Bark: https://huggingface.co/suno/bark
 - Tacotron2: https://pytorch.org/audio/stable/tutorials/tacotron2_pipeline_tutorial.html

## Dataset:
- I didnt train the models. I just use for prediction so we have just audio for predict. 

## Train:
- firstly, we use pretrained Whisper Model for Turkish speech to text
- Secondly, we use pretrained Bart Model for translating from Turkish text that is output of Whisper to English 
- After that , we use pretrained Tacotron Model with the output of Bart for transforming from text to speech.
- Before using the Tacotron model, I segmented the text into chunks separated by punctuation marks and trained them separately, then combined them at the end. I observed that this method yielded better results with Tacotron. It struggled with converting long texts into speech and even became muddled towards the end of the audio.
- I used Custom audio to transform

## Usage: 
- You can use directly Audio to Audio with Concatinated_Models folder. You may need to set paths in config file acording to your Data paths 
- If you want to use the models separately, you can go to their respective folders and use only what you need from there
- Predictions of Models will save "Prediction" folder 







