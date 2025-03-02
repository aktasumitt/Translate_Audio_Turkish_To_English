
# Turkish to English Translation App

This project is an AI-based translation app that converts Turkish speech to English text and then back to speech. The app is built using Flask and utilizes three models from Hugging Face: Whisper for speech recognition, Opus-MT-TC-Big-Fi-En for translation, and Facebook MMS-TTS for text-to-speech conversion.

## Features

- **Speech Recognition**: Converts Turkish speech to text using the Whisper model.
- **Translation**: Translates Turkish text to English using the Opus-MT-TC-Big-Fi-En model.
- **Text-to-Speech**: Converts English text back to speech using the Facebook MMS-TTS model.
- **Multiple Options**: You can run each step separately or combine them to perform speech-to-speech translation.

## Requirements

Before running the app, make sure you have the necessary dependencies installed:

1. Clone this repository:
    ```cmd
    git clone https://github.com/aktasumitt/Translate_Audio_Turkish_To_English.git
    cd Translate_Audio_Turkish_To_English
    ```

2. Install dependencies:
    ```cmd
    pip install -r requirements.txt
    ```

3. Download models for just first time:
    ```cmd
    python -m src.pipelines.model_ingestion
    ```

## Running the App

To run the app on your local ("localhost:5000"), use the following command:

```cmd
python app.py
```

This will start the Flask application, and you can access it locally on your browser.

## Models Used

- **Whisper**: For speech recognition (Turkish to text).
- **Opus-MT-TC-Big-Fi-En**: For translating the Turkish text to English.
- **Facebook MMS-TTS**: For converting the English text back into speech.

## How to Use

The app provides the flexibility to run each process individually or together. You can:

1. Recognize speech (Turkish to text).
2. Translate the text (Turkish to English).
3. Convert text to speech (English to speech).

Simply follow the on-screen instructions to choose your preferred operation.
