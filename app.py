from flask import Flask, request, render_template, send_file
import os
from pathlib import Path
from src.pipelines.prediction_pipeline.speech_recognation_pipeline import SpeechRecognationPredictionPipeline
from src.pipelines.prediction_pipeline.translate_pipeline import TranslatePredictionPipeline
from src.pipelines.prediction_pipeline.text_to_speech_pipeline import TextToSpeechPipeline
from src.configuration.configuration import Configurations
from src.utils import load_json, save_as_json
from src.exception.exception import ExceptionNetwork,sys

app = Flask(__name__)
config = Configurations().prediction_config()

UPLOAD_FOLDER = Path("results/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# 1. Speech Recognition
@app.route("/speech_recognition", methods=["POST"])
def speech_recognition():
    try:
        audio_file = request.files.get("audio")
        if not audio_file:
            return render_template("index.html", recognized_text="Hata: Ses dosyası gönderilmedi.")

        audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(audio_path)

        speech_recognition_pipeline = SpeechRecognationPredictionPipeline()
        speech_recognition_pipeline.run_speech_recognation(audio_path)
        result = load_json(config.recognated_text_save_path)

        return render_template("index.html", recognized_text=result)

    except Exception as e:
        raise ExceptionNetwork(e,sys)

# 2. Translate
@app.route("/translate", methods=["POST"])
def translate():
    try:
        manual_text = request.form.get("manual_text")
        source_json = request.files.get("source_json")

        if source_json:
            data = load_json(source_json)
            source_text = data.get("text", "")
        elif manual_text:
            source_text = manual_text.strip()
        else:
            return render_template("index.html", translated_text="Hata: Metin girişi veya JSON dosyası gerekli.")

        text_path = os.path.join(UPLOAD_FOLDER, "turkish_text.json")
        save_as_json(source_text, text_path)

        translator = TranslatePredictionPipeline()
        translator.run_translator(text_path)

        translated_text = load_json(config.translated_text_path)
        return render_template("index.html", translated_text=translated_text)

    except Exception as e:
        raise ExceptionNetwork(e,sys)

# 3. Text-to-Speech
@app.route("/text_to_speech", methods=["POST"])
def text_to_speech():
    try:
        manual_text = request.form.get("manual_text")
        source_json = request.files.get("source_json")

        if source_json:
            data = load_json(source_json)
            source_text = data.get("text", "")
        elif manual_text:
            source_text = manual_text.strip()
        else:
            return render_template("index.html", audio_path=None)

        text_path_en = os.path.join(UPLOAD_FOLDER, "english_text.json")
        save_as_json(source_text, text_path_en)

        text2speech = TextToSpeechPipeline()
        text2speech.run_text_to_speech(text_path_en)

        audio_path = config.translated_audio_save_path

        if not os.path.exists(audio_path):
            return render_template("index.html", speech_audio_path=None)

        return send_file(audio_path, mimetype="audio/wav", as_attachment=True)

    except Exception as e:
        raise ExceptionNetwork(e,sys)

# 4. Speech-to-Speech
@app.route("/speech_to_speech", methods=["POST"])
def speech_to_speech():
    try:
        audio_file = request.files.get("audio")
        if not audio_file:
            return render_template("index.html", speech_audio_path=None)

        audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(audio_path)

        speech_recognition_pipeline = SpeechRecognationPredictionPipeline()
        speech_recognition_pipeline.run_speech_recognation(audio_path)

        translator = TranslatePredictionPipeline()
        translator.run_translator(config.recognated_text_save_path)

        text2speech = TextToSpeechPipeline()
        text2speech.run_text_to_speech(config.translated_text_path)

        audio_path = config.translated_audio_save_path

        if not os.path.exists(audio_path):
            return render_template("index.html", speech_audio_path=None)

        return send_file(audio_path, mimetype="audio/wav", as_attachment=True)

    except Exception as e:
        raise ExceptionNetwork(e,sys)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
