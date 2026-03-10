from flask import Flask, request, jsonify, render_template
import whisper
from transformers import pipeline
import os

app = Flask(__name__)

# 1. UPGRADE TO THE "SMALL" MODEL (Better for Hindi and other languages)
print("Loading Whisper model (Small)... This will take a moment.")
whisper_model = whisper.load_model("small")

# 2. UPGRADE TO A MULTILINGUAL SENTIMENT MODEL
print("Loading Multilingual Sentiment Pipeline...")
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    temp_path = "temp_audio.webm"
    audio_file.save(temp_path)

    try:
        # Transcribe the audio (fp16=False prevents warnings on Windows CPUs)
        result = whisper_model.transcribe(temp_path, fp16=False)
        text = result["text"]

        # If the transcription is empty, handle it
        if not text.strip():
             return jsonify({'error': 'No speech detected.'}), 400

        # Analyze Sentiment with the new Multilingual Pipeline
        sentiment_result = sentiment_pipeline(text)[0]
        
        # We use .upper() because the new model outputs lowercase "positive"/"negative" 
        # but your HTML UI is looking for uppercase words!
        label = sentiment_result['label'].upper()
        
        return jsonify({
            'text': text,
            'sentiment': label,
            'confidence': round(sentiment_result['score'] * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)