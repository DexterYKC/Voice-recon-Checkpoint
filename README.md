# Nutrition Chatbot Vocal — Checkpoint

Petit chatbot FAQ + reconnaissance vocale:
- Texte: question → réponse par similarité (Jaccard sur tokens lemmatisés).
- Voix: 
  - Enregistrement navigateur (st_audiorec) → transcription Google Speech (SpeechRecognition)
  - Upload audio (wav/mp3/m4a) → conversion via FFmpeg (pydub) → transcription

## Démo locale
```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
streamlit run app.py
