import io
import os, tempfile
import speech_recognition as sr
import re
import string
import json
from pathlib import Path
from st_audiorec import st_audiorec


import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import which

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe   = which("ffprobe")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

FAQ_PATH = Path("data/faq.txt")
assert FAQ_PATH.exists(), "data/faq.txt introuvable — crée le fichier FAQ avant de lancer."

lemmatizer = WordNetLemmatizer()
STOP = set(stopwords.words("english")) | set(stopwords.words("french"))
PUNCT = set(string.punctuation)

def normalize_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def preprocess(sentence: str):
    tokens = word_tokenize(normalize_text(sentence))
    tokens = [t for t in tokens if t not in PUNCT and t not in STOP]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
def load_faq(path: Path):
    """
    Format attendu:
    Q: question...
    R: reponse...
    (ligne vide pour séparer les paires)
    """
    raw = path.read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    pairs = []
    for b in blocks:
        q = re.search(r"(?im)^q:\s*(.+)$", b)
        r = re.search(r"(?im)^r:\s*(.+)$", b)
        if q and r:
            q_txt = q.group(1).strip()
            r_txt = r.group(1).strip()
            pairs.append((q_txt, r_txt))
    return pairs

FAQ = load_faq(FAQ_PATH)
assert len(FAQ) > 0, "FAQ vide. Ajoute au moins text dans data/faq.txt"

FAQ_INDEX = []
for q, r in FAQ:
    FAQ_INDEX.append({
        "q_raw": q,
        "q_tokens": set(preprocess(q)),
        "r": r
    })

def chatbot_reply(user_text: str) -> str:
    tokens = set(preprocess(user_text))
    best_sim, best_ans = 0.0, None
    for item in FAQ_INDEX:
        sim = jaccard(tokens, item["q_tokens"])
        if sim > best_sim:
            best_sim = sim
            best_ans = item["r"]
    if best_ans is None or best_sim < 0.1:
        return "Je n'ai pas compris, Reformue"
    return best_ans

# ---------- Speech-to-Text ----------
def speech_to_text_from_mic(timeout=5, phrase_time_limit=10, language="fr-FR"):
    """
    Requiert un micro en local (PyAudio installé).
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Parle maintenant… (silence pour arrêter)")
        r.pause_threshold = 0.8
        r.energy_threshold = 200
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        text = r.recognize_google(audio, language=language)
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        st.error(f"Erreur API STT: {e}")
        return None

def speech_to_text_from_file(file_or_uploaded, language="fr-FR"):
    import io, tempfile
    r = sr.Recognizer()

    # recup bytes + nom 
    if hasattr(file_or_uploaded, "read"):
        data = file_or_uploaded.read()
        name = getattr(file_or_uploaded, "name", "uploaded.bin").lower()
    elif isinstance(file_or_uploaded, (bytes, bytearray)):
        data = bytes(file_or_uploaded)
        name = "uploaded.bin"
    else:
        return None, "Type non support"

    if name.endswith(".wav"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(data)
                wav_path = tmp.name
            with sr.AudioFile(wav_path) as source:
                audio = r.record(source)
            text = r.recognize_google(audio, language=language)
            return text, None
        except sr.UnknownValueError:
            return None, "Audio non compris"
        except Exception as e:
            return None, f"Erreur transcription WAV: {e}"
        finally:
            try: os.remove(wav_path)
            except: pass

    if AudioSegment.converter is None:
        return None, "FFmpeg introuvable. Installe-le puis relance (voir instruc)."

    try:
        audio_seg = AudioSegment.from_file(io.BytesIO(data))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_seg.set_frame_rate(16000).set_channels(1).export(tmp.name, format="wav")
            wav_path = tmp.name
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language=language)
        return text, None
    except sr.UnknownValueError:
        return None, "Audio incomp"
    except FileNotFoundError:
        return None, "FFmpeg manquant. Installe FFmpeg puis relance."
    except Exception as e:
        return None, f"Erreur transcription (ffmpeg): {e}"
    finally:
        try: os.remove(wav_path)
        except: pass

# ---------- UI ----------
st.set_page_config(page_title="Nutr Chatbot Vocal", layout="centered")
st.title("Nutr Chatbot Vocal — Checkpoint")

with st.expander("Instruc", expanded=True):
    st.markdown(
        """
        **Deux entree :**
        1) **Texte** : ecris ta question clique *Envoyer*  
        2) **Voix** : 
           - **Micro (web)** bouton *Enregistrer au micro*  
           - **Upload audio** (wav/mp3/m4a) bouton *Transcrire le fichier*  
        
        **Exemples de questions :**
        - *"combien de calories pour une prise de masse propre ?"*
        - *"quelle est la meilleure source de proteines ?"*
        - *"je m entraine combien de fois par semaine ?"*
        """
    )

tab_text, tab_voice = st.tabs(["°°° Texte", "Voix"])

if "history" not in st.session_state:
    st.session_state["history"] = []   # liste de role

def push(role, text):
    st.session_state["history"].append((role, text))

with tab_text:
    user_msg = st.text_input("Ta question :", placeholder="Ex: combien de calories pour une prise de masse ?")
    if st.button("Envoyer", type="primary"):
        if not user_msg.strip():
            st.warning("Écris quelque chose d abord.")
        else:
            push("user", user_msg)
            ans = chatbot_reply(user_msg)
            push("bot", ans)

with tab_voice:
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Enregistrer depuis le micro (navigateur)")
        wav_bytes = st_audiorec()
        if wav_bytes and st.button("Transcrire l enregistrement"):
            text, err = speech_to_text_from_file(wav_bytes)
            if err:
                st.error(err)
            else:
                st.success(f"Transcription : {text}")
                push("user", text)
                push("bot", chatbot_reply(text))
    with col2:
        audio_file = st.file_uploader("ou uploader un fichier audio (wav/mp3/m4a)", type=["wav","mp3","m4a"])
        if st.button("Transcrire le fichier"):
            if audio_file is None:
                st.warning("Ajoute un fichier audio d abord.")
            else:
                text, err = speech_to_text_from_file(audio_file) 
                if err:
                    st.error(err)
                else:
                    st.success(f"Transcription : {text}")
                    push("user", text)
                    push("bot", chatbot_reply(text))

                if text:
                    st.success(f"Transcription : {text}")
                    push("user", text)
                    ans = chatbot_reply(text)
                    push("bot", ans)
                else:
                    st.warning("Impossible de transcrire ce fichier.")

st.markdown("---")
st.subheader("Historique")
for role, msg in st.session_state["history"]:
    if role == "user":
        st.markdown(f"** Moi :** {msg}")
    else:
        st.markdown(f"** Bot :** {msg}")
st.caption("Formats acceptés: **WAV** (PCM). Convertis-le en WAV.")
