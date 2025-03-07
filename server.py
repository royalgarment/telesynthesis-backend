from flask import Flask, Response, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from google.cloud import speech
from openai import OpenAI
import os
import sounddevice as sd
import wavio
import queue
import json
import time
import numpy as np
import threading
import logging
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://policy_user:secure_password@localhost/telesynthesis_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Google Speech-to-Text setup
logger.info("Initializing Google Speech-to-Text client...")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\ryanw\git\telesynthesis-backend\astute-loop-177006-9165f80d7c12.json"
try:
    speech_client = speech.SpeechClient()
    logger.info("Google Speech-to-Text client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Google Speech-to-Text client: {e}")
    raise

logger.info("Initializing OpenAI client...")
try:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

conversation_history = {}
event_queues = {}
AUDIO_FILE_PATH = "temp_audio.wav"
AUDIO_OUTPUT_DIR = "audio_logs"
DURATION = 5
SAMPLE_RATE = 44100
recording_thread = None
stop_recording_flag = threading.Event()

if not os.path.exists(AUDIO_OUTPUT_DIR):
    os.makedirs(AUDIO_OUTPUT_DIR)

class Policy(db.Model):
    __tablename__ = 'policies'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(100))
    text = db.Column(db.Text, nullable=False)
    keywords = db.Column(db.String(255))
    color_code = db.Column(db.String(7))
    requires_disclaimer = db.Column(db.Boolean, default=False)

class Disclaimer(db.Model):
    __tablename__ = 'disclaimers'
    id = db.Column(db.Integer, primary_key=True)
    policy_id = db.Column(db.Integer, db.ForeignKey('policies.id'))
    text = db.Column(db.Text, nullable=False)

def event_stream(session_id):
    q = queue.Queue()
    event_queues[session_id] = q
    try:
        while True:
            data = q.get()
            yield f"event: {data['event']}\ndata: {json.dumps(data['data'])}\n\n"
    except GeneratorExit:
        logger.info(f"Client {session_id} disconnected from SSE stream")
        del event_queues[session_id]

@app.route("/events")
def sse_stream():
    session_id = request.args.get("session_id", request.remote_addr + str(time.time()))
    logger.info(f"Client connected to SSE with session ID: {session_id}")
    conversation_history[session_id] = []
    return Response(event_stream(session_id), mimetype="text/event-stream")

@app.route("/start_recording", methods=["POST"])
def start_recording():
    global recording_thread
    session_id = request.json.get("session_id")
    if not session_id or session_id not in event_queues:
        return {"error": "Invalid or missing session_id"}, 400
    if recording_thread is None or not recording_thread.is_alive():
        logger.info(f"Starting recording loop for {session_id}...")
        stop_recording_flag.clear()
        recording_thread = threading.Thread(target=record_audio_loop, args=(session_id,))
        recording_thread.start()
        event_queues[session_id].put({"event": "recording_started", "data": {"status": "Recording started"}})
        return {"status": "Recording started"}, 200
    else:
        logger.info(f"Recording already in progress for {session_id}")
        return {"status": "Recording already in progress"}, 200

@app.route("/stop_recording", methods=["POST"])
def stop_recording():
    global recording_thread
    session_id = request.json.get("session_id")
    if not session_id or session_id not in event_queues:
        return {"error": "Invalid or missing session_id"}, 400
    logger.info(f"Stopping recording for {session_id}...")
    stop_recording_flag.set()
    if recording_thread is not None and threading.current_thread() != recording_thread:
        recording_thread.join(timeout=3)
        if recording_thread.is_alive():
            logger.warning(f"Recording thread for {session_id} did not stop cleanly")
        recording_thread = None
    event_queues[session_id].put({"event": "recording_stopped", "data": {"status": "Recording stopped"}})
    return {"status": "Recording stopped"}, 200

def trim_silence(audio, threshold=0.02):
    audio_rms = np.sqrt(np.mean(audio**2))
    if audio_rms < threshold:
        return None
    energy = np.abs(audio)
    mask = energy > (threshold * np.max(energy))
    return audio[np.where(mask)[0][0]:np.where(mask)[0][-1] + 1]

def record_audio_loop(session_id):
    logger.info(f"🎤 Starting 5s recording loop for {session_id}...")
    while not stop_recording_flag.is_set():
        try:
            if os.path.exists(AUDIO_FILE_PATH):
                os.remove(AUDIO_FILE_PATH)
            audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
            sd.wait()
            audio_rms = np.sqrt(np.mean(audio**2))
            logger.info(f"🎵 Audio RMS for {session_id}: {audio_rms:.6f}")
            trimmed_audio = trim_silence(audio)
            if trimmed_audio is None:
                logger.info(f"🔇 Silent chunk detected for {session_id}, skipping...")
                continue
            wavio.write(AUDIO_FILE_PATH, trimmed_audio, SAMPLE_RATE, sampwidth=2)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            saved_audio_path = os.path.join(AUDIO_OUTPUT_DIR, f"audio_{session_id}_{timestamp}.wav")
            shutil.copy(AUDIO_FILE_PATH, saved_audio_path)
            logger.info(f"✅ Recorded and saved 5s chunk for {session_id} to {saved_audio_path}")
            with app.app_context():
                if session_id in conversation_history:
                    process_audio(session_id)
                else:
                    logger.info(f"Session {session_id} disconnected, stopping recording")
                    break
        except Exception as e:
            logger.error(f"Recording loop error for {session_id}: {str(e)}")
            event_queues[session_id].put({"event": "error", "data": {"error": str(e)}})
            break
    logger.info(f"🛑 Recording loop stopped for {session_id}")
    if os.path.exists(AUDIO_FILE_PATH):
        os.remove(AUDIO_FILE_PATH)

def process_audio(session_id):
    try:
        logger.info(f"Transcribing audio for {session_id} with Google Speech-to-Text...")
        start_time = time.time()
        with open(AUDIO_FILE_PATH, "rb") as audio_file:
            content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="en-US",
        )
        response = speech_client.recognize(config=config, audio=audio)
        transcript = " ".join(result.alternatives[0].transcript for result in response.results if result.alternatives).strip()
        logger.info(f"Transcription for {session_id}: {transcript} (took {time.time() - start_time:.2f}s)")
        
        if not transcript or transcript.lower() in ["thank you", "thanks"]:
            last_transcript = next((msg["content"] for msg in reversed(conversation_history[session_id]) if "content" in msg), "")
            if transcript.lower() == last_transcript.lower():
                logger.info(f"Skipping duplicate transcript for {session_id}: {transcript}")
                return
        
        event_queues[session_id].put({"event": "transcript_update", "data": {"transcript": transcript}})
        conversation_history[session_id].append({"role": "user", "content": transcript})
        policy, clarification_question = detect_policy_intent(transcript, session_id)

        if policy:
            last_policy = next((msg.get("policy") for msg in reversed(conversation_history[session_id][:-1]) if "policy" in msg), None)
            if any("disclaimer" in msg["content"].lower() and ("yes" in msg["content"].lower() or "agree" in msg["content"].lower()) 
                   for msg in conversation_history[session_id][-3:]):
                logger.info(f"Disclaimer agreement detected for {session_id}")
                if last_policy and last_policy == policy.name:
                    conversation_history[session_id].append({"role": "system", "content": f"Disclaimer for {policy.name} agreed"})
            elif clarification_question:
                logger.info(f"Emitting clarification question for {session_id}: {clarification_question}")
                event_queues[session_id].put({"event": "clarification_needed", "data": {"question": clarification_question}})
                return

            logger.info(f"Detected Policy for {session_id}: {policy.name}")
            disclaimer = Disclaimer.query.filter_by(policy_id=policy.id).first().text if policy.requires_disclaimer else None
            policy_data = {
                "id": policy.id,
                "name": policy.name,
                "color_code": policy.color_code,
                "requires_disclaimer": policy.requires_disclaimer,
                "description": policy.text,
                "disclaimer": disclaimer
            }
            event_queues[session_id].put({"event": "policy_update", "data": {"policy": policy_data}})
            conversation_history[session_id].append({"role": "system", "content": f"Policy {policy.name} selected", "policy": policy.name})

            ai_response = analyze_with_ai(transcript, policy, session_id)
            event_queues[session_id].put({"event": "ai_response", "data": ai_response})
        else:
            logger.info(f"No policy detected for {session_id}")
            if clarification_question:
                logger.info(f"Emitting clarification question for {session_id}: {clarification_question}")
                event_queues[session_id].put({"event": "clarification_needed", "data": {"question": clarification_question}})
            else:
                event_queues[session_id].put({"event": "policy_update", "data": {"policy": None}})

    except Exception as e:
        logger.error(f"Processing error for {session_id}: {str(e)}")
        event_queues[session_id].put({"event": "error", "data": {"error": str(e)}})

def detect_policy_intent(transcript, session_id):
    logger.info(f"Detecting policy intent for {session_id} with transcript: '{transcript}'")
    policies = Policy.query.all()
    logger.info(f"Found {len(policies)} policies in database")
    
    policy_options = "\n".join(
        [f"Policy: {p.name}, Keywords: {p.keywords}, Summary: {p.text[:100]}..." for p in policies]
    )

    system_message = (
        "You are an AI that selects the most relevant policy based on conversation context. "
        "Focus EXCLUSIVELY on the latest user input to determine the best matching policy, "
        "unless it is ambiguous or lacks sufficient detail (e.g., 'insurance' alone). "
        "Only use the conversation history as secondary context if the latest input is unclear. "
        "Ignore prior selected policies unless reaffirmed in the latest input. "
        "Look for keywords like 'loan', 'car', 'vehicle' to match policies like 'Personal Loan Agreement'. "
        "If intent is clear (e.g., 'I want a loan'), return the policy name. "
        "If ambiguous, return 'None' and suggest a clarifying question. "
        "Respond with a JSON object: {\"policy_name\": \"[policy name or None]\", \"clarification\": \"[question or null]\"}"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": (
            f"Conversation History (use only if latest input is ambiguous):\n{json.dumps(conversation_history[session_id], indent=2)}\n\n"
            f"Latest User Input (prioritize this):\n{transcript}\n\n"
            f"Policy Options:\n{policy_options}\n\n"
            "Analyze the user's intent and respond with a JSON object as described."
        )}
    ]

    try:
        start_time = time.time()
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=100,
            temperature=0.2
        )
        ai_output = response.choices[0].message.content.strip()
        logger.info(f"OpenAI Intent Output for {session_id}: {ai_output} (took {time.time() - start_time:.2f}s)")
        result = json.loads(ai_output)
        policy_name = result["policy_name"]
        clarification = result.get("clarification", None)

        if clarification:
            return None, clarification
        return next((p for p in policies if p.name == policy_name), None) if policy_name != "None" else None, None
    
    except Exception as e:
        logger.error(f"Intent detection error for {session_id}: {str(e)}")
        return None, None

def analyze_with_ai(transcript, policy=None, session_id=None):
    try:
        system_message = (
            "You are an AI assistant ensuring policy compliance. "
            "Your response must strictly use the provided policy text from the database as the basis for your answer. "
            "Do not rely on general knowledge outside the policy text unless explicitly stated. "
            "Return a JSON object with 'policy_name' (exact name of the policy or 'general' if none), "
            "'response' (answer based solely on the policy text), "
            "'disclaimer_said' (true if telemarketer likely said the disclaimer, false otherwise), "
            "'disclaimer_agreed' (true if user agreed to the disclaimer, false otherwise). "
            "Do not include a 'disclaimer' field in the response."
        )
        policy_text = policy.text if policy else "No specific policy applies."
        disclaimer = Disclaimer.query.filter_by(policy_id=policy.id).first().text if policy and policy.requires_disclaimer else None
        
        disclaimer_said = False
        disclaimer_agreed = False
        if disclaimer:
            disclaimer_words = set(disclaimer.lower().split())
            key_disclaimer_terms = {"credit", "rates", "payments", "fee", "advisor"}
            last_messages = [msg["content"].lower() for msg in conversation_history[session_id][-3:]]
            logger.info(f"Checking disclaimer '{disclaimer}' against last messages: {last_messages}")
            
            for msg in last_messages:
                msg_words = set(msg.split())
                overlap = len(disclaimer_words & msg_words) / len(disclaimer_words)
                logger.info(f"Overlap for '{msg}': {overlap}")
                key_terms_matched = len(key_disclaimer_terms & msg_words)
                logger.info(f"Key terms matched in '{msg}': {key_terms_matched}")
                
                if overlap >= 0.3 or key_terms_matched >= 2:
                    disclaimer_said = True
                    conversation_history[session_id].append({"role": "system", "content": f"Disclaimer for {policy.name} likely said"})
                    break
        
            disclaimer_agreed = any("i agree" in msg or "yes" in msg for msg in last_messages if "disclaimer" in msg)
            if disclaimer_agreed:
                conversation_history[session_id].append({"role": "system", "content": f"Disclaimer for {policy.name} agreed"})
            logger.info(f"Disclaimer said: {disclaimer_said}, agreed: {disclaimer_agreed}")

        history_summary = json.dumps(conversation_history[session_id], indent=2)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": (
                f"Conversation History:\n{history_summary}\n\n"
                f"Policy Text: {str(policy_text)}\n"
                f"User Input: {str(transcript)}\n"
                "Using ONLY the Policy Text above (no external info), provide a JSON response with:\n"
                "{\n"
                "  \"policy_name\": \"[exact policy name or 'general']\",\n"
                "  \"response\": \"[direct quote or paraphrase from policy text]\",\n"
                "  \"disclaimer_said\": [true or false],\n"
                "  \"disclaimer_agreed\": [true or false]\n"
                "}"
            )}
        ]

        start_time = time.time()
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
            temperature=0.2
        )
        ai_output = response.choices[0].message.content.strip()
        logger.info(f"Raw OpenAI Output for {session_id}: {ai_output} (took {time.time() - start_time:.2f}s)")
        try:
            json_response = json.loads(ai_output)
            json_response["disclaimer_said"] = disclaimer_said
            json_response["disclaimer_agreed"] = disclaimer_agreed
        except json.JSONDecodeError:
            logger.warning(f"OpenAI did not return valid JSON for {session_id}, using fallback")
            json_response = {
                "policy_name": policy.name if policy else "general",
                "response": f"Based on the policy: {policy_text}",
                "disclaimer_said": disclaimer_said,
                "disclaimer_agreed": disclaimer_agreed
            }

        logger.info(f"Parsed AI Response for {session_id}: {json_response}")
        return json_response

    except Exception as e:
        logger.error(f"AI Analysis error for {session_id}: {str(e)}")
        return {
            "policy_name": "error",
            "response": f"Error analyzing: {str(e)}",
            "disclaimer_said": False,
            "disclaimer_agreed": False
        }

if __name__ == "__main__":
    logger.info("Initializing database...")
    try:
        with app.app_context():
            db.create_all()
            num_policies = db.session.query(Policy).count()
            logger.info(f"Database connected: {db.engine.url}")
            logger.info(f"Number of policies in database: {num_policies}")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

    logger.info("Starting Flask server...")
    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    except Exception as e:
        logger.error(f"Failed to start Flask server: {e}")
        raise