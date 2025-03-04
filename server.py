from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
import openai
import eventlet
import os
import io
import time
import subprocess
import json
from pydub import AudioSegment
from collections import deque

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://policy_user:secure_password@localhost/telesynthesis_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Buffers and state
audio_buffers = {}
conversation_history = {}
header_chunks = {}
TEMP_WEBM_FILE = "temp_audio.webm"
TEMP_WAV_FILE = "temp_audio.wav"
CHUNKS_TO_PROCESS = 5
DEBUG_WEBM_FILE = "debug_audio.webm"
FAILED_PROCESS_COUNT = 0
MAX_FAILED_PROCESSES = 5

# Database Models
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

@socketio.on("connect")
def handle_connect():
    session_id = request.sid
    print(f"Client connected with session ID: {session_id}")
    audio_buffers[session_id] = deque()
    conversation_history[session_id] = []
    header_chunks[session_id] = None
    global FAILED_PROCESS_COUNT
    FAILED_PROCESS_COUNT = 0
    with open(TEMP_WEBM_FILE, "wb") as f:
        f.write(b"")

@socketio.on("disconnect")
def handle_disconnect():
    session_id = request.sid
    print(f"Client {session_id} disconnected")
    if session_id in audio_buffers:
        del audio_buffers[session_id]
    if session_id in conversation_history:
        del conversation_history[session_id]
    if session_id in header_chunks:
        del header_chunks[session_id]
    for temp_file in [TEMP_WEBM_FILE, TEMP_WAV_FILE, DEBUG_WEBM_FILE]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    session_id = request.sid
    print(f"Received audio chunk from {session_id} of size {len(data)} bytes")
    if header_chunks[session_id] is None:
        header_chunks[session_id] = data
    audio_buffers[session_id].append(data)
    print(f"Buffer size for {session_id}: {len(audio_buffers[session_id])} chunks")
    if len(audio_buffers[session_id]) >= CHUNKS_TO_PROCESS:
        process_audio_buffer(session_id)

def process_audio_buffer(session_id):
    global FAILED_PROCESS_COUNT
    if session_id not in audio_buffers or not audio_buffers[session_id] or header_chunks[session_id] is None:
        print(f"Buffer empty or no header for {session_id}, skipping processing")
        return

    print(f"Processing {len(audio_buffers[session_id])} chunks for {session_id}...")
    with open(TEMP_WEBM_FILE, "wb") as f:
        print(f"Writing header chunk of size {len(header_chunks[session_id])} bytes for {session_id}")
        f.write(header_chunks[session_id])
        subsequent_chunks = list(audio_buffers[session_id])[1:]
        total_subsequent_size = sum(len(chunk) for chunk in subsequent_chunks)
        print(f"Writing {len(subsequent_chunks)} subsequent chunks, total size: {total_subsequent_size} bytes")
        f.write(b"".join(subsequent_chunks))
    with open(DEBUG_WEBM_FILE, "wb") as f:
        f.write(header_chunks[session_id])
        f.write(b"".join(list(audio_buffers[session_id])[1:]))

    try:
        print("Running FFmpeg conversion...")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", TEMP_WEBM_FILE, "-f", "wav", TEMP_WAV_FILE],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        print(f"FFmpeg stdout: {result.stdout.decode()}")
        if result.stderr:
            print(f"FFmpeg stderr: {result.stderr.decode()}")

        print("Loading WAV file...")
        audio_segment = AudioSegment.from_wav(TEMP_WAV_FILE)
        print(f"Audio duration: {len(audio_segment) / 1000:.1f}s")

        transcribe_and_analyze(audio_segment, session_id)
        audio_buffers[session_id].clear()
        FAILED_PROCESS_COUNT = 0

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        FAILED_PROCESS_COUNT += 1
        print(f"Failed process count: {FAILED_PROCESS_COUNT}")
        if FAILED_PROCESS_COUNT >= MAX_FAILED_PROCESSES:
            print(f"Too many FFmpeg failures for {session_id}, resetting header_chunk")
            header_chunks[session_id] = None
        audio_buffers[session_id].clear()
    except Exception as e:
        print(f"Processing error for {session_id}: {str(e)}")
        audio_buffers[session_id].clear()

def transcribe_and_analyze(audio_segment, session_id):
    try:
        print(f"Exporting audio to buffer for {session_id}...")
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.name = "audio.wav"
        wav_buffer.seek(0)

        print(f"Sending to Whisper API for {session_id}...")
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=wav_buffer,
            language="en"
        ).text

        print(f"Transcription for {session_id}: {transcript} (Type: {type(transcript)})")
        socketio.emit("transcript_update", {"transcript": transcript}, room=session_id)

        # Store in conversation history
        conversation_history[session_id].append({"role": "user", "content": transcript})

        # Detect policy intent first
        policy, clarification_question = detect_policy_intent(transcript, session_id)

        # Check for disclaimer agreement only after intent detection
        if policy:
            last_policy = next((msg.get("policy") for msg in reversed(conversation_history[session_id][:-1]) if "policy" in msg), None)
            if any("disclaimer" in msg["content"].lower() and ("yes" in msg["content"].lower() or "agree" in msg["content"].lower()) 
                   for msg in conversation_history[session_id][-3:]):
                print(f"Disclaimer agreement detected for {session_id}")
                if last_policy and last_policy == policy.name:
                    conversation_history[session_id].append({"role": "system", "content": f"Disclaimer for {policy.name} agreed"})
            elif clarification_question:
                print(f"Emitting clarification question for {session_id}: {clarification_question}")
                socketio.emit("clarification_needed", {"question": clarification_question}, room=session_id)
                return

            print(f"Detected Policy for {session_id}: {policy.name}")
            disclaimer = Disclaimer.query.filter_by(policy_id=policy.id).first().text if policy.requires_disclaimer else None
            policy_data = {
                "id": policy.id,
                "name": policy.name,
                "color_code": policy.color_code,
                "requires_disclaimer": policy.requires_disclaimer,
                "description": policy.text,
                "disclaimer": disclaimer
            }
            socketio.emit("policy_update", {"policy": policy_data}, room=session_id)
            conversation_history[session_id].append({"role": "system", "content": f"Policy {policy.name} selected", "policy": policy.name})

            ai_response = analyze_with_ai(transcript, policy, session_id)
            socketio.emit("ai_response", ai_response, room=session_id)
        else:
            print(f"No policy detected for {session_id}")
            if clarification_question:
                print(f"Emitting clarification question for {session_id}: {clarification_question}")
                socketio.emit("clarification_needed", {"question": clarification_question}, room=session_id)
            else:
                socketio.emit("policy_update", {"policy": None}, room=session_id)

    except Exception as e:
        print(f"Transcription/Analysis error for {session_id}: {str(e)}")
        raise

def detect_policy_intent(transcript, session_id):
    print(f"Detecting policy intent for {session_id} with transcript: '{transcript}'")
    policies = Policy.query.all()
    print(f"Found {len(policies)} policies in database")
    
    policy_options = "\n".join(
        [f"Policy: {p.name}, Keywords: {p.keywords}, Summary: {p.text[:100]}..." for p in policies]
    )

    system_message = (
        "You are an AI that selects the most relevant policy based on conversation context. "
        "Focus EXCLUSIVELY on the latest user input to determine the best matching policy, "
        "unless it is ambiguous or lacks sufficient detail (e.g., 'insurance' alone). "
        "Only use the conversation history as secondary context if the latest input is unclear. "
        "Ignore any prior selected policies unless explicitly reaffirmed in the latest input. "
        "If the intent is clear (e.g., 'car insurance' or 'homeowners insurance'), return the policy name. "
        "If ambiguous or unclear, return 'None' and suggest a clarifying question. "
        "Respond with a JSON object: {\"policy_name\": \"[policy name or None]\", \"clarification\": \"[question or null]\"}"
    )

    prompt = (
        f"Conversation History (use only if latest input is ambiguous):\n{json.dumps(conversation_history[session_id], indent=2)}\n\n"
        f"Latest User Input (prioritize this):\n{transcript}\n\n"
        f"Policy Options:\n{policy_options}\n\n"
        "Analyze the user's intent and respond with a JSON object as described."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        ai_output = response.choices[0].message.content.strip()
        print(f"GPT-4 Intent Output for {session_id}: {ai_output}")
        result = json.loads(ai_output)
        policy_name = result["policy_name"]
        clarification = result.get("clarification", None)

        if clarification:
            return None, clarification
        return next((p for p in policies if p.name == policy_name), None) if policy_name != "None" else None, None
    
    except Exception as e:
        print(f"Intent detection error for {session_id}: {str(e)}")
        return None, None

def analyze_with_ai(transcript, policy=None, session_id=None):
    try:
        system_message = (
            "You are an AI assistant ensuring policy compliance. "
            "Your response must strictly use the provided policy text from the database as the basis for your answer. "
            "Do not rely on general knowledge outside the policy text unless explicitly stated. "
            "Return a JSON object with 'policy_name' (exact name of the policy or 'general' if none), "
            "'response' (answer based solely on the policy text), "
            "'disclaimer_said' (true if the telemarketer has likely said the disclaimer, false otherwise), "
            "and 'disclaimer_agreed' (true if the user has agreed to the disclaimer, false otherwise). "
            "Do not include a 'disclaimer' field in the response as it is provided separately."
        )
        policy_text = policy.text if policy else "No specific policy applies."
        disclaimer = Disclaimer.query.filter_by(policy_id=policy.id).first().text if policy and policy.requires_disclaimer else None
        
        # Check if telemarketer likely said the disclaimer (80% confidence)
        disclaimer_said = False
        if disclaimer:
            disclaimer_words = set(disclaimer.lower().split())
            last_messages = [msg["content"].lower() for msg in conversation_history[session_id][-3:]]
            for msg in last_messages:
                msg_words = set(msg.split())
                overlap = len(disclaimer_words & msg_words) / len(disclaimer_words)
                if overlap >= 0.6:
                    disclaimer_said = True
                    conversation_history[session_id].append({"role": "system", "content": f"Disclaimer for {policy.name} likely said"})
                    break
        
        disclaimer_agreed = any(f"Disclaimer for {policy.name} agreed" in msg["content"] 
                                for msg in conversation_history[session_id]) if policy and disclaimer else False
        
        print(f"Policy Text Type for {session_id}: {type(policy_text)}, Value: {policy_text}")
        print(f"Disclaimer Type for {session_id}: {type(disclaimer)}, Value: {disclaimer}")
        print(f"Disclaimer Said for {session_id}: {disclaimer_said}")
        print(f"Disclaimer Agreed for {session_id}: {disclaimer_agreed}")
        
        history_summary = json.dumps(conversation_history[session_id], indent=2)
        prompt = (
            f"Conversation History:\n{history_summary}\n\n"
            f"Policy Text: {str(policy_text)}\n"
            f"User Input: {str(transcript)}\n"
            "Using ONLY the Policy Text above (do not add external information), provide a JSON response with the following structure:\n"
            "{\n"
            "  \"policy_name\": \"[exact policy name or 'general']\",\n"
            "  \"response\": \"[direct quote or paraphrase from policy text]\",\n"
            "  \"disclaimer_said\": [true or false],\n"
            "  \"disclaimer_agreed\": [true or false]\n"
            "}"
        )

        print(f"Sending prompt to GPT-4 for {session_id}: {prompt}")
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        ai_output = response.choices[0].message.content.strip()
        print(f"Raw AI Output for {session_id}: {ai_output}")
        try:
            json_response = json.loads(ai_output)
            json_response["disclaimer_said"] = disclaimer_said
            json_response["disclaimer_agreed"] = disclaimer_agreed
        except json.JSONDecodeError:
            print(f"GPT-4 did not return valid JSON for {session_id}, using fallback")
            json_response = {
                "policy_name": policy.name if policy else "general",
                "response": f"Based on the policy: {policy_text}",
                "disclaimer_said": disclaimer_said,
                "disclaimer_agreed": disclaimer_agreed
            }

        print(f"Parsed AI Response for {session_id}: {json_response}")
        return json_response

    except Exception as e:
        print(f"AI Analysis error for {session_id}: {str(e)}")
        return {
            "policy_name": "error",
            "response": f"Error analyzing: {str(e)}",
            "disclaimer_said": False,
            "disclaimer_agreed": False
        }

def background_task():
    while True:
        eventlet.sleep(1)
        for session_id in audio_buffers:
            if len(audio_buffers[session_id]) >= CHUNKS_TO_PROCESS:
                process_audio_buffer(session_id)

eventlet.spawn(background_task)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print(f"Database connected: {db.engine.url}")
        num_policies = db.session.query(Policy).count()
        print(f"Number of policies in database: {num_policies}")
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)