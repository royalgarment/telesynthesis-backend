from flask import Flask
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
audio_buffer = deque()
header_chunk = None
TEMP_WEBM_FILE = "temp_audio.webm"
TEMP_WAV_FILE = "temp_audio.wav"
CHUNKS_TO_PROCESS = 20
DEBUG_WEBM_FILE = "debug_audio.webm"
FAILED_PROCESS_COUNT = 0
MAX_FAILED_PROCESSES = 5

# Database Models
class Policy(db.Model):
    __tablename__ = 'policies'  # Explicit table name
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(100))
    text = db.Column(db.Text, nullable=False)
    keywords = db.Column(db.String(255))
    color_code = db.Column(db.String(7))
    requires_disclaimer = db.Column(db.Boolean, default=False)

class Disclaimer(db.Model):
    __tablename__ = 'disclaimers'  # Explicit table name
    id = db.Column(db.Integer, primary_key=True)
    policy_id = db.Column(db.Integer, db.ForeignKey('policies.id'))
    text = db.Column(db.Text, nullable=False)

@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    global header_chunk
    print(f"Received audio chunk of size {len(data)} bytes")
    if header_chunk is None:
        header_chunk = data
    audio_buffer.append(data)
    print(f"Buffer size: {len(audio_buffer)} chunks")
    if len(audio_buffer) >= CHUNKS_TO_PROCESS:
        process_audio_buffer()

@socketio.on("connect")
def handle_connect():
    print("Client connected")
    global header_chunk, FAILED_PROCESS_COUNT
    header_chunk = None
    FAILED_PROCESS_COUNT = 0
    with open(TEMP_WEBM_FILE, "wb") as f:
        f.write(b"")

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")
    global audio_buffer, header_chunk, FAILED_PROCESS_COUNT
    audio_buffer.clear()
    header_chunk = None
    FAILED_PROCESS_COUNT = 0
    for temp_file in [TEMP_WEBM_FILE, TEMP_WAV_FILE, DEBUG_WEBM_FILE]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def process_audio_buffer():
    global audio_buffer, header_chunk, FAILED_PROCESS_COUNT
    if not audio_buffer or header_chunk is None:
        print("Buffer empty or no header, skipping processing")
        return

    print(f"Processing {len(audio_buffer)} chunks...")
    with open(TEMP_WEBM_FILE, "wb") as f:
        print(f"Writing header chunk of size {len(header_chunk)} bytes")
        f.write(header_chunk)
        subsequent_chunks = list(audio_buffer)[1:]
        total_subsequent_size = sum(len(chunk) for chunk in subsequent_chunks)
        print(f"Writing {len(subsequent_chunks)} subsequent chunks, total size: {total_subsequent_size} bytes")
        f.write(b"".join(subsequent_chunks))
    with open(DEBUG_WEBM_FILE, "wb") as f:
        f.write(header_chunk)
        f.write(b"".join(list(audio_buffer)[1:]))

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

        transcribe_and_analyze(audio_segment)
        audio_buffer.clear()
        FAILED_PROCESS_COUNT = 0

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        FAILED_PROCESS_COUNT += 1
        print(f"Failed process count: {FAILED_PROCESS_COUNT}")
        if FAILED_PROCESS_COUNT >= MAX_FAILED_PROCESSES:
            print("Too many FFmpeg failures, resetting header_chunk")
            header_chunk = None
        audio_buffer.clear()
    except Exception as e:
        print(f"Processing error: {str(e)}")
        audio_buffer.clear()

def transcribe_and_analyze(audio_segment):
    try:
        print("Exporting audio to buffer...")
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.name = "audio.wav"
        wav_buffer.seek(0)

        print("Sending to Whisper API...")
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=wav_buffer,
            language="en"
        ).text

        print(f"Transcription: {transcript} (Type: {type(transcript)})")
        socketio.emit("transcript_update", {"transcript": transcript})

        policy = detect_policy_intent(transcript)
        print(f"Detected Policy: {policy.name if policy else 'None'}")
        policy_data = {
            "id": policy.id,
            "name": policy.name,
            "color_code": policy.color_code,
            "requires_disclaimer": policy.requires_disclaimer
        } if policy else None
        socketio.emit("policy_update", {"policy": policy_data})

        ai_response = analyze_with_ai(transcript, policy)
        socketio.emit("ai_response", ai_response)

    except Exception as e:
        print(f"Transcription/Analysis error: {str(e)}")
        raise

def detect_policy_intent(transcript):
    print(f"Checking policies for transcript: '{transcript}'")
    policies = Policy.query.all()
    print(f"Found {len(policies)} policies in database")
    for policy in policies:
        keywords = policy.keywords.split(",")
        print(f"Policy: {policy.name}, Keywords: {keywords}")
        if any(keyword.strip().lower() in transcript.lower() for keyword in keywords):
            print(f"Matched policy: {policy.name}")
            return policy
    print("No policy matched")
    return None

def analyze_with_ai(transcript, policy=None):
    try:
        system_message = (
            "You are an AI assistant ensuring policy compliance. "
            "Your response must strictly use the provided policy text from the database as the basis for your answer. "
            "Do not rely on general knowledge outside the policy text unless explicitly stated. "
            "Return a JSON object with 'policy_name' (exact name of the policy or 'general' if none), "
            "'response' (answer based solely on the policy text), and 'disclaimer' (from the policy if applicable, otherwise null)."
        )
        policy_text = policy.text if policy else "No specific policy applies."
        disclaimer = Disclaimer.query.filter_by(policy_id=policy.id).first().text if policy and policy.requires_disclaimer else None
        
        print(f"Policy Text Type: {type(policy_text)}, Value: {policy_text}")
        print(f"Disclaimer Type: {type(disclaimer)}, Value: {disclaimer}")
        prompt = (
            f"Policy Text: {str(policy_text)}\n"
            f"User Input: {str(transcript)}\n"
            "Using ONLY the Policy Text above (do not add external information), provide a JSON response with the following structure:\n"
            "{\n"
            "  \"policy_name\": \"[exact policy name or 'general']\",\n"
            "  \"response\": \"[direct quote or paraphrase from policy text]\",\n"
            "  \"disclaimer\": \"[disclaimer text or null]\"\n"
            "}"
        )

        print(f"Sending prompt to GPT-4: {prompt}")
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        ai_output = response.choices[0].message.content.strip()
        print(f"Raw AI Output: {ai_output}")
        try:
            json_response = json.loads(ai_output)
        except json.JSONDecodeError:
            print("GPT-4 did not return valid JSON, using fallback")
            json_response = {
                "policy_name": policy.name if policy else "general",
                "response": f"Based on the policy: {policy_text}",
                "disclaimer": disclaimer
            }

        print(f"Parsed AI Response: {json_response}")
        return json_response

    except Exception as e:
        print(f"AI Analysis error: {str(e)}")
        return {
            "policy_name": "error",
            "response": f"Error analyzing: {str(e)}",
            "disclaimer": None
        }

def background_task():
    while True:
        eventlet.sleep(1)
        if len(audio_buffer) >= CHUNKS_TO_PROCESS:
            process_audio_buffer()

eventlet.spawn(background_task)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print(f"Database connected: {db.engine.url}")
        num_policies = db.session.query(Policy).count()
        print(f"Number of policies in database: {num_policies}")
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)