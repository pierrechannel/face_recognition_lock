import os
import time
from pathlib import Path

# Base directory configuration
BASE_DIR = os.getenv('SECURITY_SYSTEM_DIR', os.path.expanduser('~/.security_system'))
os.makedirs(BASE_DIR, exist_ok=True)

# Configuration pour Raspberry Pi et serveur backend
BACKEND_API_URL = os.getenv('BACKEND_API_URL', 'https://apps.mediabox.bi:26875/')
DEVICE_ID = os.getenv('DEVICE_ID', f'raspberry_pi_{int(time.time())}')

# Headers pour les requêtes vers le backend (sans authentification)
BACKEND_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": f"RaspberryPi-Security-{DEVICE_ID}"
}

# Configuration des messages vocaux
VOICE_MESSAGES = {
    "access_granted": "Access granted. Welcome.",
    "access_denied": "Access denied. Unauthorized person detected.",
    "system_startup": "Security system initialized and ready.",
    "sync_complete": "Synchronization complete.",
    "error_alert": "System error detected. Please check logs.",
    "door_opening": "Door opening. Please proceed.",
    "recognition_processing": "Processing recognition.",
    "unknown_person": "Unknown person detected.",
    "person_recognized": "Person recognized: {name}",
    "system_ready": "Facial recognition security system is now active.",
    "synchronizing": "Synchronizing with server.",
    "connection_error": "Connection error. Operating in offline mode.",
    "server_reconnected": "Server connection restored."
}

# Configuration système pour Raspberry Pi
KNOWN_FACES_FILE = os.path.join(BASE_DIR, "known_faces.pkl")
OFFLINE_LOGS_FILE = os.path.join(BASE_DIR, "offline_access_logs.pkl")
CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '640'))
CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '480'))
FACE_RECOGNITION_TOLERANCE = float(os.getenv('FACE_RECOGNITION_TOLERANCE', '0.6'))
SYNC_INTERVAL = int(os.getenv('SYNC_INTERVAL', '300'))  # 5 minutes

# Paramètres de connexion
CONNECTION_TIMEOUT = int(os.getenv('CONNECTION_TIMEOUT', '10'))  # 10 secondes par défaut
MAX_RETRY_ATTEMPTS = int(os.getenv('MAX_RETRY_ATTEMPTS', '3'))

# Configuration de la caméra
CAMERA_FPS = int(os.getenv('CAMERA_FPS', '30'))
CAMERA_DEVICE = int(os.getenv('CAMERA_DEVICE', '0'))  # 0 = caméra par défaut

# Configuration de l'audio
AUDIO_VOLUME = float(os.getenv('AUDIO_VOLUME', '0.8'))  # Volume de 0.0 à 1.0
AUDIO_RATE = int(os.getenv('AUDIO_RATE', '150'))  # Vitesse de la parole

# Configuration de journalisation (logging)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.path.join(BASE_DIR, 'security_system.log')
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Validation de base des paramètres
if not BACKEND_API_URL.startswith(('http://', 'https://')):
    print(f"Warning: BACKEND_API_URL should be a valid HTTP/HTTPS URL: {BACKEND_API_URL}")