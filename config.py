import os
import time

# Configuration pour Raspberry Pi et serveur backend
BACKEND_API_URL = os.getenv('BACKEND_API_URL', 'https://your-backend-server.com/api')
API_KEY = os.getenv('API_KEY', 'your_api_key_here')
DEVICE_ID = os.getenv('DEVICE_ID', f'raspberry_pi_{int(time.time())}')

# Headers pour les requêtes vers le backend
BACKEND_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
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
KNOWN_FACES_FILE = "known_faces.pkl"
OFFLINE_LOGS_FILE = "offline_access_logs.pkl"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FACE_RECOGNITION_TOLERANCE = 0.6
SYNC_INTERVAL = 300  # 5 minutes