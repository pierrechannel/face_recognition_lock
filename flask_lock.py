from flask import Flask, request, jsonify, Response
import cv2
import face_recognition
import numpy as np
import requests
import json
import os
import pickle
import datetime
import time
import logging
from threading import Thread, Lock
import base64
from io import BytesIO
from PIL import Image
import threading
from collections import deque
import pyttsx3
import schedule
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Flask app initialization
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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

class BackendAPIClient:
    """Client pour communiquer avec le serveur backend"""
    
    def __init__(self, base_url, headers):
        self.base_url = base_url.rstrip('/')
        self.headers = headers
        self.session = requests.Session()
        
        # Configuration des retry pour la robustesse
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.is_online = False
        self.last_sync_time = 0
        
    def test_connection(self):
        """Test la connexion au serveur backend"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=10
            )
            self.is_online = response.status_code == 200
            return self.is_online
        except Exception as e:
            logging.error(f"Test connexion backend échoué: {e}")
            self.is_online = False
            return False
    
    def get_persons(self):
        """Récupère la liste des personnes depuis le backend"""
        try:
            response = self.session.get(
                f"{self.base_url}/persons",
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                return True, data.get('persons', [])
            else:
                logging.error(f"Erreur récupération personnes: {response.status_code}")
                return False, []
                
        except Exception as e:
            logging.error(f"Erreur connexion pour récupération personnes: {e}")
            return False, []
    
    def post_access_log(self, access_data):
        """Envoie un log d'accès au backend"""
        try:
            response = self.session.post(
                f"{self.base_url}/access-logs",
                headers=self.headers,
                json=access_data,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                logging.info(f"Log d'accès envoyé: {access_data['person_name']}")
                return True
            else:
                logging.error(f"Erreur envoi log: {response.status_code}")
                return False
                
        except Exception as e:
            logging.error(f"Erreur connexion pour envoi log: {e}")
            return False
    
    def download_person_image(self, image_url):
        """Télécharge l'image d'une personne depuis le backend"""
        try:
            response = self.session.get(image_url, timeout=15)
            if response.status_code == 200:
                return response.content
            return None
        except Exception as e:
            logging.error(f"Erreur téléchargement image {image_url}: {e}")
            return None

class TTSManager:
    """Gestionnaire Text-to-Speech optimisé pour Raspberry Pi"""
    
    def __init__(self):
        self.is_active = False
        self.tts_queue = deque()
        self.tts_lock = threading.Lock()
        self.playback_thread = None
        self.tts_engine = None
        
        try:
            self.setup_tts_engine()
        except Exception as e:
            logging.warning(f"TTS non disponible: {e}")
            self.is_active = False

    def setup_tts_engine(self):
        """Configuration du moteur TTS pour Raspberry Pi"""
        try:
            self.tts_engine = pyttsx3.init(driverName='espeak')  # Utilise espeak sur RPi
            
            # Configuration optimisée pour Raspberry Pi
            self.tts_engine.setProperty('rate', 150)  # Vitesse plus lente pour clarté
            self.tts_engine.setProperty('volume', 0.9)
            
            # Utilise une voix anglaise si disponible
            voices = self.tts_engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'english' in voice.name.lower() or 'en' in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            self.is_active = True
            logging.info("Système TTS initialisé pour Raspberry Pi")
            
            # Démarrage du thread de lecture
            self.playback_thread = threading.Thread(target=self.tts_playback_loop, daemon=True)
            self.playback_thread.start()
            
        except Exception as e:
            logging.error(f"Erreur initialisation TTS: {e}")
            self.is_active = False

    def speak(self, message_key, **kwargs):
        """Ajoute un message à la queue TTS"""
        if not self.is_active:
            return False
            
        if message_key in VOICE_MESSAGES:
            message = VOICE_MESSAGES[message_key]
            
            if kwargs:
                try:
                    message = message.format(**kwargs)
                except KeyError as e:
                    logging.warning(f"Paramètre manquant pour le message {message_key}: {e}")
            
            with self.tts_lock:
                self.tts_queue.append(message)
            
            logging.debug(f"Message TTS ajouté: {message}")
            return True
        return False

    def speak_custom(self, message):
        """Ajoute un message personnalisé à la queue TTS"""
        if not self.is_active:
            return False
            
        with self.tts_lock:
            self.tts_queue.append(message)
        return True

    def tts_playback_loop(self):
        """Boucle de lecture TTS optimisée pour RPi"""
        while self.is_active:
            try:
                with self.tts_lock:
                    if self.tts_queue:
                        message = self.tts_queue.popleft()
                        
                        try:
                            self.tts_engine.say(message)
                            self.tts_engine.runAndWait()
                        except Exception as e:
                            logging.error(f"Erreur lecture TTS: {e}")
                
                time.sleep(0.2)  # Pause plus longue pour RPi
                
            except Exception as e:
                logging.error(f"Erreur boucle TTS: {e}")
                time.sleep(1)

    def cleanup(self):
        """Nettoyage des ressources TTS"""
        self.is_active = False
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass

# Configuration système pour Raspberry Pi
KNOWN_FACES_FILE = "known_faces.pkl"
OFFLINE_LOGS_FILE = "offline_access_logs.pkl"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FACE_RECOGNITION_TOLERANCE = 0.6
SYNC_INTERVAL = 300  # 5 minutes

class RaspberryPiFacialRecognitionSecurity:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.access_lock = Lock()
        self.video_capture = None
        self.offline_logs = []
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('security_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Client API backend
        self.backend_client = BackendAPIClient(BACKEND_API_URL, BACKEND_HEADERS)
        
        # Initialisation du TTS
        self.tts_manager = TTSManager()
        
        # Variables pour le contrôle d'accès
        self.last_recognition_time = 0
        self.recognition_cooldown = 3  # 3 secondes entre reconnaissances
        
        # Chargement des données locales
        self.load_known_faces()
        self.load_offline_logs()
        
        # Initialisation de la webcam
        self.init_camera()
        
        # Synchronisation initiale
        self.sync_with_backend()
        
        # Planification des tâches
        self.setup_scheduled_tasks()
        
        # Annonce de démarrage
        if self.tts_manager.is_active:
            self.tts_manager.speak("system_startup")
        
        self.logger.info(f"Système Raspberry Pi initialisé - Device ID: {DEVICE_ID}")

    def init_camera(self):
        """Initialise la webcam pour Raspberry Pi"""
        try:
            # Configuration spécifique pour Raspberry Pi Camera
            self.video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
            
            # Paramètres optimisés pour RPi
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.video_capture.set(cv2.CAP_PROP_FPS, 15)  # 15 FPS pour économiser les ressources
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.video_capture.isOpened():
                self.logger.error("Impossible d'ouvrir la webcam")
                return False
            
            self.logger.info("Webcam initialisée pour Raspberry Pi")
            return True
        except Exception as e:
            self.logger.error(f"Erreur initialisation webcam: {e}")
            return False

    def load_known_faces(self):
        """Charge les visages connus depuis le fichier local"""
        if os.path.exists(KNOWN_FACES_FILE):
            try:
                with open(KNOWN_FACES_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                    self.known_face_ids = data.get('ids', [])
                self.logger.info(f"Chargé {len(self.known_face_names)} visages connus")
            except Exception as e:
                self.logger.error(f"Erreur chargement visages: {e}")

    def save_known_faces(self):
        """Sauvegarde les visages connus"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'ids': self.known_face_ids,
                'last_update': time.time()
            }
            with open(KNOWN_FACES_FILE, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info("Visages sauvegardés")
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde visages: {e}")

    def load_offline_logs(self):
        """Charge les logs hors ligne"""
        if os.path.exists(OFFLINE_LOGS_FILE):
            try:
                with open(OFFLINE_LOGS_FILE, 'rb') as f:
                    self.offline_logs = pickle.load(f)
                self.logger.info(f"Chargé {len(self.offline_logs)} logs hors ligne")
            except Exception as e:
                self.logger.error(f"Erreur chargement logs hors ligne: {e}")

    def save_offline_logs(self):
        """Sauvegarde les logs hors ligne"""
        try:
            with open(OFFLINE_LOGS_FILE, 'wb') as f:
                pickle.dump(self.offline_logs, f)
        except Exception as e:
            self.logger.error(f"Erreur sauvegarde logs hors ligne: {e}")

    def sync_with_backend(self):
        """Synchronise avec le serveur backend"""
        try:
            if self.tts_manager.is_active:
                self.tts_manager.speak("synchronizing")
            
            # Test de connexion
            if not self.backend_client.test_connection():
                self.logger.warning("Backend non accessible - Mode hors ligne")
                if self.tts_manager.is_active:
                    self.tts_manager.speak("connection_error")
                return False
            
            # Récupération des personnes
            success, persons = self.backend_client.get_persons()
            if success:
                self.update_known_faces_from_backend(persons)
                self.logger.info(f"Synchronisé {len(persons)} personnes")
                
                # Envoi des logs hors ligne
                self.send_offline_logs()
                
                if self.tts_manager.is_active:
                    self.tts_manager.speak("sync_complete")
                
                return True
            else:
                if self.tts_manager.is_active:
                    self.tts_manager.speak("error_alert")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur synchronisation: {e}")
            if self.tts_manager.is_active:
                self.tts_manager.speak("connection_error")
            return False

    def update_known_faces_from_backend(self, persons):
        """Met à jour les visages depuis le backend"""
        new_encodings = []
        new_names = []
        new_ids = []
        
        for person in persons:
            try:
                person_id = person.get('id')
                name = person.get('name')
                image_url = person.get('image_url')
                
                if not all([person_id, name, image_url]):
                    continue
                
                # Téléchargement de l'image
                image_data = self.backend_client.download_person_image(image_url)
                if not image_data:
                    continue
                
                # Traitement de l'image
                image = Image.open(BytesIO(image_data))
                image_array = np.array(image)
                
                # Extraction de l'encodage facial
                face_encodings = face_recognition.face_encodings(image_array)
                if face_encodings:
                    new_encodings.append(face_encodings[0])
                    new_names.append(name)
                    new_ids.append(person_id)
                    
            except Exception as e:
                self.logger.error(f"Erreur traitement personne {person.get('name', 'Unknown')}: {e}")
        
        # Mise à jour atomique
        with self.access_lock:
            self.known_face_encodings = new_encodings
            self.known_face_names = new_names
            self.known_face_ids = new_ids
            self.save_known_faces()

    def send_offline_logs(self):
        """Envoie les logs stockés hors ligne"""
        if not self.offline_logs:
            return
        
        sent_logs = []
        for log in self.offline_logs:
            if self.backend_client.post_access_log(log):
                sent_logs.append(log)
        
        # Supprime les logs envoyés
        for log in sent_logs:
            self.offline_logs.remove(log)
        
        if sent_logs:
            self.save_offline_logs()
            self.logger.info(f"Envoyé {len(sent_logs)} logs hors ligne")

    def process_face_recognition(self, frame):
        """Traite la reconnaissance faciale - optimisé pour RPi"""
        # Redimensionnement plus important pour économiser les ressources
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Détection des visages
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")  # HOG plus rapide sur RPi
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        face_ids = []
        
        for face_encoding in face_encodings:
            name = "Inconnu"
            person_id = None
            
            if self.known_face_encodings:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding,
                    tolerance=FACE_RECOGNITION_TOLERANCE
                )
                
                if True in matches:
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        person_id = self.known_face_ids[best_match_index]
            
            face_names.append(name)
            face_ids.append(person_id)
        
        return face_locations, face_names, face_ids

    def log_access_attempt(self, person_id, person_name, access_granted, image_base64):
        """Enregistre une tentative d'accès"""
        log_entry = {
            "person_id": person_id,
            "person_name": person_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "access_granted": access_granted,
            "image": image_base64,
            "device_id": DEVICE_ID,
            "location": "raspberry_pi_entrance"
        }
        
        # Tentative d'envoi immédiat
        if self.backend_client.is_online and self.backend_client.post_access_log(log_entry):
            self.logger.info(f"Log envoyé immédiatement: {person_name}")
        else:
            # Stockage hors ligne
            self.offline_logs.append(log_entry)
            self.save_offline_logs()
            self.logger.info(f"Log stocké hors ligne: {person_name}")
        
        # Annonces vocales
        self.handle_access_result(access_granted, person_name)

    def handle_access_result(self, access_granted, person_name):
        """Gère les annonces vocales et actions d'accès"""
        if access_granted:
            if self.tts_manager.is_active:
                if person_name != "Inconnu":
                    self.tts_manager.speak("person_recognized", name=person_name)
                    Thread(target=self._delayed_access_announcement).start()
                else:
                    self.tts_manager.speak("access_granted")
            
            self.logger.info(f"ACCÈS AUTORISÉ: {person_name}")
            
            # Ici vous pouvez ajouter le code pour ouvrir la porte/serrure
            # GPIO control for door lock/relay
            
        else:
            if self.tts_manager.is_active:
                if person_name == "Inconnu":
                    self.tts_manager.speak("unknown_person")
                    Thread(target=self._delayed_deny_announcement).start()
                else:
                    self.tts_manager.speak("access_denied")
            
            self.logger.warning(f"ACCÈS REFUSÉ: {person_name}")

    def _delayed_access_announcement(self):
        """Annonce d'accès avec délai"""
        time.sleep(1.5)
        if self.tts_manager.is_active:
            self.tts_manager.speak("access_granted")
            time.sleep(0.5)
            self.tts_manager.speak("door_opening")

    def _delayed_deny_announcement(self):
        """Annonce de refus avec délai"""
        time.sleep(1.0)
        if self.tts_manager.is_active:
            self.tts_manager.speak("access_denied")

    def setup_scheduled_tasks(self):
        """Configure les tâches planifiées"""
        schedule.every(5).minutes.do(self.sync_with_backend)
        schedule.every(1).hours.do(self.cleanup_old_logs)
        
        # Démarre le thread pour les tâches planifiées
        scheduler_thread = Thread(target=self.run_scheduler, daemon=True)
        scheduler_thread.start()

    def run_scheduler(self):
        """Exécute les tâches planifiées"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Vérifie chaque minute

    def cleanup_old_logs(self):
        """Nettoie les anciens logs hors ligne"""
        if not self.offline_logs:
            return
        
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=7)
        
        initial_count = len(self.offline_logs)
        self.offline_logs = [
            log for log in self.offline_logs 
            if datetime.datetime.fromisoformat(log['timestamp']) > cutoff_time
        ]
        
        if len(self.offline_logs) < initial_count:
            self.save_offline_logs()
            self.logger.info(f"Nettoyé {initial_count - len(self.offline_logs)} anciens logs")

    def get_frame(self):
        """Récupère une frame de la webcam"""
        if self.video_capture is None or not self.video_capture.isOpened():
            return None
        
        ret, frame = self.video_capture.read()
        return frame if ret else None

    def image_to_base64(self, image):
        """Convertit une image OpenCV en base64"""
        try:
            # Compression JPEG pour réduire la taille
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, buffer = cv2.imencode('.jpg', image, encode_param)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Erreur conversion image: {e}")
            return None

    def cleanup(self):
        """Nettoyage des ressources"""
        if self.video_capture:
            self.video_capture.release()
        if hasattr(self, 'tts_manager'):
            self.tts_manager.cleanup()
        cv2.destroyAllWindows()
        self.logger.info("Nettoyage terminé")

# Instance globale du système
security_system = RaspberryPiFacialRecognitionSecurity()

# Routes Flask simplifiées pour Raspberry Pi

@app.route('/health', methods=['GET'])
def health_check():
    """Vérification de l'état du système"""
    return jsonify({
        "status": "healthy",
        "device_id": DEVICE_ID,
        "timestamp": datetime.datetime.now().isoformat(),
        "known_persons": len(security_system.known_face_names),
        "camera_status": "active" if security_system.video_capture and security_system.video_capture.isOpened() else "inactive",
        "backend_online": security_system.backend_client.is_online,
        "offline_logs_count": len(security_system.offline_logs)
    })

@app.route('/capture', methods=['GET'])
def capture_and_recognize():
    """Capture et reconnaît les visages"""
    try:
        frame = security_system.get_frame()
        if frame is None:
            return jsonify({"error": "Impossible de capturer l'image"}), 500
        
        current_time = time.time()
        
        # Contrôle du cooldown
        if (current_time - security_system.last_recognition_time) < security_system.recognition_cooldown:
            return jsonify({
                "message": "Cooldown actif",
                "next_recognition_in": security_system.recognition_cooldown - (current_time - security_system.last_recognition_time)
            })
        
        # Reconnaissance faciale
        face_locations, face_names, face_ids = security_system.process_face_recognition(frame)
        
        results = []
        if face_locations:  # Si des visages sont détectés
            for i, (location, name, person_id) in enumerate(zip(face_locations, face_names, face_ids)):
                access_granted = name != "Inconnu"
                
                # Log de l'accès
                image_base64 = security_system.image_to_base64(frame)
                security_system.log_access_attempt(person_id, name, access_granted, image_base64)
                
                results.append({
                    "face_index": i,
                    "person_id": person_id,
                    "name": name,
                    "access_granted": access_granted,
                    "confidence": "high" if name != "Inconnu" else "low"
                })
            
            security_system.last_recognition_time = current_time
        
        return jsonify({
            "faces_detected": len(results),
            "results": results,
            "timestamp": datetime.datetime.now().isoformat(),
            "device_id": DEVICE_ID
        })
        
    except Exception as e:
        security_system.logger.error(f"Erreur capture: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sync', methods=['POST'])
def manual_sync():
    """Synchronisation manuelle avec le backend"""
    try:
        success = security_system.sync_with_backend()
        if success:
            return jsonify({
                "message": "Synchronisation réussie",
                "known_persons": len(security_system.known_face_names),
                "offline_logs_sent": len([]) # Les logs ont été envoyés
            })
        else:
            return jsonify({"error": "Échec de la synchronisation"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_system_status():
    """Statut détaillé du système"""
    return jsonify({
        "device_id": DEVICE_ID,
        "system_time": datetime.datetime.now().isoformat(),
        "known_persons": len(security_system.known_face_names),
        "camera_active": security_system.video_capture and security_system.video_capture.isOpened(),
        "tts_active": security_system.tts_manager.is_active,
        "backend_online": security_system.backend_client.is_online,
        "offline_logs_pending": len(security_system.offline_logs),
        "last_recognition": security_system.last_recognition_time,
        "backend_url": BACKEND_API_URL,
        "recognition_cooldown": security_system.recognition_cooldown
    })

@app.route('/announce', methods=['POST'])
def make_announcement():
    """Fait une annonce vocale"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Message requis"}), 400
        
        message = data['message']
        message_type = data.get('type', 'custom')
        
        if message_type == 'predefined' and message in VOICE_MESSAGES:
            success = security_system.tts_manager.speak(message)
        else:
            success = security_system.tts_manager.speak_custom(message)
        
        return jsonify({
            "message": "Annonce programmée" if success else "TTS non disponible",
            "success": success
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/logs/offline', methods=['GET'])
def get_offline_logs():
    """Récupère les logs en attente d'envoi"""
    return jsonify({
        "offline_logs": security_system.offline_logs,
        "count": len(security_system.offline_logs)
    })

if __name__ == '__main__':
    try:
        # Configuration pour Raspberry Pi
        app.run(
            host='0.0.0.0', 
            port=5000, 
            debug=False, 
            threaded=True,
            use_reloader=False  # Important pour éviter les problèmes sur RPi
        )
    except KeyboardInterrupt:
        print("Arrêt du serveur...")
    finally:
        security_system.cleanup()