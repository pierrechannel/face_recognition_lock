from flask import Flask, request, jsonify, Response
import cv2
import face_recognition
import numpy as np
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

# Flask app initialization
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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
    "connection_error": "Connection error. Operating in offline mode."
}

class TTSManager:
    """Gestionnaire Text-to-Speech pour annonces vocales"""
    
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
        """Configuration du moteur TTS"""
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configuration de la voix
            voices = self.tts_engine.getProperty('voices')
            if voices:
                english_voice = None
                for voice in voices:
                    if 'english' in voice.name.lower() or 'en' in voice.id.lower():
                        english_voice = voice
                        break
                
                if english_voice:
                    self.tts_engine.setProperty('voice', english_voice.id)
                else:
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            self.tts_engine.setProperty('rate', 180)
            self.tts_engine.setProperty('volume', 0.8)
            
            self.is_active = True
            logging.info("Système TTS initialisé")
            
            self.playback_thread = threading.Thread(target=self.tts_playback_loop, daemon=True)
            self.playback_thread.start()
            
        except Exception as e:
            raise Exception(f"Erreur initialisation TTS: {e}")

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
        else:
            logging.warning(f"Message vocal non configuré: {message_key}")
            return False

    def tts_playback_loop(self):
        """Boucle de lecture TTS"""
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
                
                time.sleep(0.1)
                
            except Exception as e:
                logging.error(f"Erreur boucle TTS: {e}")
                time.sleep(0.5)

    def cleanup(self):
        """Nettoyage des ressources TTS"""
        self.is_active = False
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        logging.info("Système TTS fermé")

# Configuration système
KNOWN_FACES_FILE = "known_faces.pkl"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FACE_RECOGNITION_TOLERANCE = 0.6
ACCESS_DISPLAY_DURATION = 3

class FlaskFacialRecognitionSecurity:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.access_lock = Lock()
        self.video_capture = None
        self.access_logs = []
        
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
        
        # Initialisation du TTS
        self.tts_manager = TTSManager()
        
        # Variables pour l'affichage des résultats
        self.access_result = None
        self.access_result_time = 0
        
        # Chargement des visages connus
        self.load_known_faces()
        
        # Initialisation de la webcam
        self.init_camera()
        
        # Annonce de démarrage
        if self.tts_manager.is_active:
            self.tts_manager.speak("system_startup")
        
        self.logger.info("Système de sécurité Flask initialisé")

    def init_camera(self):
        """Initialise la webcam"""
        try:
            self.video_capture = cv2.VideoCapture(0)
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            
            if not self.video_capture.isOpened():
                self.logger.error("Impossible d'ouvrir la webcam")
                return False
            
            self.logger.info("Webcam initialisée")
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
                self.logger.error(f"Erreur lors du chargement des visages: {e}")

    def save_known_faces(self):
        """Sauvegarde les visages connus dans un fichier local"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'ids': self.known_face_ids
            }
            with open(KNOWN_FACES_FILE, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info("Visages sauvegardés localement")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {e}")

    def add_person(self, name, person_id, image_data):
        """Ajoute une nouvelle personne au système"""
        try:
            # Décodage de l'image base64
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            image_array = np.array(image)
            
            # Extraction de l'encodage facial
            face_encodings = face_recognition.face_encodings(image_array)
            if not face_encodings:
                return False, "Aucun visage détecté dans l'image"
            
            # Ajout aux listes
            with self.access_lock:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
                self.known_face_ids.append(person_id)
                self.save_known_faces()
            
            self.logger.info(f"Personne ajoutée: {name}")
            return True, "Personne ajoutée avec succès"
            
        except Exception as e:
            self.logger.error(f"Erreur ajout personne: {e}")
            return False, str(e)

    def remove_person(self, person_id):
        """Supprime une personne du système"""
        try:
            with self.access_lock:
                if person_id in self.known_face_ids:
                    index = self.known_face_ids.index(person_id)
                    self.known_face_encodings.pop(index)
                    self.known_face_names.pop(index)
                    self.known_face_ids.pop(index)
                    self.save_known_faces()
                    self.logger.info(f"Personne supprimée: ID {person_id}")
                    return True, "Personne supprimée avec succès"
                else:
                    return False, "Personne non trouvée"
        except Exception as e:
            self.logger.error(f"Erreur suppression personne: {e}")
            return False, str(e)

    def process_face_recognition(self, frame):
        """Traite la reconnaissance faciale sur une frame"""
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        face_ids = []
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding,
                tolerance=FACE_RECOGNITION_TOLERANCE
            )
            name = "Inconnu"
            person_id = None
            
            if self.known_face_encodings:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    person_id = self.known_face_ids[best_match_index]
            
            face_names.append(name)
            face_ids.append(person_id)
        
        return face_locations, face_names, face_ids

    def image_to_base64(self, image):
        """Convertit une image OpenCV en base64"""
        try:
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
        except Exception as e:
            self.logger.error(f"Erreur conversion image: {e}")
            return None

    def log_access_attempt(self, person_id, person_name, access_granted, image_base64):
        """Enregistre une tentative d'accès"""
        log_entry = {
            "person_id": person_id,
            "person_name": person_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "access_granted": access_granted,
            "image": image_base64,
            "device_id": "flask_security_system"
        }
        
        self.access_logs.append(log_entry)
        
        # Garde seulement les 100 derniers logs
        if len(self.access_logs) > 100:
            self.access_logs = self.access_logs[-100:]
        
        # Annonces vocales
        if access_granted:
            if self.tts_manager.is_active:
                if person_name != "Inconnu":
                    self.tts_manager.speak("person_recognized", name=person_name)
                    Thread(target=self._delayed_access_announcement).start()
                else:
                    self.tts_manager.speak("access_granted")
            self.logger.info(f"ACCÈS AUTORISÉ: {person_name}")
        else:
            if self.tts_manager.is_active:
                if person_name == "Inconnu":
                    self.tts_manager.speak("unknown_person")
                    Thread(target=self._delayed_deny_announcement).start()
                else:
                    self.tts_manager.speak("access_denied")
            self.logger.warning(f"ACCÈS REFUSÉ: {person_name}")

    def _delayed_access_announcement(self):
        """Annonce l'ouverture d'accès avec un délai"""
        time.sleep(1.5)
        if self.tts_manager.is_active:
            self.tts_manager.speak("access_granted")
            time.sleep(0.5)
            self.tts_manager.speak("door_opening")

    def _delayed_deny_announcement(self):
        """Annonce le refus d'accès avec un délai"""
        time.sleep(1.0)
        if self.tts_manager.is_active:
            self.tts_manager.speak("access_denied")

    def get_frame(self):
        """Récupère une frame de la webcam"""
        if self.video_capture is None or not self.video_capture.isOpened():
            return None
        
        ret, frame = self.video_capture.read()
        if ret:
            return frame
        return None

    def cleanup(self):
        """Nettoyage des ressources"""
        if self.video_capture:
            self.video_capture.release()
        if hasattr(self, 'tts_manager'):
            self.tts_manager.cleanup()
        cv2.destroyAllWindows()

# Instance globale du système de sécurité
security_system = FlaskFacialRecognitionSecurity()

# Routes Flask

@app.route('/health', methods=['GET'])
def health_check():
    """Vérification de l'état du système"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "known_persons": len(security_system.known_face_names),
        "camera_status": "active" if security_system.video_capture and security_system.video_capture.isOpened() else "inactive"
    })

@app.route('/persons', methods=['GET'])
def get_persons():
    """Récupère la liste des personnes enregistrées"""
    persons = []
    for i, (name, person_id) in enumerate(zip(security_system.known_face_names, security_system.known_face_ids)):
        persons.append({
            "id": person_id,
            "name": name,
            "index": i
        })
    
    return jsonify({
        "persons": persons,
        "total": len(persons)
    })

@app.route('/persons', methods=['POST'])
def add_person():
    """Ajoute une nouvelle personne"""
    try:
        data = request.get_json()
        
        if not data or 'name' not in data or 'image' not in data:
            return jsonify({"error": "Données manquantes (name, image requis)"}), 400
        
        name = data['name']
        person_id = data.get('id', f"person_{int(time.time())}")
        image_data = data['image']
        
        success, message = security_system.add_person(name, person_id, image_data)
        
        if success:
            return jsonify({"message": message, "person_id": person_id}), 201
        else:
            return jsonify({"error": message}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/persons/<person_id>', methods=['DELETE'])
def remove_person(person_id):
    """Supprime une personne"""
    try:
        success, message = security_system.remove_person(person_id)
        
        if success:
            return jsonify({"message": message})
        else:
            return jsonify({"error": message}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """Reconnaît un visage à partir d'une image"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Image requise"}), 400
        
        image_data = data['image']
        
        # Décodage de l'image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image_array = np.array(image)
        
        # Reconnaissance faciale
        face_locations, face_names, face_ids = security_system.process_face_recognition(image_array)
        
        results = []
        for i, (location, name, person_id) in enumerate(zip(face_locations, face_names, face_ids)):
            access_granted = name != "Inconnu"
            
            # Log de la tentative d'accès
            image_base64 = security_system.image_to_base64(image_array)
            security_system.log_access_attempt(person_id, name, access_granted, image_base64)
            
            results.append({
                "face_index": i,
                "person_id": person_id,
                "name": name,
                "access_granted": access_granted,
                "face_location": {
                    "top": int(location[0] * 4),
                    "right": int(location[1] * 4),
                    "bottom": int(location[2] * 4),
                    "left": int(location[3] * 4)
                }
            })
        
        return jsonify({
            "faces_detected": len(results),
            "results": results,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/camera/capture', methods=['GET'])
def capture_frame():
    """Capture une frame de la webcam"""
    try:
        frame = security_system.get_frame()
        if frame is None:
            return jsonify({"error": "Impossible de capturer l'image"}), 500
        
        # Reconnaissance faciale sur la frame
        face_locations, face_names, face_ids = security_system.process_face_recognition(frame)
        
        # Conversion en base64
        image_base64 = security_system.image_to_base64(frame)
        
        results = []
        for i, (location, name, person_id) in enumerate(zip(face_locations, face_names, face_ids)):
            results.append({
                "face_index": i,
                "person_id": person_id,
                "name": name,
                "access_granted": name != "Inconnu",
                "face_location": {
                    "top": int(location[0] * 4),
                    "right": int(location[1] * 4),
                    "bottom": int(location[2] * 4),
                    "left": int(location[3] * 4)
                }
            })
        
        return jsonify({
            "image": image_base64,
            "faces_detected": len(results),
            "results": results,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/access-logs', methods=['GET'])
def get_access_logs():
    """Récupère les logs d'accès"""
    limit = request.args.get('limit', 50, type=int)
    
    logs = security_system.access_logs[-limit:] if limit > 0 else security_system.access_logs
    
    return jsonify({
        "logs": logs,
        "total": len(security_system.access_logs)
    })

@app.route('/access-logs', methods=['POST'])
def add_access_log():
    """Ajoute un log d'accès externe"""
    try:
        data = request.get_json()
        
        required_fields = ['person_name', 'access_granted']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Champs requis: person_name, access_granted"}), 400
        
        log_entry = {
            "person_id": data.get('person_id'),
            "person_name": data['person_name'],
            "timestamp": data.get('timestamp', datetime.datetime.now().isoformat()),
            "access_granted": data['access_granted'],
            "image": data.get('image'),
            "device_id": data.get('device_id', 'external_system')
        }
        
        security_system.access_logs.append(log_entry)
        
        return jsonify({"message": "Log ajouté avec succès"}), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/system/stats', methods=['GET'])
def get_system_stats():
    """Statistiques du système"""
    total_logs = len(security_system.access_logs)
    granted_access = sum(1 for log in security_system.access_logs if log['access_granted'])
    denied_access = total_logs - granted_access
    
    return jsonify({
        "known_persons": len(security_system.known_face_names),
        "total_access_attempts": total_logs,
        "granted_access": granted_access,
        "denied_access": denied_access,
        "camera_active": security_system.video_capture and security_system.video_capture.isOpened(),
        "tts_active": security_system.tts_manager.is_active
    })

@app.route('/system/announce', methods=['POST'])
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
        
        if success:
            return jsonify({"message": "Annonce programmée"})
        else:
            return jsonify({"error": "Système TTS non disponible"}), 503
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint non trouvé"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Erreur interne du serveur"}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("Arrêt du serveur...")
    finally:
        security_system.cleanup()