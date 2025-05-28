import cv2
import face_recognition
import numpy as np
import pickle
import datetime
import os
import time
import logging
from threading import Thread, Lock
import schedule
from PIL import Image
from io import BytesIO
from backend_client import BackendAPIClient
from tts_manager import TTSManager
from config import (BACKEND_API_URL, BACKEND_HEADERS, VOICE_MESSAGES,
                   KNOWN_FACES_FILE, OFFLINE_LOGS_FILE, CAMERA_WIDTH,
                   CAMERA_HEIGHT, FACE_RECOGNITION_TOLERANCE, SYNC_INTERVAL)
from utils import image_to_base64

class RaspberryPiFacialRecognitionSecurity:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.access_lock = Lock()
        self.video_capture = None
        self.offline_logs = []
        self.device_id = BACKEND_API_URL
        self.backend_api_url = BACKEND_API_URL
        self.recognition_cooldown = 3
        self.time = time
        self.voice_messages = VOICE_MESSAGES
        
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
        self.tts_manager = TTSManager(VOICE_MESSAGES)
        
        # Variables pour le contrôle d'accès
        self.last_recognition_time = 0
        
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
        
        self.logger.info(f"Système Raspberry Pi initialisé - Device ID: {self.device_id}")

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
            "device_id": self.device_id,
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
        return image_to_base64(image, self.logger)

    def cleanup(self):
        """Nettoyage des ressources"""
        if self.video_capture:
            self.video_capture.release()
        if hasattr(self, 'tts_manager'):
            self.tts_manager.cleanup()
        cv2.destroyAllWindows()
        self.logger.info("Nettoyage terminé")