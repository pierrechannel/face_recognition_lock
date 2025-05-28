from flask import Flask, request, jsonify, Response
from security_system import RaspberryPiFacialRecognitionSecurity
import datetime
import cv2
import json
import threading
import time
import requests
import base64
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Instance globale du système
security_system = RaspberryPiFacialRecognitionSecurity()

# Configuration pour le streaming
STREAMING_SERVER_URL = "https://apps.mediabox.bi:26875/streaming"  # À configurer
streaming_active = False
streaming_thread = None
stream_lock = threading.Lock()

class StreamingManager:
    def __init__(self):
        self.active = False
        self.fps = 10  # Frames per second pour le streaming
        self.quality = 80  # Qualité JPEG (0-100)
        self.detection_enabled = True
        self.last_frame = None
        self.clients = []  # Liste des clients connectés pour le streaming local
        
    def set_streaming_server(self, url):
        global STREAMING_SERVER_URL
        STREAMING_SERVER_URL = url
        
    def generate_frame_with_detection(self):
        """Génère une frame avec détection de visages"""
        frame = security_system.get_frame()
        if frame is None:
            return None
            
        if self.detection_enabled:
            # Détection des visages
            face_locations, face_names, face_ids = security_system.process_face_recognition(frame)
            
            # Dessiner les rectangles et noms sur l'image
            for location, name in zip(face_locations, face_names):
                top, right, bottom, left = location
                
                # Couleur selon reconnaissance
                color = (0, 255, 0) if name != "Inconnu" else (0, 0, 255)
                
                # Rectangle autour du visage
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Nom sous le rectangle
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                
        return frame
        
    def frame_to_jpeg(self, frame):
        """Convertit une frame en JPEG"""
        if frame is None:
            return None
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return buffer.tobytes()
        
    def start_streaming_to_server(self):
        """Démarre le streaming vers un serveur externe"""
        while self.active:
            try:
                frame = self.generate_frame_with_detection()
                if frame is not None:
                    # Convertir en JPEG
                    jpeg_bytes = self.frame_to_jpeg(frame)
                    if jpeg_bytes:
                        # Encoder en base64 pour l'envoi HTTP
                        frame_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
                        
                        # Préparer les données à envoyer
                        data = {
                            "device_id": security_system.device_id,
                            "timestamp": datetime.datetime.now().isoformat(),
                            "frame": frame_b64,
                            "detection_enabled": self.detection_enabled
                        }
                        
                        # Envoyer au serveur de streaming
                        response = requests.post(
                            f"{STREAMING_SERVER_URL}/receive_stream",
                            json=data,
                            timeout=5
                        )
                        
                        if response.status_code != 200:
                            security_system.logger.warning(f"Erreur envoi stream: {response.status_code}")
                            
                time.sleep(1.0 / self.fps)  # Contrôler le FPS
                
            except Exception as e:
                security_system.logger.error(f"Erreur streaming: {e}")
                time.sleep(1)

streaming_manager = StreamingManager()

def generate_local_stream():
    """Générateur pour le streaming local via HTTP"""
    while streaming_manager.active:
        frame = streaming_manager.generate_frame_with_detection()
        if frame is not None:
            jpeg_bytes = streaming_manager.frame_to_jpeg(frame)
            if jpeg_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
        time.sleep(1.0 / streaming_manager.fps)

@app.route('/health', methods=['GET'])
def health_check():
    """Vérification de l'état du système"""
    return jsonify({
        "status": "healthy",
        "device_id": security_system.device_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "known_persons": len(security_system.known_face_names),
        "camera_status": "active" if security_system.video_capture and security_system.video_capture.isOpened() else "inactive",
        "backend_online": security_system.backend_client.is_online,
        "offline_logs_count": len(security_system.offline_logs),
        "streaming_active": streaming_manager.active,
        "streaming_server": STREAMING_SERVER_URL
    })

@app.route('/capture', methods=['GET'])
def capture_and_recognize():
    """Capture et reconnaît les visages"""
    try:
        frame = security_system.get_frame()
        if frame is None:
            return jsonify({"error": "Impossible de capturer l'image"}), 500
        
        current_time = security_system.time.time()
        
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
            "device_id": security_system.device_id
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
                "offline_logs_sent": len([])  # Les logs ont été envoyés
            })
        else:
            return jsonify({"error": "Échec de la synchronisation"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_system_status():
    """Statut détaillé du système"""
    return jsonify({
        "device_id": security_system.device_id,
        "system_time": datetime.datetime.now().isoformat(),
        "known_persons": len(security_system.known_face_names),
        "camera_active": security_system.video_capture and security_system.video_capture.isOpened(),
        "tts_active": security_system.tts_manager.is_active,
        "backend_online": security_system.backend_client.is_online,
        "offline_logs_pending": len(security_system.offline_logs),
        "last_recognition": security_system.last_recognition_time,
        "backend_url": security_system.backend_api_url,
        "recognition_cooldown": security_system.recognition_cooldown,
        "streaming": {
            "active": streaming_manager.active,
            "fps": streaming_manager.fps,
            "quality": streaming_manager.quality,
            "detection_enabled": streaming_manager.detection_enabled,
            "server_url": STREAMING_SERVER_URL
        }
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
        
        if message_type == 'predefined' and message in security_system.voice_messages:
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

# === NOUVELLES ROUTES POUR LE STREAMING ===

@app.route('/stream/start', methods=['POST'])
def start_streaming():
    """Démarre le streaming"""
    global streaming_thread
    
    try:
        data = request.get_json() or {}
        
        # Configuration optionnelle
        if 'server_url' in data:
            streaming_manager.set_streaming_server(data['server_url'])
        if 'fps' in data:
            streaming_manager.fps = max(1, min(30, data['fps']))
        if 'quality' in data:
            streaming_manager.quality = max(10, min(100, data['quality']))
        if 'detection_enabled' in data:
            streaming_manager.detection_enabled = data['detection_enabled']
        
        if streaming_manager.active:
            return jsonify({"message": "Streaming déjà actif"}), 400
        
        streaming_manager.active = True
        
        # Démarrer le thread de streaming vers serveur externe
        streaming_thread = threading.Thread(target=streaming_manager.start_streaming_to_server)
        streaming_thread.daemon = True
        streaming_thread.start()
        
        return jsonify({
            "message": "Streaming démarré",
            "config": {
                "fps": streaming_manager.fps,
                "quality": streaming_manager.quality,
                "detection_enabled": streaming_manager.detection_enabled,
                "server_url": STREAMING_SERVER_URL
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stream/stop', methods=['POST'])
def stop_streaming():
    """Arrête le streaming"""
    streaming_manager.active = False
    
    return jsonify({
        "message": "Streaming arrêté",
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/stream/config', methods=['GET', 'POST'])
def stream_config():
    """Configuration du streaming"""
    if request.method == 'GET':
        return jsonify({
            "fps": streaming_manager.fps,
            "quality": streaming_manager.quality,
            "detection_enabled": streaming_manager.detection_enabled,
            "server_url": STREAMING_SERVER_URL,
            "active": streaming_manager.active
        })
    
    elif request.method == 'POST':
        data = request.get_json() or {}
        
        if 'fps' in data:
            streaming_manager.fps = max(1, min(30, data['fps']))
        if 'quality' in data:
            streaming_manager.quality = max(10, min(100, data['quality']))
        if 'detection_enabled' in data:
            streaming_manager.detection_enabled = data['detection_enabled']
        if 'server_url' in data:
            streaming_manager.set_streaming_server(data['server_url'])
        
        return jsonify({
            "message": "Configuration mise à jour",
            "config": {
                "fps": streaming_manager.fps,
                "quality": streaming_manager.quality,
                "detection_enabled": streaming_manager.detection_enabled,
                "server_url": STREAMING_SERVER_URL
            }
        })

@app.route('/stream/live')
def live_stream():
    """Stream vidéo en direct (MJPEG) - pour visualisation locale"""
    if not streaming_manager.active:
        streaming_manager.active = True
    
    return Response(
        generate_local_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stream/snapshot', methods=['GET'])
def get_snapshot():
    """Capture une image instantanée avec détection"""
    try:
        frame = streaming_manager.generate_frame_with_detection()
        if frame is None:
            return jsonify({"error": "Impossible de capturer l'image"}), 500
        
        # Convertir en base64
        jpeg_bytes = streaming_manager.frame_to_jpeg(frame)
        if jpeg_bytes:
            frame_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
            
            return jsonify({
                "image": frame_b64,
                "timestamp": datetime.datetime.now().isoformat(),
                "device_id": security_system.device_id,
                "detection_enabled": streaming_manager.detection_enabled
            })
        else:
            return jsonify({"error": "Erreur encodage image"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(
            host='0.0.0.0', 
            port=5001, 
            debug=True, 
            threaded=True,
            use_reloader=False  # Important pour éviter les problèmes sur RPi
        )
    except KeyboardInterrupt:
        print("Arrêt du serveur...")
    finally:
        streaming_manager.active = False
        security_system.cleanup()
        security_system.run_real_time_recognition()