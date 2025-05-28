from flask import Flask, request, jsonify
from security_system import RaspberryPiFacialRecognitionSecurity
import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Instance globale du système
security_system = RaspberryPiFacialRecognitionSecurity()

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
        "offline_logs_count": len(security_system.offline_logs)
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

if __name__ == '__main__':
    try:
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