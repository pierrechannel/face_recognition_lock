from security_system import RaspberryPiFacialRecognitionSecurity
import datetime
import cv2
import threading
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Instance globale du système
security_system = RaspberryPiFacialRecognitionSecurity()

# Variables pour le streaming
stream_active = False
stream_lock = threading.Lock()
stream_frame = None

def update_stream_frame():
    """Met à jour la frame pour le streaming en continu"""
    global stream_active, stream_frame
    
    while stream_active:
        try:
            frame = security_system.get_frame()
            if frame is not None:
                # Redimensionner pour réduire la bande passante si nécessaire
                # frame = cv2.resize(frame, (320, 240))
                
                # Ajouter des informations sur l'image si souhaité
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Convertir en JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                
                # Mettre à jour la frame de streaming
                with stream_lock:
                    stream_frame = buffer.tobytes()
            
            # Pause pour limiter l'utilisation CPU
            time.sleep(0.05)  # ~20 FPS
            
        except Exception as e:
            security_system.logger.error(f"Erreur mise à jour stream: {e}")
            time.sleep(0.5)  # Pause plus longue en cas d'erreur

def generate_frames():
    """Générateur pour le streaming MJPEG"""
    global stream_frame, stream_active
    
    # Démarrer le thread de mise à jour si pas déjà actif
    if not stream_active:
        stream_active = True
        threading.Thread(target=update_stream_frame, daemon=True).start()
    
    while True:
        try:
            # Attendre qu'une frame soit disponible
            while stream_frame is None:
                time.sleep(0.1)
            
            # Récupérer la frame actuelle
            with stream_lock:
                frame_data = stream_frame
            
            # Envoyer la frame au format MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            
            # Petite pause pour éviter de surcharger le client
            time.sleep(0.05)
            
        except Exception as e:
            security_system.logger.error(f"Erreur génération stream: {e}")
            time.sleep(0.5)

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

@app.route('/stream', methods=['GET'])
def video_stream():
    """Flux vidéo en direct (MJPEG)"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream/start', methods=['POST'])
def start_stream():
    """Démarre le streaming vidéo"""
    global stream_active
    
    if not stream_active:
        stream_active = True
        threading.Thread(target=update_stream_frame, daemon=True).start()
        return jsonify({"message": "Streaming démarré"})
    else:
        return jsonify({"message": "Streaming déjà actif"})

@app.route('/stream/stop', methods=['POST'])
def stop_stream():
    """Arrête le streaming vidéo"""
    global stream_active
    
    if stream_active:
        stream_active = False
        return jsonify({"message": "Streaming arrêté"})
    else:
        return jsonify({"message": "Streaming déjà inactif"})

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
        "streaming_active": stream_active
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

@app.route('/stream/html', methods=['GET'])
def stream_page():
    """Page HTML simple pour visualiser le stream"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raspberry Pi Security Camera</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 20px; }
            h1 { color: #333; }
            .stream-container { 
                margin: 20px auto; 
                max-width: 800px; 
                border: 1px solid #ccc; 
                padding: 10px;
            }
            img { max-width: 100%; height: auto; }
            .controls { margin: 20px 0; }
            button {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 4px;
            }
            button.stop { background-color: #f44336; }
            .status { margin-top: 20px; font-style: italic; }
        </style>
    </head>
    <body>
        <h1>Raspberry Pi Security Camera</h1>
        <div class="stream-container">
            <img src="/stream" alt="Live Stream" id="stream">
        </div>
        <div class="controls">
            <button onclick="fetch('/stream/start', {method: 'POST'})">Démarrer Stream</button>
            <button class="stop" onclick="fetch('/stream/stop', {method: 'POST'})">Arrêter Stream</button>
            <button onclick="captureAndRecognize()">Capturer et Reconnaître</button>
        </div>
        <div class="status" id="status"></div>
        
        <script>
            // Fonction pour capturer et afficher les résultats
            async function captureAndRecognize() {
                document.getElementById('status').innerText = 'Reconnaissance en cours...';
                try {
                    const response = await fetch('/capture');
                    const data = await response.json();
                    
                    if (data.error) {
                        document.getElementById('status').innerText = 'Erreur: ' + data.error;
                    } else {
                        let statusText = `Visages détectés: ${data.faces_detected}\\n`;
                        
                        if (data.results && data.results.length > 0) {
                            data.results.forEach(result => {
                                statusText += `Personne: ${result.name} (${result.access_granted ? 'Accès autorisé' : 'Accès refusé'})\\n`;
                            });
                        } else {
                            statusText += 'Aucun visage détecté';
                        }
                        
                        document.getElementById('status').innerText = statusText;
                    }
                } catch (error) {
                    document.getElementById('status').innerText = 'Erreur de connexion';
                }
            }
        </script>
    </body>
    </html>
    """
    return html

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
        # Arrêter le streaming si actif
        stream_active = False
    finally:
        security_system.cleanup()


