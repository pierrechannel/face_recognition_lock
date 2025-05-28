import pyttsx3
import threading
import logging
from collections import deque

class TTSManager:
    """Gestionnaire Text-to-Speech optimisé pour Raspberry Pi"""
    
    def __init__(self, voice_messages):
        self.is_active = False
        self.tts_queue = deque()
        self.tts_lock = threading.Lock()
        self.playback_thread = None
        self.tts_engine = None
        self.voice_messages = voice_messages
        
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
            
        if message_key in self.voice_messages:
            message = self.voice_messages[message_key]
            
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