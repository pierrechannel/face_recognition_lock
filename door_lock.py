import cv2
import face_recognition
import numpy as np
import pickle
import datetime
import os
import time
import logging
from threading import Thread, Lock, Timer
import schedule
from PIL import Image
from io import BytesIO
from backend_client import BackendAPIClient
from tts_manager import TTSManager
from config import (BACKEND_API_URL, BACKEND_HEADERS, VOICE_MESSAGES,
                   KNOWN_FACES_FILE, OFFLINE_LOGS_FILE, CAMERA_WIDTH,
                   CAMERA_HEIGHT, FACE_RECOGNITION_TOLERANCE, SYNC_INTERVAL)
from utils import image_to_base64

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("WARNING: RPi.GPIO not available. Door lock control will be simulated.")

class DoorLockController:
    def __init__(self, relay_pin=18, led_green_pin=16, led_red_pin=20, buzzer_pin=21):
        self.relay_pin = relay_pin
        self.led_green_pin = led_green_pin
        self.led_red_pin = led_red_pin
        self.buzzer_pin = buzzer_pin
        self.lock_timer = None
        self.is_door_open = False
        
        if GPIO_AVAILABLE:
            self.setup_gpio()
        else:
            print("GPIO simulation mode activated")

    def setup_gpio(self):
        """Initialize GPIO pins for door lock control"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup relay pin (door lock)
        GPIO.setup(self.relay_pin, GPIO.OUT)
        GPIO.output(self.relay_pin, GPIO.LOW)  # Door locked by default
        
        # Setup LED indicators
        GPIO.setup(self.led_green_pin, GPIO.OUT)
        GPIO.setup(self.led_red_pin, GPIO.OUT)
        
        # Setup buzzer
        GPIO.setup(self.buzzer_pin, GPIO.OUT)
        
        # Initial state: door locked, red LED on
        self.lock_door()

    def unlock_door(self, duration=5):
        """Unlock the door for a specified duration (seconds)"""
        if GPIO_AVAILABLE:
            GPIO.output(self.relay_pin, GPIO.HIGH)  # Unlock
            GPIO.output(self.led_green_pin, GPIO.HIGH)
            GPIO.output(self.led_red_pin, GPIO.LOW)
        
        self.is_door_open = True
        print(f"🔓 Door UNLOCKED for {duration} seconds")
        
        # Cancel any existing timer
        if self.lock_timer:
            self.lock_timer.cancel()
        
        # Set timer to automatically lock after duration
        self.lock_timer = Timer(duration, self.lock_door)
        self.lock_timer.start()
        
        # Success beep
        self.beep_success()

    def lock_door(self):
        """Lock the door"""
        if GPIO_AVAILABLE:
            GPIO.output(self.relay_pin, GPIO.LOW)   # Lock
            GPIO.output(self.led_green_pin, GPIO.LOW)
            GPIO.output(self.led_red_pin, GPIO.HIGH)
        
        self.is_door_open = False
        print("🔒 Door LOCKED")

    def beep_success(self):
        """Play success beep pattern"""
        if not GPIO_AVAILABLE:
            print("🔊 SUCCESS BEEP")
            return
        
        def beep_pattern():
            for _ in range(2):
                GPIO.output(self.buzzer_pin, GPIO.HIGH)
                time.sleep(0.1)
                GPIO.output(self.buzzer_pin, GPIO.LOW)
                time.sleep(0.1)
        
        Thread(target=beep_pattern, daemon=True).start()

    def beep_denied(self):
        """Play access denied beep pattern"""
        if not GPIO_AVAILABLE:
            print("🔊 ACCESS DENIED BEEP")
            return
        
        def beep_pattern():
            for _ in range(3):
                GPIO.output(self.buzzer_pin, GPIO.HIGH)
                time.sleep(0.3)
                GPIO.output(self.buzzer_pin, GPIO.LOW)
                time.sleep(0.2)
        
        Thread(target=beep_pattern, daemon=True).start()

    def beep_unknown(self):
        """Play unknown person beep pattern"""
        if not GPIO_AVAILABLE:
            print("🔊 UNKNOWN PERSON BEEP")
            return
        
        def beep_pattern():
            GPIO.output(self.buzzer_pin, GPIO.HIGH)
            time.sleep(1.0)
            GPIO.output(self.buzzer_pin, GPIO.LOW)
        
        Thread(target=beep_pattern, daemon=True).start()

    def cleanup(self):
        """Cleanup GPIO resources"""
        if self.lock_timer:
            self.lock_timer.cancel()
        
        if GPIO_AVAILABLE:
            self.lock_door()  # Ensure door is locked
            GPIO.cleanup()