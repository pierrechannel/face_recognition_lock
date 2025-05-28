import cv2
import base64

def image_to_base64(image, logger):
    """Convertit une image OpenCV en base64"""
    try:
        # Compression JPEG pour réduire la taille
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, buffer = cv2.imencode('.jpg', image, encode_param)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Erreur conversion image: {e}")
        return None