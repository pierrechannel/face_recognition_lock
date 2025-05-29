import requests
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
                f"{self.base_url}/warehouse_access",
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
                f"{self.base_url}/warehouse_acces/create",
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