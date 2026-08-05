# Firmware ESP32-CAM — vue de face (Phase 3C)

Sketch Arduino qui fait tourner un ESP32-CAM AI-Thinker en serveur MJPEG HTTP,
consommé ensuite par `VideoReader` côté Python (flux réseau, Phase 3C).

## 1. Installer Arduino IDE + support ESP32

1. Installer [Arduino IDE](https://www.arduino.cc/en/software) (2.x).
2. Fichier > Préférences > "URL de gestionnaire de cartes supplémentaires" :
   `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
3. Outils > Type de carte > Gestionnaire de cartes > installer **esp32** (Espressif Systems).

## 2. Matériel (ESP32-CAM + base USB "ESP32-CAM-MB")

Avec le kit ESP32-CAM-MB, l'ESP32-CAM se clipse sur la base USB (puce CH340 + auto-reset) —
pas de câblage FTDI ni de manipulation manuelle de GPIO0 nécessaire.

1. Clipser l'ESP32-CAM sur la base MB (respecter le sens — la caméra dépasse du bord de la base).
2. Brancher la base en USB à l'ordinateur.
3. Si Windows ne détecte pas de port COM : installer le driver **CH340** (le kit ESP32-CAM-MB
   utilise généralement cette puce ; cherche "CH340 driver" pour Windows si besoin).

## 3. Configurer le WiFi

```bash
cp firmware/esp32cam_mjpeg/secrets.h.example firmware/esp32cam_mjpeg/secrets.h
```

Éditer `secrets.h` avec ton SSID/mot de passe (fichier gitignoré, jamais commité).

## 4. Flasher

1. Dans Arduino IDE :
   - Outils > Type de carte > **AI Thinker ESP32-CAM**
   - Outils > Partition Scheme > **Huge APP (3MB No OTA/1MB SPIFFS)**
   - Outils > Port > le port COM de la base ESP32-CAM-MB
2. Ouvrir `firmware/esp32cam_mjpeg/esp32cam_mjpeg.ino`, cliquer Téléverser.
   La base MB gère l'auto-reset/mode flash automatiquement — normalement pas besoin
   d'appuyer sur un bouton. Si le téléversement échoue avec un timeout ("Failed to connect"),
   essaie de maintenir le bouton **IO0** (ou **BOOT**) de la base enfoncé pendant le début
   du transfert, puis relâche une fois que "Writing..." apparaît.
3. Une fois le flash terminé, ouvrir le Moniteur série (115200 bauds) : la carte affiche
   son IP et l'URL du flux, ex. `http://192.168.1.42/stream`.
   Si rien ne s'affiche, appuyer sur le bouton **RESET** de la base pour relancer le sketch.

## 5. Vérifier le flux

Ouvrir l'URL affichée (`http://<ip>/stream`) dans un navigateur ou VLC (Média > Ouvrir un flux
réseau) — tu dois voir la vidéo en direct.

## Notes

- Résolution : VGA (640×480) si la carte a de la PSRAM (cas standard AI-Thinker), sinon QVGA (320×240).
- ~15-25 fps attendu (cf. CLAUDE.md), suffisant pour l'analyse plan frontal (genoux/coudes),
  pas besoin de la précision temporelle utilisée pour la vue profil.
- Le flux `/stream` reste ouvert tant que le client (Python `VideoReader`) est connecté.
- ⚠️ Le firmware ne gère **qu'un seul client à la fois** (`WiFiServer` basique) : si un onglet
  navigateur a encore `/stream` ouvert, l'ESP32 reste bloqué à lui servir des frames et
  n'accepte aucune autre connexion (Python se met alors à attendre indéfiniment). Fermer
  l'onglet/le lecteur précédent avant de lancer une capture Python.
