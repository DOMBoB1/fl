# Classroom Monitor

Sistem de monitorizare a atenției și a oboselii într-o sală de clasă. Aplicația
analizează în timp real cadrele video de la o cameră, detectează și urmărește
fețele/capetele studenților, estimează nivelul de atenție și de oboseală pentru
fiecare student și pentru întreaga clasă, generează alerte și exportă rapoarte
Excel la finalul unei sesiuni.

Proiectul are două componente independente:

- **`backend/`** — API REST (FastAPI) cu motorul de viziune computerizată.
- **`frontend/my-react-app/`** — interfața web (React + Vite) care capturează
  imaginea de la cameră și afișează statisticile.

---

## Arhitectură

```
Cameră (browser)
      │  cadre JPEG
      ▼
Frontend React  ──HTTP──►  Backend FastAPI  ──►  Motor de analiză
 (dashboard)                 (server.py)          (engine + module CV)
      ▲                          │
      └────── statistici/JSON ───┘
                                 │
                                 ▼
                         SQLite + rapoarte .xlsx
```

Frontend-ul capturează periodic cadre de la `navigator.mediaDevices` și le
trimite la endpoint-ul `/analyze`. Backend-ul rulează detecția (YOLO),
identificarea fețelor (FaceNet), estimarea privirii și a oboselii, apoi
returnează statistici agregate pe care interfața le afișează sub formă de
indicatori, grafice și alerte.

---

## Componenta backend

Tehnologii principale: FastAPI, Ultralytics YOLO, MediaPipe, keras-facenet /
TensorFlow, OpenCV, openpyxl, SQLite.

Module relevante:

| Fișier | Rol |
| --- | --- |
| `server.py` | Definește API-ul FastAPI și endpoint-urile HTTP. |
| `engine.py`, `engine_core.py` | Motorul principal de monitorizare și logica unei sesiuni. |
| `config.py` | Toți parametrii de configurare (praguri, dimensiuni, alerte). |
| `attention.py` | Estimarea atenției pe baza direcției privirii (gaze). |
| `fatigue.py` | Calculul oboselii (PERCLOS, clipiri, ochi închiși). |
| `decision_rules.py` | Reguli care transformă metricile în decizii și mesaje. |
| `tracker_flow.py` | Urmărirea (tracking) fețelor între cadre cu optical flow. |
| `box_logic.py` | Validarea și prelucrarea bounding box-urilor. |
| `face_identity.py` | Identificarea studenților prin embedding-uri faciale. |
| `multi_face_detector.py`, `head_detector.py` | Detecția fețelor și a capetelor. |
| `session_stats.py` | Agregarea statisticilor pe sesiune și pe student. |
| `db.py`, `dataset_store.py` | Persistența în SQLite. |
| `raport_manager.py` | Generarea rapoartelor Excel. |
| `evaluate_detection*.py`, `prepare_yolo_dataset.py` | Utilitare de evaluare și pregătire a setului de date YOLO. |

### Endpoint-uri API

| Metodă | Rută | Descriere |
| --- | --- | --- |
| `GET` | `/health` | Verificare stare server. |
| `POST` | `/analyze` | Primește un cadru (multipart `image`/`file`/`frame`) și returnează statistici. |
| `POST` | `/session/start` | Pornește o sesiune de monitorizare. |
| `POST` | `/session/stop` | Oprește sesiunea și returnează sumarul. |
| `GET` | `/session/summary` | Sumarul sesiunii curente sau încheiate. |
| `GET` | `/session/report` | Exportă raportul sesiunii ca fișier `.xlsx`. |

### Rulare backend

Necesită Python 3.11.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

Alternativ, cu Docker:

```bash
cd backend
docker build -t classroom-monitor .
docker run -p 8000:8000 classroom-monitor
```

Modelul de detecție este încărcat din calea configurată în `config.py`
(`YOLO_MODEL_PATH`). Pentru identificarea capetelor printr-un model găzduit se
pot seta variabilele de mediu `ROBOFLOW_API_KEY`, `ROBOFLOW_MODEL_ID` și
`ROBOFLOW_MODEL_VERSION`.

---

## Componenta frontend

Aplicație React 19 cu Vite. Capturează imaginea de la cameră, o trimite la
backend pentru analiză și afișează un dashboard cu indicatori de atenție și
oboseală, alerte pe student și pe clasă, plus opțiuni de export al raportului.

Backend-ul țintă este configurat în `src/App.jsx` prin constanta `BACKEND`
(implicit `http://localhost:8000`).

### Rulare frontend

Necesită Node.js.

```bash
cd frontend/my-react-app
npm install
npm run dev
```

Aplicația pornește implicit pe `http://localhost:5173`. Backend-ul acceptă
cereri de la orice port `localhost`/`127.0.0.1` (configurat prin CORS în
`server.py`).

### Comenzi disponibile

- `npm run dev` — server de dezvoltare.
- `npm run build` — build de producție.
- `npm run preview` — previzualizarea build-ului.
- `npm run lint` — verificare ESLint.

---

## Flux de utilizare

1. Pornește backend-ul, apoi frontend-ul.
2. Deschide aplicația în browser și permite accesul la cameră.
3. Pornește o sesiune (`/session/start`).
4. În timpul sesiunii sunt afișate atenția, oboseala și alertele în timp real.
5. Oprește sesiunea (`/session/stop`) și descarcă raportul Excel
   (`/session/report`).

---

## Note

- Datele sesiunilor sunt salvate local în SQLite, iar rapoartele în
  `backend/reports/`. Aceste fișiere sunt ignorate de Git (vezi `.gitignore`).
- Parametrii de detecție, pragurile de alertă și ferestrele temporale se
  ajustează din `backend/config.py`.
