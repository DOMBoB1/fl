# Classroom Monitor

Sistem de monitorizare a atenției și a oboselii într-o sală de clasă. Proiectul
are două componente: backend-ul (FastAPI) în `backend/` și frontend-ul
(React + Vite) în `frontend/my-react-app/`.

## Rulare backend

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

## Rulare frontend

Necesită Node.js.

```bash
cd frontend/my-react-app
npm install
npm run dev
```
