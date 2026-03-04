import { useEffect, useRef, useState } from "react";
import "./App.css";

function Stat({ label, value }) {
  return (
    <div className="stat">
      <div className="statLabel">{label}</div>
      <div className="statValue">{value}</div>
    </div>
  );
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const timerRef = useRef(null);

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("idle");

  const [faces, setFaces] = useState(0);
  const [fatigue, setFatigue] = useState(0);
  const [attention, setAttention] = useState(0);
  const [alertActive, setAlertActive] = useState(false);
  const [fps, setFps] = useState(0);

  useEffect(() => {
    let mounted = true;

    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
        if (!mounted) return;
        if (videoRef.current) videoRef.current.srcObject = stream;
        setStatus("camera ready");
      } catch {
        setStatus("camera error");
      }
    })();

    return () => {
      mounted = false;
      if (timerRef.current) clearInterval(timerRef.current);

      const v = videoRef.current;
      if (v?.srcObject) {
        const tracks = v.srcObject.getTracks?.() || [];
        tracks.forEach((t) => t.stop());
      }
    };
  }, []);

  async function sendFrameOnce() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;

    const targetW = 960;
    const targetH = Math.round((vh / vw) * targetW);

    canvas.width = targetW;
    canvas.height = targetH;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, targetW, targetH);

    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.65)
    );
    if (!blob) return;

    const form = new FormData();
    form.append("image", blob, "frame.jpg");

    const res = await fetch("http://localhost:8000/analyze", {
      method: "POST",
      body: form,
    });

    const data = await res.json();
    const s = data?.stats || data;

    if (!s) return;

    setFaces(Number(s.faces ?? 0));
    setFatigue(Math.round(s.class_avg_fatigue_pct ?? 0));
    setAttention(Math.round(s.class_avg_attention_pct ?? 0));
    setAlertActive(Boolean(s.alert_active ?? false));
    setFps(Number(s.fps ?? 0));
  }

  function start() {
    if (running) return;
    setRunning(true);
    setStatus("running");

    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      sendFrameOnce().catch(() => {});
    }, 80); // ~12.5 fps (pune 100 pentru 10fps dacă e greu)
  }

  function stop() {
    if (!running) return;
    setRunning(false);
    setStatus("stopped");
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
  }

  const statusPill =
    status === "running"
      ? "pill pillLive"
      : status === "camera ready"
      ? "pill pillReady"
      : status === "camera error"
      ? "pill pillErr"
      : "pill";

  return (
    <div className="meet">
      <div className="top">
        <div className="stage">
          <div className="stageHeader">
            <div className="stageTitle">Classroom Monitor</div>
            <div className={statusPill}>
              {status === "running" ? "LIVE" : status}
            </div>
          </div>

          <div className="videoWrap">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="video"
            />
            <div className="overlayTopLeft">
              <div className="miniStat">
                <span className="miniKey">Faces</span>
                <span className="miniVal">{faces}</span>
              </div>
              <div className="miniStat">
                <span className="miniKey">Fatigue</span>
                <span className="miniVal">{fatigue}%</span>
              </div>
              <div className="miniStat">
                <span className="miniKey">Attention</span>
                <span className="miniVal">{attention}%</span>
              </div>
            </div>

            {alertActive && (
              <div className="alertBanner">
                ALERT: Class fatigue is high
              </div>
            )}
          </div>

          <canvas ref={canvasRef} className="hiddenCanvas" />
        </div>

        <div className="sidebar">
          <div className="card">
            <div className="cardTitle">Statistici</div>
            <Stat label="Fețe" value={faces} />
            <Stat label="Fatigue Avg" value={`${fatigue}%`} />
            <Stat label="Attention Avg" value={`${attention}%`} />
            <Stat label="Alertă" value={alertActive ? "ON" : "OFF"} />
            <Stat label="FPS" value={fps.toFixed(1)} />
          </div>

          <div className="card">
            <div className="cardTitle">Info</div>
            <div className="info">
              <div>• Start/Stop din bara de jos</div>
              <div>• Pentru fluiditate: 10–15 FPS + rezoluție redusă</div>
              <div>• Recomandat: WebSocket (după ce confirmăm că merge REST)</div>
            </div>
          </div>
        </div>
      </div>

      <div className="bottomBar">
        <div className="bottomLeft">
          <div className="hint">
            Backend: <span className="mono">http://localhost:8000</span>
          </div>
        </div>

        <div className="bottomCenter">
          <button
            className="btn btnPrimary"
            onClick={start}
            disabled={running || status === "camera error"}
          >
            Start
          </button>
          <button className="btn btnGhost" onClick={stop} disabled={!running}>
            Stop
          </button>
        </div>

        <div className="bottomRight">
          <div className="hint">
            status: <span className="mono">{status}</span>
          </div>
        </div>
      </div>
    </div>
  );
}