import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const BACKEND = "http://localhost:8000";

function Stat({ label, value }) {
  return (
    <div className="stat">
      <div className="statLabel">{label}</div>
      <div className="statValue">{value}</div>
    </div>
  );
}

function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));
}

function iou(a, b) {
  const ax1 = a[0], ay1 = a[1], ax2 = a[2], ay2 = a[3];
  const bx1 = b[0], by1 = b[1], bx2 = b[2], by2 = b[3];

  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);

  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;

  if (inter <= 0) return 0;

  const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
  const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);

  return inter / (areaA + areaB - inter + 1e-9);
}

function normalizeBox(box) {
  if (!box || !Array.isArray(box) || box.length !== 4) return null;

  let [x1, y1, x2, y2] = box.map((v) => Number(v));

  if (![x1, y1, x2, y2].every(Number.isFinite)) return null;

  x1 = clamp(x1, 0, 1);
  y1 = clamp(y1, 0, 1);
  x2 = clamp(x2, 0, 1);
  y2 = clamp(y2, 0, 1);

  if (x2 <= x1 || y2 <= y1) return null;
  return [x1, y1, x2, y2];
}

function boxW(box) {
  return box ? Math.max(0, box[2] - box[0]) : 0;
}

function boxH(box) {
  return box ? Math.max(0, box[3] - box[1]) : 0;
}

function makeHeadBoxFromFace(faceBox) {
  const b = normalizeBox(faceBox);
  if (!b) return null;

  const [x1, y1, x2, y2] = b;
  const w = x2 - x1;
  const h = y2 - y1;

  const expandX = w * 0.30;
  const expandTop = h * 0.75;
  const expandBottom = h * 0.22;

  return normalizeBox([
    x1 - expandX,
    y1 - expandTop,
    x2 + expandX,
    y2 + expandBottom,
  ]);
}

function isLikelyFaceBox(box, score = 1) {
  const b = normalizeBox(box);
  if (!b) return false;

  const w = boxW(b);
  const h = boxH(b);
  const area = w * h;
  const ratio = w / Math.max(h, 1e-9);

  if (score < 0.35) return false;
  if (area < 0.0022) return false;
  if (w < 0.03 || h < 0.05) return false;
  if (ratio < 0.45 || ratio > 1.35) return false;

  const nearEdge =
    b[0] < 0.01 || b[1] < 0.01 || b[2] > 0.99 || b[3] > 0.99;

  if (nearEdge && area < 0.02) return false;

  return true;
}

function nmsFaces(faces, iouThresh = 0.55) {
  if (!faces || faces.length === 0) return [];

  const prepared = faces
    .map((f, index) => {
      const rawFace = normalizeBox(f.face_bbox_n || f.bbox_n);
      const score = Number(f.score ?? f.confidence ?? 1);

      if (!rawFace) return null;
      if (!isLikelyFaceBox(rawFace, score)) return null;

      return {
        ...f,
        _idx: index,
        _score: score,
        _face: rawFace,
        _area: boxW(rawFace) * boxH(rawFace),
      };
    })
    .filter(Boolean)
    .sort((a, b) => {
      if (b._score !== a._score) return b._score - a._score;
      return b._area - a._area;
    });

  const kept = [];
  for (const f of prepared) {
    let keep = true;
    for (const k of kept) {
      if (iou(f._face, k._face) >= iouThresh) {
        keep = false;
        break;
      }
    }
    if (keep) kept.push(f);
  }

  return kept;
}

function formatReportTime(ts) {
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function ReportItem({ report }) {
  if (!report) {
    return <div className="reportEmpty">-</div>;
  }

  return (
    <div className="reportItem">
      <div className="reportTop">
        <span className="reportTime">{formatReportTime(report.created_at)}</span>
        <span className="reportInterval">{report.interval_s}s</span>
      </div>

      <div className="reportLine">
        Avg faces: <strong>{Number(report.faces_avg ?? 0).toFixed(1)}</strong>
      </div>
      <div className="reportLine">
        Avg fatigue: <strong>{Number(report.fatigue_avg ?? 0).toFixed(1)}%</strong>
      </div>
      <div className="reportLine">
        Avg attention: <strong>{Number(report.attention_avg ?? 0).toFixed(1)}%</strong>
      </div>
      <div className="reportLine">
        Alerts: <strong>{Number(report.alert_count ?? 0)}</strong>
      </div>
    </div>
  );
}

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(null);
  const rafRef = useRef(null);
  const timerRef = useRef(null);
  const inFlightRef = useRef(false);

  const boxesOnRef = useRef(false);
  const showFaceRef = useRef(true);
  const showHeadRef = useRef(true);

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("idle");
  const [boxesOn, setBoxesOn] = useState(false);
  const [showFace, setShowFace] = useState(true);
  const [showHead, setShowHead] = useState(true);
  const [reportReady, setReportReady] = useState(false);

  const [faces, setFaces] = useState(0);
  const [fatigue, setFatigue] = useState(0);
  const [attention, setAttention] = useState(0);
  const [alertActive, setAlertActive] = useState(false);
  const [fps, setFps] = useState(0);
  const [recentReports, setRecentReports] = useState([]);

  const facesDataRef = useRef([]);
  const smoothFaceRef = useRef(new Map());
  const smoothHeadRef = useRef(new Map());

  const statusPill = useMemo(() => {
    if (status === "running") return "pill pillLive";
    if (status === "camera ready") return "pill pillReady";
    if (status === "camera error") return "pill pillErr";
    return "pill";
  }, [status]);

  useEffect(() => {
    boxesOnRef.current = boxesOn;
  }, [boxesOn]);

  useEffect(() => {
    showFaceRef.current = showFace;
  }, [showFace]);

  useEffect(() => {
    showHeadRef.current = showHead;
  }, [showHead]);

  useEffect(() => {
    let mounted = true;

    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
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
      timerRef.current = null;

      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;

      const v = videoRef.current;
      if (v?.srcObject) {
        const tracks = v.srcObject.getTracks?.() || [];
        tracks.forEach((t) => t.stop());
      }
    };
  }, []);

  function ensureOverlaySize() {
    const v = videoRef.current;
    const ov = overlayRef.current;
    if (!v || !ov) return;

    const rect = v.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));

    if (ov.width !== w || ov.height !== h) {
      ov.width = w;
      ov.height = h;
    }
  }

  function smoothBox(mapRef, id, rawBox, now) {
    let [x1, y1, x2, y2] = rawBox;
    const prev = mapRef.current.get(id);
    const alpha = 0.22;

    if (prev) {
      x1 = prev.x1 * (1 - alpha) + x1 * alpha;
      y1 = prev.y1 * (1 - alpha) + y1 * alpha;
      x2 = prev.x2 * (1 - alpha) + x2 * alpha;
      y2 = prev.y2 * (1 - alpha) + y2 * alpha;
    }

    mapRef.current.set(id, { x1, y1, x2, y2, t: now });
    return [x1, y1, x2, y2];
  }

  function cleanupOldBoxes(mapRef, now, maxAge = 1200) {
    for (const [id, st] of mapRef.current.entries()) {
      if (now - st.t > maxAge) {
        mapRef.current.delete(id);
      }
    }
  }

  function drawOneBox(ctx, ov, boxN, color, mapRef, id, now, label = "") {
    const box = normalizeBox(boxN);
    if (!box) return;

    let x1 = box[0] * ov.width;
    let y1 = box[1] * ov.height;
    let x2 = box[2] * ov.width;
    let y2 = box[3] * ov.height;

    x1 = clamp(x1, 0, ov.width - 1);
    y1 = clamp(y1, 0, ov.height - 1);
    x2 = clamp(x2, 0, ov.width - 1);
    y2 = clamp(y2, 0, ov.height - 1);

    [x1, y1, x2, y2] = smoothBox(mapRef, id, [x1, y1, x2, y2], now);

    const w = x2 - x1;
    const h = y2 - y1;
    if (w < 2 || h < 2) return;

    const dpr = window.devicePixelRatio || 1;

    ctx.lineWidth = Math.max(2, 2 * dpr);
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8 * dpr;
    ctx.strokeRect(x1, y1, w, h);
    ctx.shadowBlur = 0;

    if (label) {
      const padX = 8 * dpr;
      const padY = 5 * dpr;
      ctx.font = `${12 * dpr}px system-ui, sans-serif`;
      const tw = ctx.measureText(label).width;
      const bh = 22 * dpr;
      const by = Math.max(0, y1 - bh - 4 * dpr);

      ctx.fillStyle = color;
      ctx.fillRect(x1, by, tw + padX * 2, bh);

      ctx.fillStyle = "white";
      ctx.fillText(label, x1 + padX, by + 15 * dpr);
    }
  }

  function drawOverlay() {
    const ov = overlayRef.current;
    const v = videoRef.current;
    if (!ov || !v) {
      rafRef.current = requestAnimationFrame(drawOverlay);
      return;
    }

    ensureOverlaySize();
    const ctx = ov.getContext("2d");
    if (!ctx) {
      rafRef.current = requestAnimationFrame(drawOverlay);
      return;
    }

    ctx.clearRect(0, 0, ov.width, ov.height);

    if (boxesOnRef.current) {
      const facesData = facesDataRef.current || [];
      const filtered = nmsFaces(facesData, 0.55);
      const now = performance.now();

      for (const f of filtered) {
        const baseId = String(f.id ?? f.track_id ?? f._idx ?? Math.random());
        const faceBox = normalizeBox(f.face_bbox_n || f.bbox_n);
        const headBox = normalizeBox(f.head_bbox_n) || makeHeadBoxFromFace(faceBox);

        if (showHeadRef.current && headBox) {
          drawOneBox(
            ctx,
            ov,
            headBox,
            "rgba(255, 140, 60, 0.96)",
            smoothHeadRef,
            `head_${baseId}`,
            now,
            "HEAD"
          );
        }

        if (showFaceRef.current && faceBox) {
          drawOneBox(
            ctx,
            ov,
            faceBox,
            "rgba(38, 182, 222, 0.96)",
            smoothFaceRef,
            `face_${baseId}`,
            now,
            "FACE"
          );
        }
      }

      cleanupOldBoxes(smoothFaceRef, now);
      cleanupOldBoxes(smoothHeadRef, now);
    }

    rafRef.current = requestAnimationFrame(drawOverlay);
  }

  useEffect(() => {
    if (!rafRef.current) rafRef.current = requestAnimationFrame(drawOverlay);
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, []);

  async function sendFrameOnce() {
    const video = videoRef.current;
    const cap = captureRef.current;
    if (!video || !cap) return;
    if (inFlightRef.current) return;
    if (!video.videoWidth || !video.videoHeight) return;

    inFlightRef.current = true;

    try {
      const vw = video.videoWidth || 1280;
      const vh = video.videoHeight || 720;

      const targetW = 960;
      const targetH = Math.round((vh / vw) * targetW);

      cap.width = targetW;
      cap.height = targetH;

      const ctx = cap.getContext("2d", { willReadFrequently: false });
      if (!ctx) {
        inFlightRef.current = false;
        return;
      }

      ctx.drawImage(video, 0, 0, targetW, targetH);

      const blob = await new Promise((resolve) =>
        cap.toBlob(resolve, "image/jpeg", 0.82)
      );
      if (!blob) {
        inFlightRef.current = false;
        return;
      }

      const form = new FormData();
      form.append("image", blob, "frame.jpg");
      form.append("boxes", boxesOnRef.current ? "1" : "0");
      form.append("show_face", showFaceRef.current ? "1" : "0");
      form.append("show_head", showHeadRef.current ? "1" : "0");

      const res = await fetch(`${BACKEND}/analyze`, {
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

      facesDataRef.current = Array.isArray(s.faces_data) ? s.faces_data : [];
      setRecentReports(
        Array.isArray(s.recent_reports) ? s.recent_reports.slice(0, 3) : []
      );
    } catch (err) {
      console.error("Analyze error:", err);
    } finally {
      inFlightRef.current = false;
    }
  }

  async function start() {
    if (running) return;

    try {
      await fetch(`${BACKEND}/session/start`, { method: "POST" });
    } catch (err) {
      console.error("Session start error:", err);
    }

    setRunning(true);
    setStatus("running");
    setReportReady(false);

    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      sendFrameOnce();
    }, 260);
  }

  async function stop() {
    if (!running) return;

    setRunning(false);
    setStatus("stopped");

    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    inFlightRef.current = false;

    try {
      await fetch(`${BACKEND}/session/stop`, { method: "POST" });
      setReportReady(true);
    } catch (err) {
      console.error("Session stop error:", err);
      setReportReady(true);
    }
  }

  function toggleBoxes() {
    setBoxesOn((v) => !v);
    facesDataRef.current = [];
    smoothFaceRef.current.clear();
    smoothHeadRef.current.clear();
  }

  async function downloadReport() {
    try {
      const res = await fetch(`${BACKEND}/session/report`);

      if (!res.ok) {
        const err = await res.json().catch(() => null);
        alert(err?.error || err?.detail || "Could not generate report");
        return;
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;

      const disposition = res.headers.get("Content-Disposition");
      const match = disposition?.match(/filename="?([^"]+)"?/);
      a.download = match?.[1] || "session_report.xlsx";

      document.body.appendChild(a);
      a.click();
      a.remove();

      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Report download error:", err);
      alert("Report download failed");
    }
  }

  return (
    <div className="meet">
      <div className="top">
        <div className="stage">
          <div className="stageHeader">
            <div className="stageTitle">Class Monitor</div>
            <div className={statusPill}>{status === "running" ? "LIVE" : status}</div>
          </div>

          <div className="videoWrap">
            <video ref={videoRef} autoPlay playsInline muted className="video" />
            <canvas ref={overlayRef} className="overlayCanvas" />

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
              <div className="alertBanner">ALERT: Class fatigue is high</div>
            )}
          </div>

          <canvas ref={captureRef} className="hiddenCanvas" />
        </div>

        <div className="sidebar">
          <div className="card">
            <div className="cardTitle">Statistics</div>
            <Stat label="Faces" value={faces} />
            <Stat label="Fatigue Avg" value={`${fatigue}%`} />
            <Stat label="Attention Avg" value={`${attention}%`} />
            <Stat label="Alert" value={alertActive ? "ON" : "OFF"} />
            <Stat label="FPS" value={fps.toFixed(1)} />
          </div>

          <div className="card reportsCard">
            <div className="cardTitle">Info</div>
            <div className="reportsList">
              {recentReports.length > 0 ? (
                recentReports.map((report) => (
                  <ReportItem key={report.id} report={report} />
                ))
              ) : (
                <>
                  <ReportItem report={null} />
                  <ReportItem report={null} />
                  <ReportItem report={null} />
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="bottomBar">
        <div className="bottomLeft">
          <div className="hint">
            Backend: <span className="mono">{BACKEND}</span>
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

          <button
            className={`btn btnToggle ${boxesOn ? "on" : "off"}`}
            onClick={toggleBoxes}
            disabled={status === "camera error"}
            title="Desenează bbox-uri pe overlay"
          >
            Boxes: {boxesOn ? "ON" : "OFF"}
          </button>

          <div className={`boxOptions ${boxesOn ? "enabled" : "disabled"}`}>
            <label className={`boxCheck ${showFace ? "active" : ""}`}>
              <input
                type="checkbox"
                checked={showFace}
                onChange={(e) => setShowFace(e.target.checked)}
                disabled={!boxesOn}
              />
              <span>Face</span>
            </label>

            <label className={`boxCheck ${showHead ? "active" : ""}`}>
              <input
                type="checkbox"
                checked={showHead}
                onChange={(e) => setShowHead(e.target.checked)}
                disabled={!boxesOn}
              />
              <span>Head</span>
            </label>
          </div>

          <button
            className="btn btnReport"
            onClick={downloadReport}
            disabled={!reportReady || running}
            title="Exportă raportul sesiunii curente în Excel"
          >
            Report
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