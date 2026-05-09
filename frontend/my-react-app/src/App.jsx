import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const BACKEND = "http://localhost:8000";
const DEFAULT_SESSION_MINUTES = 90;
const HISTORY_LIMIT = 240;
const ALERT_BOX_HOLD_MS = 5 * 60 * 1000;

const DEFAULT_REPORT_CONFIG = {
  reportType: "teacher",
  chartsIndividual: true,
  chartsGroup: true,
  chartsAlerts: false,
};

function Stat({ label, value }) {
  return (
    <div className="stat">
      <div className="statLabel">{label}</div>
      <div className="statValue">{value}</div>
    </div>
  );
}

function InfoStat({ label, value, info }) {
  const dotRef = useRef(null);
  const [bubblePos, setBubblePos] = useState(null);

  function hideBubble() {
    setBubblePos(null);
    if (window.__activeInfoBubbleHide === hideBubble) {
      window.__activeInfoBubbleHide = null;
    }
  }

  function showBubble() {
    if (!dotRef.current) return;

    if (window.__activeInfoBubbleHide && window.__activeInfoBubbleHide !== hideBubble) {
      window.__activeInfoBubbleHide();
    }

    const rect = dotRef.current.getBoundingClientRect();
    const top = Math.min(rect.bottom + 10, window.innerHeight - 90);
    const left = Math.max(16, rect.left - 12);

    window.__activeInfoBubbleHide = hideBubble;
    setBubblePos({ top, left });
  }

  useEffect(() => {
    return () => {
      if (window.__activeInfoBubbleHide === hideBubble) {
        window.__activeInfoBubbleHide = null;
      }
    };
  }, []);

  return (
    <div className="stat statInfo">
      <div className="statLabel statLabelInfo">
        <span>{label}</span>
        <span ref={dotRef} className="infoDot" onPointerEnter={showBubble} onPointerLeave={hideBubble}>
          i
          {bubblePos ? (
            <span
              className="infoBubble"
              style={{ top: `${bubblePos.top}px`, left: `${bubblePos.left}px` }}
            >
              {info}
            </span>
          ) : null}
        </span>
      </div>
      <div className="statValue">{value}</div>
    </div>
  );
}

function SectionCard({ title, open, onToggle, children, bodyClassName = "", rightSlot = null, className = "" }) {
  return (
    <div className={`card collapsibleCard ${open ? "open" : "closed"} ${className}`}>
      <button className="cardHeaderBtn" onClick={onToggle} type="button">
        <div className="cardHeaderLeft">
          <span className={`cardChevron ${open ? "open" : ""}`}>⌄</span>
          <div className="cardTitle">{title}</div>
        </div>
        {rightSlot ? <div className="cardHeaderRight">{rightSlot}</div> : null}
      </button>
      <div className={`cardBody ${open ? "open" : "closed"} ${bodyClassName}`}>{children}</div>
    </div>
  );
}

function SubSection({ title, open, onToggle, children, className = "" }) {
  return (
    <div className={`statSubcard ${className}`}>
      <button className="statSubheader" onClick={onToggle} type="button">
        <span>{title}</span>
        <span className={`statSubchevron ${open ? "open" : ""}`}>⌄</span>
      </button>
      {open ? <div className="statSubbody">{children}</div> : null}
    </div>
  );
}

function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));
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

function makeAlertBoxFromFaceAndHead(faceBox, headBox) {
  const face = normalizeBox(faceBox);
  const head = normalizeBox(headBox);

  if (face && head) {
    const fcx = (face[0] + face[2]) / 2;
    const fcy = (face[1] + face[3]) / 2;
    const fw = face[2] - face[0];
    const fh = face[3] - face[1];
    const hw = head[2] - head[0];
    const hh = head[3] - head[1];

    const targetW = Math.min(hw * 0.62, Math.max(fw * 1.28, fw + Math.max(0, hw - fw) * 0.24));
    const targetH = Math.min(hh * 0.62, Math.max(fh * 1.28, fh + Math.max(0, hh - fh) * 0.24));

    let x1 = fcx - targetW / 2;
    let y1 = fcy - targetH / 2;
    let x2 = fcx + targetW / 2;
    let y2 = fcy + targetH / 2;

    if (x1 < head[0]) {
      x2 += head[0] - x1;
      x1 = head[0];
    }
    if (x2 > head[2]) {
      x1 -= x2 - head[2];
      x2 = head[2];
    }
    if (y1 < head[1]) {
      y2 += head[1] - y1;
      y1 = head[1];
    }
    if (y2 > head[3]) {
      y1 -= y2 - head[3];
      y2 = head[3];
    }

    return normalizeBox([x1, y1, x2, y2]);
  }

  if (face) {
    const cx = (face[0] + face[2]) / 2;
    const cy = (face[1] + face[3]) / 2;
    const w = (face[2] - face[0]) * 1.28;
    const h = (face[3] - face[1]) * 1.28;
    return normalizeBox([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
  }

  if (head) {
    const cx = (head[0] + head[2]) / 2;
    const cy = (head[1] + head[3]) / 2;
    const w = (head[2] - head[0]) * 0.62;
    const h = (head[3] - head[1]) * 0.62;
    return normalizeBox([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
  }

  return null;
}

function makeAlertBoxFromFaceData(item) {
  if (!item || typeof item !== "object") return null;
  return makeAlertBoxFromFaceAndHead(item.face_bbox_n || item.bbox_n, item.head_bbox_n);
}

function formatReportTime(ts) {
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatDuration(totalSeconds) {
  const s = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;

  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  return `${m}:${String(sec).padStart(2, "0")}`;
}

function pluralizeStudent(count) {
  return `${count} student${count === 1 ? "" : "s"}`;
}

function formatLongDuration(totalSeconds) {
  const s = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);

  if (h > 0) return `${h} hour${h === 1 ? "" : "s"} and ${m} minute${m === 1 ? "" : "s"}`;
  return `${m} minute${m === 1 ? "" : "s"}`;
}

function getCameraLabel(status) {
  if (status === "camera ready" || status === "running" || status === "stopped" || status === "starting") return "Camera: ready";
  if (status === "camera error") return "Camera: not ready";
  return "Camera: checking";
}

function getCameraDetails(status) {
  if (status === "camera ready" || status === "running" || status === "stopped" || status === "starting") return "Camera is ready to analyze the next session.";
  if (status === "camera error") return "Camera cannot be found. Check permissions or connect an external camera.";
  return "Camera status is being checked.";
}

function ReportItem({ report }) {
  if (!report) return <div className="reportEmpty">No report data yet</div>;

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
    </div>
  );
}

function buildGroupDecision({ heads, fatigue, attention, fatigueAlertActive, attentionAlertActive }) {
  const suggestions = [];
  const observations = [];

  if (fatigueAlertActive && attentionAlertActive) {
    observations.push(`The group shows high fatigue (${fatigue}%) and low attention (${attention}%).`);
    suggestions.push("Use a short pause or a quick activity change.");
    suggestions.push("Ask a simple recap question to bring the class back into the lesson.");
  } else if (fatigueAlertActive) {
    observations.push(`The group fatigue level is high (${fatigue}%).`);
    suggestions.push("Give students a short pause, stretch moment, or easier transition task.");
  } else if (attentionAlertActive) {
    observations.push(`The group attention level is low (${attention}%).`);
    suggestions.push("Change the rhythm of the lesson and ask a direct question to the group.");
  } else {
    observations.push(`Group state is stable. Detected people: ${heads}, average fatigue: ${fatigue}%, average attention: ${attention}%.`);
    suggestions.push("Continue the current teaching flow and monitor the next changes.");
  }

  return { summary: observations.join(" "), suggestions };
}

function buildStudentDecision({ studentAlerts, thresholds }) {
  const suggestions = [];
  const observations = [];

  if (!Array.isArray(studentAlerts) || studentAlerts.length === 0) {
    observations.push("No student-specific alert is active.");
    suggestions.push("Keep monitoring student-level alerts during the session.");
    return { summary: observations.join(" "), suggestions };
  }

  const critical = studentAlerts.filter((s) => String(s.severity || "").toLowerCase() === "critical");
  const fatigueOnly = studentAlerts.filter((s) => Number(s.fatigue_pct ?? 0) >= thresholds.student_fatigue_on);
  const attentionOnly = studentAlerts.filter((s) => Number(s.attention_pct ?? 100) <= thresholds.student_attention_on);

  observations.push(`${pluralizeStudent(studentAlerts.length)} currently trigger student-level alerts.`);

  if (critical.length > 0) suggestions.push(`Prioritize ${critical.length === 1 ? "the critical student alert" : "the critical student alerts"}.`);
  if (fatigueOnly.length > 0) suggestions.push("For fatigue alerts, use a short pause or ask a low-pressure question.");
  if (attentionOnly.length > 0) suggestions.push("For attention alerts, use a direct question or move closer to the affected area.");

  return { summary: observations.join(" "), suggestions: [...new Set(suggestions)] };
}

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(null);
  const rafRef = useRef(null);
  const timerRef = useRef(null);
  const countdownRef = useRef(null);
  const inFlightRef = useRef(false);
  const audioRef = useRef(null);
  const boxesOnRef = useRef(false);
  const showFaceRef = useRef(true);
  const showHeadRef = useRef(true);
  const alertOverlayEnabledRef = useRef(true);
  const prevAnyAlertRef = useRef(false);
  const sessionEndRef = useRef(null);
  const smoothFaceRef = useRef(new Map());
  const smoothHeadRef = useRef(new Map());
  const facesDataRef = useRef([]);
  const smoothAlertHeadRef = useRef(new Map());
  const alertOverlayMemoryRef = useRef(new Map());
  const alertOverlayNextIndexRef = useRef(1);

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("idle");
  const [boxesOn, setBoxesOn] = useState(false);
  const [showFace, setShowFace] = useState(true);
  const [showHead, setShowHead] = useState(true);
  const [boxesOptionsOpen, setBoxesOptionsOpen] = useState(false);
  const [developerMode, setDeveloperMode] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [groupStatsOpen, setGroupStatsOpen] = useState(true);
  const [individualStatsOpen, setIndividualStatsOpen] = useState(true);
  const [developerStatsOpen, setDeveloperStatsOpen] = useState(true);
  const [groupDecisionOpen, setGroupDecisionOpen] = useState(true);
  const [individualDecisionOpen, setIndividualDecisionOpen] = useState(true);

  const [settings, setSettings] = useState({
    sessionMinutes: DEFAULT_SESSION_MINUTES,
    soundAlerts: true,
    showOverlayInfo: true,
    showStatisticsPanel: true,
    showAlertPanel: true,
    showDecisionPanel: true,
    showInfoPanel: true,
    alertStudentOverlay: true,
    reportEnabled: true,
    reportPreview: true,
    reportType: DEFAULT_REPORT_CONFIG.reportType,
    chartsIndividual: DEFAULT_REPORT_CONFIG.chartsIndividual,
    chartsGroup: DEFAULT_REPORT_CONFIG.chartsGroup,
    chartsAlerts: DEFAULT_REPORT_CONFIG.chartsAlerts,
  });

  const [raportReady, setRaportReady] = useState(false);
  const [statsOpen, setStatsOpen] = useState(true);
  const [alertsOpen, setAlertsOpen] = useState(true);
  const [decisionOpen, setDecisionOpen] = useState(true);
  const [infoOpen, setInfoOpen] = useState(false);
  const [faces, setFaces] = useState(0);
  const [heads, setHeads] = useState(0);
  const [fatigue, setFatigue] = useState(0);
  const [attention, setAttention] = useState(0);
  const [alertActive, setAlertActive] = useState(false);
  const [fatigueAlertActive, setFatigueAlertActive] = useState(false);
  const [attentionAlertActive, setAttentionAlertActive] = useState(false);
  const [studentAlerts, setStudentAlerts] = useState([]);
  const [pinnedAlertStudents, setPinnedAlertStudents] = useState([]);
  const [fps, setFps] = useState(0);
  const [recentReports, setRecentReports] = useState([]);
  const [history, setHistory] = useState([]);
  const [remainingSeconds, setRemainingSeconds] = useState(DEFAULT_SESSION_MINUTES * 60);

  const [thresholds, setThresholds] = useState({
    class_fatigue_on: 50,
    class_fatigue_off: 45,
    class_attention_on: 50,
    class_attention_off: 55,
    student_fatigue_on: 60,
    student_fatigue_off: 52,
    student_attention_on: 50,
    student_attention_off: 55,
    student_fatigue_critical: 70,
    student_attention_critical: 30,
    min_active_students_for_class_alert: 1,
  });

  const configuredSessionSeconds = Math.max(1, Number(settings.sessionMinutes) || DEFAULT_SESSION_MINUTES) * 60;

  const alertMessage = useMemo(() => {
    if (fatigueAlertActive && attentionAlertActive) return `High fatigue (${fatigue}%) and low attention (${attention}%)`;
    if (fatigueAlertActive) return `High fatigue: ${fatigue}%`;
    if (attentionAlertActive) return `Low attention: ${attention}%`;
    if (studentAlerts.length > 0) return `${pluralizeStudent(studentAlerts.length)} need${studentAlerts.length === 1 ? "s" : ""} attention`;
    return "No active alert";
  }, [fatigueAlertActive, attentionAlertActive, studentAlerts, fatigue, attention]);

  const groupDecision = useMemo(
    () => buildGroupDecision({ heads, fatigue, attention, fatigueAlertActive, attentionAlertActive }),
    [heads, fatigue, attention, fatigueAlertActive, attentionAlertActive]
  );

  const studentDecision = useMemo(
    () => buildStudentDecision({ studentAlerts, thresholds }),
    [studentAlerts, thresholds]
  );

  function resetUiForNewSession() {
    setFaces(0);
    setHeads(0);
    setFatigue(0);
    setAttention(0);
    setAlertActive(false);
    setFatigueAlertActive(false);
    setAttentionAlertActive(false);
    setStudentAlerts([]);
    setPinnedAlertStudents([]);
    setFps(0);
    setRecentReports([]);
    setRaportReady(false);
    setHistory([]);
    setRemainingSeconds(Math.max(1, Number(settings.sessionMinutes) || DEFAULT_SESSION_MINUTES) * 60);

    facesDataRef.current = [];
    smoothFaceRef.current.clear();
    smoothHeadRef.current.clear();
    smoothAlertHeadRef.current.clear();
    alertOverlayMemoryRef.current.clear();
    alertOverlayNextIndexRef.current = 1;
    prevAnyAlertRef.current = false;
  }

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
    alertOverlayEnabledRef.current = Boolean(settings.alertStudentOverlay);
    if (!settings.alertStudentOverlay) {
      alertOverlayMemoryRef.current.clear();
      smoothAlertHeadRef.current.clear();
      setPinnedAlertStudents([]);
    }
  }, [settings.alertStudentOverlay]);

  useEffect(() => {
    if (!running) setRemainingSeconds(configuredSessionSeconds);
  }, [configuredSessionSeconds, running]);

  useEffect(() => {
    setSettings((prev) => ({
      ...prev,
      reportType: developerMode ? "developer" : "teacher",
      chartsIndividual: developerMode ? true : DEFAULT_REPORT_CONFIG.chartsIndividual,
      chartsGroup: developerMode ? true : DEFAULT_REPORT_CONFIG.chartsGroup,
      chartsAlerts: developerMode ? true : DEFAULT_REPORT_CONFIG.chartsAlerts,
    }));
  }, [developerMode]);

  useEffect(() => {
    const anyNow =
      Boolean(fatigueAlertActive) ||
      Boolean(attentionAlertActive) ||
      studentAlerts.some((s) => String(s.severity || "").toLowerCase() === "critical");

    if (settings.soundAlerts && !prevAnyAlertRef.current && anyNow && audioRef.current) {
      try {
        audioRef.current.currentTime = 0;
        const p = audioRef.current.play();
        if (p && typeof p.catch === "function") p.catch(() => {});
      } catch (_err) {}
    }

    prevAnyAlertRef.current = anyNow;
  }, [fatigueAlertActive, attentionAlertActive, studentAlerts, settings.soundAlerts]);

  useEffect(() => {
    let mounted = true;

    async function initCamera() {
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
    }

    initCamera();

    return () => {
      mounted = false;

      if (timerRef.current) clearInterval(timerRef.current);
      if (countdownRef.current) clearInterval(countdownRef.current);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);

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
      if (now - st.t > maxAge) mapRef.current.delete(id);
    }
  }

  function drawOneBox(ctx, ov, boxN, color, mapRef, id, now, label = "", dashed = false) {
    const box = normalizeBox(boxN);
    if (!box) return;

    let x1 = clamp(box[0] * ov.width, 0, ov.width - 1);
    let y1 = clamp(box[1] * ov.height, 0, ov.height - 1);
    let x2 = clamp(box[2] * ov.width, 0, ov.width - 1);
    let y2 = clamp(box[3] * ov.height, 0, ov.height - 1);

    [x1, y1, x2, y2] = smoothBox(mapRef, id, [x1, y1, x2, y2], now);

    const w = x2 - x1;
    const h = y2 - y1;
    if (w < 2 || h < 2) return;

    const dpr = window.devicePixelRatio || 1;

    ctx.lineWidth = Math.max(2, 2 * dpr);
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8 * dpr;
    ctx.setLineDash(dashed ? [10 * dpr, 6 * dpr] : []);
    ctx.strokeRect(x1, y1, w, h);
    ctx.shadowBlur = 0;
    ctx.setLineDash([]);

    if (label) {
      const padX = 8 * dpr;
      const bh = 22 * dpr;

      ctx.font = `${12 * dpr}px system-ui, sans-serif`;
      const tw = ctx.measureText(label).width;
      const by = Math.max(0, y1 - bh - 4 * dpr);

      ctx.fillStyle = color;
      ctx.fillRect(x1, by, tw + padX * 2, bh);
      ctx.fillStyle = "white";
      ctx.fillText(label, x1 + padX, by + 15 * dpr);
    }
  }

  function getIdentityKeys(item) {
    if (!item || typeof item !== "object") return [];

    return [item.person_id, item.student_id, item.track_id, item.id, item.face_id, item.head_id]
      .filter((v) => v !== undefined && v !== null && v !== "")
      .map((v) => String(v));
  }

  function boxCenter(box) {
    const b = normalizeBox(box);
    if (!b) return null;
    return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
  }

  function boxDistance(a, b) {
    const ca = boxCenter(a);
    const cb = boxCenter(b);
    if (!ca || !cb) return Number.POSITIVE_INFINITY;
    return Math.hypot(ca[0] - cb[0], ca[1] - cb[1]);
  }

  function bestLiveBoxForMemory(memoryItem, facesData) {
    if (!memoryItem || !Array.isArray(facesData) || facesData.length === 0) return null;

    const memoryKeys = new Set([String(memoryItem.key), String(memoryItem.studentId)].filter(Boolean));

    for (const f of facesData) {
      const keys = getIdentityKeys(f);
      if (keys.some((key) => memoryKeys.has(String(key)))) {
        const box = makeAlertBoxFromFaceData(f);
        if (box) return box;
      }
    }

    let bestBox = null;
    let bestDist = Number.POSITIVE_INFINITY;

    for (const f of facesData) {
      const candidate = makeAlertBoxFromFaceData(f);
      if (!candidate) continue;

      const dist = boxDistance(memoryItem.box, candidate);
      if (dist < bestDist) {
        bestDist = dist;
        bestBox = candidate;
      }
    }

    return bestDist <= 0.18 ? bestBox : null;
  }

  function buildAlertHeadBoxes(facesData, alerts) {
    const result = [];
    const used = new Set();
    const byId = new Map();

    for (const f of Array.isArray(facesData) ? facesData : []) {
      const keys = getIdentityKeys(f);
      for (const key of keys) byId.set(key, f);
    }

    for (const alert of Array.isArray(alerts) ? alerts : []) {
      const alertBox = makeAlertBoxFromFaceData(alert);
      const alertKeys = getIdentityKeys(alert);
      let matched = null;

      for (const key of alertKeys) {
        if (byId.has(key)) {
          matched = byId.get(key);
          break;
        }
      }

      const box = alertBox || makeAlertBoxFromFaceData(matched);
      const key = alertKeys[0] || getIdentityKeys(matched || {})[0] || `alert_${result.length}`;

      if (box && !used.has(key)) {
        used.add(key);
        result.push({
          key,
          box,
          severity: String(alert.severity || "warning").toLowerCase(),
        });
      }
    }

    return result;
  }

  function refreshAlertOverlayMemory(facesData, alerts) {
    const now = Date.now();
    const memory = alertOverlayMemoryRef.current;

    for (const [key, item] of memory.entries()) {
      if (Number(item.expiresAt || 0) <= now) memory.delete(key);
    }

    const current = buildAlertHeadBoxes(facesData, alerts);
    const byKey = new Map(current.map((item) => [String(item.key), item]));

    for (const alert of Array.isArray(alerts) ? alerts : []) {
      const keys = getIdentityKeys(alert);
      const key = String(keys[0] || `alert_${alert.student_id ?? alert.id ?? memory.size}`);
      const matched = byKey.get(key) || current.find((item) => keys.includes(String(item.key)));

      if (!matched?.box) continue;

      const previous = memory.get(key);
      const alertIndex = Number(previous?.alertIndex || 0) > 0 ? Number(previous.alertIndex) : alertOverlayNextIndexRef.current++;

      memory.set(key, {
        key,
        alertIndex,
        box: matched.box,
        severity: String(alert.severity || matched.severity || "warning").toLowerCase(),
        studentId: String(alert.student_id ?? alert.person_id ?? alert.id ?? key),
        fatiguePct: Math.round(Number(alert.fatigue_pct ?? 0)),
        attentionPct: Math.round(Number(alert.attention_pct ?? 0)),
        message: String(alert.message || alert.reason || "Student requires attention."),
        updatedAt: now,
        expiresAt: now + ALERT_BOX_HOLD_MS,
      });
    }

    return getAlertOverlayItems();
  }

  function getAlertOverlayItems() {
    const now = Date.now();
    const memory = alertOverlayMemoryRef.current;
    const facesData = Array.isArray(facesDataRef.current) ? facesDataRef.current : [];
    const items = [];

    for (const [key, item] of memory.entries()) {
      if (Number(item.expiresAt || 0) <= now) {
        memory.delete(key);
        continue;
      }

      const liveBox = bestLiveBoxForMemory(item, facesData);

      if (liveBox) {
        item.box = liveBox;
        item.lastSeenAt = now;
      }

      items.push({
        ...item,
        key,
        active: now - Number(item.updatedAt || 0) < 3500,
        tracking: Boolean(liveBox),
        secondsLeft: Math.max(0, Math.ceil((Number(item.expiresAt || now) - now) / 1000)),
      });
    }

    return items.sort((a, b) => {
      const aw = a.severity === "critical" ? 1 : 0;
      const bw = b.severity === "critical" ? 1 : 0;
      if (aw !== bw) return bw - aw;
      return Number(b.updatedAt || 0) - Number(a.updatedAt || 0);
    });
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

    const facesData = Array.isArray(facesDataRef.current) ? facesDataRef.current : [];
    const now = performance.now();

    if (boxesOnRef.current) {
      for (const f of facesData) {
        const baseId = String(f.id ?? f.track_id ?? Math.random());
        const faceBox = normalizeBox(f.face_bbox_n || f.bbox_n);
        const headBox = normalizeBox(f.head_bbox_n);
        const faceKind = String(f.face_kind || "face").toLowerCase();
        const headInferred = Boolean(f.head_inferred);

        if (showHeadRef.current && headBox) {
          drawOneBox(
            ctx,
            ov,
            headBox,
            headInferred ? "rgba(255, 170, 90, 0.96)" : "rgba(255, 140, 60, 0.96)",
            smoothHeadRef,
            `head_${baseId}`,
            now,
            headInferred ? "HEAD*" : "HEAD",
            headInferred
          );
        }

        if (showFaceRef.current && faceBox) {
          const side = faceKind === "side_face" || faceKind === "sideface";

          drawOneBox(
            ctx,
            ov,
            faceBox,
            side ? "rgba(182, 70, 255, 0.96)" : "rgba(38, 182, 222, 0.96)",
            smoothFaceRef,
            `face_${baseId}`,
            now,
            side ? "SIDE FACE" : "FACE"
          );
        }
      }
    }

    if (alertOverlayEnabledRef.current) {
      for (const item of getAlertOverlayItems()) {
        const label = `ALERT ${item.alertIndex || 1}`;

        drawOneBox(
          ctx,
          ov,
          item.box,
          item.severity === "critical" ? "rgba(255, 35, 60, 1)" : "rgba(255, 82, 95, 0.98)",
          smoothAlertHeadRef,
          `alert_head_${item.key}`,
          now,
          label
        );
      }
    }

    cleanupOldBoxes(smoothFaceRef, now);
    cleanupOldBoxes(smoothHeadRef, now);
    cleanupOldBoxes(smoothAlertHeadRef, now);

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

    if (!video || !cap || inFlightRef.current || !video.videoWidth || !video.videoHeight) return;

    inFlightRef.current = true;

    try {
      const vw = video.videoWidth || 1280;
      const vh = video.videoHeight || 720;
      const targetW = 960;
      const targetH = Math.round((vh / vw) * targetW);

      cap.width = targetW;
      cap.height = targetH;

      const ctx = cap.getContext("2d", { willReadFrequently: false });
      if (!ctx) return;

      ctx.clearRect(0, 0, targetW, targetH);
      ctx.drawImage(video, 0, 0, targetW, targetH);

      const blob = await new Promise((resolve) => cap.toBlob(resolve, "image/jpeg", 0.82));
      if (!blob) return;

      const form = new FormData();
      form.append("image", blob, "frame.jpg");
      form.append("boxes", "1");
      form.append("show_face", showFaceRef.current ? "1" : "0");
      form.append("show_head", showHeadRef.current ? "1" : "0");

      const res = await fetch(`${BACKEND}/analyze`, {
        method: "POST",
        body: form,
      });

      const data = await res.json().catch(() => null);

      if (!res.ok) {
        console.error("Analyze backend error:", data || res.statusText);
        return;
      }

      const s = data?.stats || data;
      if (!s) return;

      const nextFaces = Number(s.faces ?? 0);
      const nextHeads = Number(s.heads ?? 0);
      const nextFatigue = Math.round(Number(s.class_avg_fatigue_pct ?? 0));
      const nextAttention = Math.round(Number(s.class_avg_attention_pct ?? 0));
      const nextStudentAlerts = Array.isArray(s.student_alerts) ? s.student_alerts : [];
      const nextFatigueAlert = Boolean(s.fatigue_alert_active ?? false);
      const nextAttentionAlert = Boolean(s.attention_alert_active ?? false);
      const nextAnyAlert = Boolean(s.alert_active ?? (nextFatigueAlert || nextAttentionAlert || nextStudentAlerts.length > 0));      const nextFacesData = Array.isArray(s.faces_data) ? s.faces_data : [];

      facesDataRef.current = nextFacesData;

      const overlayAlerts =
        nextStudentAlerts.length > 0
          ? nextStudentAlerts
          : nextAnyAlert
            ? nextFacesData.map((face, idx) => ({
                id: face.id ?? face.track_id ?? idx + 1,
                student_id: face.id ?? face.track_id ?? idx + 1,
                person_id: face.person_id,
                track_id: face.track_id,
                face_id: face.face_id,
                head_id: face.head_id,
                head_bbox_n: face.head_bbox_n,
                face_bbox_n: face.face_bbox_n,
                bbox_n: face.bbox_n,
                severity: nextFatigueAlert && nextAttentionAlert ? "critical" : "warning",
                fatigue_pct: nextFatigue,
                attention_pct: nextAttention,
                message:
                  nextFatigueAlert && nextAttentionAlert
                    ? "Class fatigue and attention alert. Check this detected student."
                    : nextFatigueAlert
                      ? "High fatigue alert. Check this detected student."
                      : nextAttentionAlert
                        ? "Low attention alert. Check this detected student."
                        : "Alert active. Check this detected student.",
              }))
            : [];

      setFaces(nextFaces);
      setHeads(nextHeads);
      setFatigue(nextFatigue);
      setAttention(nextAttention);
      setAlertActive(nextAnyAlert);
      setFatigueAlertActive(nextFatigueAlert);
      setAttentionAlertActive(nextAttentionAlert);
      setStudentAlerts(nextStudentAlerts);
      setPinnedAlertStudents(settings.alertStudentOverlay ? refreshAlertOverlayMemory(nextFacesData, overlayAlerts) : []);
      setFps(Number(s.fps ?? 0));

      if (s.thresholds && typeof s.thresholds === "object") {
        setThresholds((prev) => ({ ...prev, ...s.thresholds }));
      }

      setRecentReports(Array.isArray(s.recent_reports) ? s.recent_reports.slice(0, 3) : []);

      setHistory((prev) =>
        [
          ...prev,
          {
            t: Date.now(),
            faces: nextFaces,
            heads: nextHeads,
            fatigue: nextFatigue,
            attention: nextAttention,
            alerts: nextStudentAlerts.length,
            classAlerts: Number(nextFatigueAlert) + Number(nextAttentionAlert),
          },
        ].slice(-HISTORY_LIMIT)
      );
    } catch (err) {
      console.error("Analyze error:", err);
    } finally {
      inFlightRef.current = false;
    }
  }

  function updateSetting(key, value) {
    setSettings((prev) => ({ ...prev, [key]: value }));
  }

  async function start() {
    if (running) return;

    resetUiForNewSession();
    setStatus("starting");

    try {
      const res = await fetch(`${BACKEND}/session/start`, { method: "POST" });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        console.error("Session start backend error:", data || res.statusText);
        setStatus("camera ready");
        return;
      }
    } catch (err) {
      console.error("Session start error:", err);
      setStatus("camera ready");
      return;
    }

    sessionEndRef.current = Date.now() + configuredSessionSeconds * 1000;

    setRemainingSeconds(configuredSessionSeconds);
    setRunning(true);
    setStatus("running");
    setRaportReady(false);

    if (timerRef.current) clearInterval(timerRef.current);
    if (countdownRef.current) clearInterval(countdownRef.current);

    timerRef.current = setInterval(sendFrameOnce, 260);

    countdownRef.current = setInterval(() => {
      const end = sessionEndRef.current || Date.now();
      const left = Math.max(0, Math.ceil((end - Date.now()) / 1000));

      setRemainingSeconds(left);

      if (left <= 0) {
        stop();
      }
    }, 1000);

    sendFrameOnce();
  }

  async function stop() {
    setRunning(false);
    setStatus("stopped");

    if (timerRef.current) clearInterval(timerRef.current);
    if (countdownRef.current) clearInterval(countdownRef.current);

    timerRef.current = null;
    countdownRef.current = null;

    try {
      const res = await fetch(`${BACKEND}/session/stop`, { method: "POST" });
      const data = await res.json().catch(() => null);

      if (!res.ok) console.error("Session stop backend error:", data || res.statusText);

      setRaportReady(true);
    } catch (err) {
      console.error("Session stop error:", err);
      setRaportReady(true);
    }
  }

  function toggleBoxes() {
    setBoxesOn((v) => !v);
    setBoxesOptionsOpen(true);

    facesDataRef.current = [];
    smoothFaceRef.current.clear();
    smoothHeadRef.current.clear();
    smoothAlertHeadRef.current.clear();
    alertOverlayMemoryRef.current.clear();
    alertOverlayNextIndexRef.current = 1;

    setPinnedAlertStudents([]);
  }

  async function downloadRaport() {
    try {
      const params = new URLSearchParams();

      params.set("report_type", settings.reportType || DEFAULT_REPORT_CONFIG.reportType);
      params.set("chart_individual", settings.chartsIndividual ? "1" : "0");
      params.set("chart_group", settings.chartsGroup ? "1" : "0");
      params.set("chart_alerts", settings.chartsAlerts ? "1" : "0");

      const res = await fetch(`${BACKEND}/session/report?${params.toString()}`);

      if (!res.ok) {
        const err = await res.json().catch(() => null);
        alert(err?.error || err?.detail || "Could not generate raport");
        return;
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");

      a.href = url;

      const disposition = res.headers.get("Content-Disposition");
      const match = disposition?.match(/filename="?([^"]+)"?/);

      a.download = match?.[1] || "raport_sesiune.xlsx";

      document.body.appendChild(a);
      a.click();
      a.remove();

      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Raport download error:", err);
      alert("Raport download failed");
    }
  }

  return (
    <div className="meet">
      <audio ref={audioRef} src="/sounds.mp3" preload="auto" />

      <div className="top">
        <div className="stage">
          <div className="stageHeader">
            <div className="stageTitleWrap">
              <div className="stageTitle">Class Monitor</div>
            </div>

            <div className="stageHeaderRight">
              <div className={`cameraStatus cameraStatusTop ${status === "camera error" ? "cameraStatusError" : "cameraStatusReady"}`}>
                {getCameraLabel(status)}
                <div className="timerBubble cameraBubble">{getCameraDetails(status)}</div>
              </div>
            </div>
          </div>

          <div className="videoWrap">
            <video ref={videoRef} autoPlay playsInline muted className="video" style={{ transform: "none" }} />
            <canvas ref={overlayRef} className="overlayCanvas" />

            {settings.showOverlayInfo && (
              <div className="overlayTopLeft">
                <div className="miniStat">
                  <span className="miniKey">People</span>
                  <span className="miniVal">{heads}</span>
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
            )}

            <div className={`alertBanner ${alertActive || studentAlerts.length > 0 ? "active" : "idle"}`}>{alertMessage}</div>
          </div>

          <canvas ref={captureRef} className="hiddenCanvas" />
        </div>

        <div className="sidebar">
          {settings.showStatisticsPanel && (
            <SectionCard title="Statistics" open={statsOpen} onToggle={() => setStatsOpen((v) => !v)}>
              <SubSection title="Group" open={groupStatsOpen} onToggle={() => setGroupStatsOpen((v) => !v)}>
                <Stat label="Detected people" value={heads} />
                <Stat label="Average fatigue" value={`${fatigue}%`} />
                <Stat label="Average attention" value={`${attention}%`} />
              </SubSection>

              <SubSection title="Individual" open={individualStatsOpen} onToggle={() => setIndividualStatsOpen((v) => !v)}>
                <InfoStat
                  label="Student alerts"
                  value={studentAlerts.length}
                  info="The number of active individual alerts. An alert appears when a student passes one of the attention or fatigue thresholds."
                />
                <InfoStat
                  label="Fatigue alerts"
                  value={studentAlerts.filter((s) => Number(s.fatigue_pct ?? 0) >= thresholds.student_fatigue_on).length}
                  info={`The number of students with fatigue alerts. It triggers when student fatigue is at least ${thresholds.student_fatigue_on}% and clears after fatigue drops below ${thresholds.student_fatigue_off}%.`}
                />
                <InfoStat
                  label="Attention alerts"
                  value={studentAlerts.filter((s) => Number(s.attention_pct ?? 100) <= thresholds.student_attention_on).length}
                  info={`The number of students with attention alerts. It triggers when student attention is at most ${thresholds.student_attention_on}% and clears after attention goes above ${thresholds.student_attention_off}%.`}
                />
                <InfoStat
                  label="Critical fatigue"
                  value={studentAlerts.filter((s) => Number(s.fatigue_pct ?? 0) >= thresholds.student_fatigue_critical).length}
                  info={`The number of students with critical fatigue. Critical fatigue starts at ${thresholds.student_fatigue_critical}% and means the student should be checked more urgently.`}
                />
                <InfoStat
                  label="Critical attention"
                  value={studentAlerts.filter((s) => Number(s.attention_pct ?? 100) <= thresholds.student_attention_critical).length}
                  info={`The number of students with critical attention. Critical attention starts at ${thresholds.student_attention_critical}% or lower and means the student is probably not following the session.`}
                />
              </SubSection>

              {developerMode && (
                <SubSection title="Developer" open={developerStatsOpen} onToggle={() => setDeveloperStatsOpen((v) => !v)} className="developerSubcard">
                  <Stat label="Detected faces" value={faces} />
                  <Stat label="Detected heads" value={heads} />
                  <Stat label="FPS" value={fps.toFixed(1)} />
                </SubSection>
              )}
            </SectionCard>
          )}

          {settings.showAlertPanel && (
            <SectionCard title="Alerts" open={alertsOpen} onToggle={() => setAlertsOpen((v) => !v)}>
              <div className={`alertPanel ${alertActive || studentAlerts.length > 0 ? "alertPanelOn" : ""}`}>{alertMessage}</div>

              <div className="studentAlertsList">
                {(settings.alertStudentOverlay && pinnedAlertStudents.length > 0 ? pinnedAlertStudents : studentAlerts).length > 0 ? (
                  (settings.alertStudentOverlay && pinnedAlertStudents.length > 0 ? pinnedAlertStudents : studentAlerts).map((item, idx) => {
                    const severity = String(item.severity || "warning").toLowerCase();
                    const fatigueValue = Math.round(Number(item.fatiguePct ?? item.fatigue_pct ?? 0));
                    const attentionValue = Math.round(Number(item.attentionPct ?? item.attention_pct ?? 0));
                    const studentLabel = item.studentId ?? item.student_id ?? item.id ?? "";
                    const alertIndex = item.alertIndex ?? idx + 1;
                    const heldText = item.secondsLeft ? ` · visible for ${Math.ceil(item.secondsLeft / 60)}m` : "";

                    return (
                      <div
                        key={`${idx}-${severity}-${fatigueValue}-${attentionValue}-${studentLabel}`}
                        className={`studentAlertItem ${severity === "critical" ? "studentAlertCritical" : "studentAlertWarning"}`}
                      >
                        <div className="studentAlertTop">
                          <span className="studentAlertId">
                            Alert {alertIndex}
                            {studentLabel ? ` · Student ${studentLabel}` : ""}
                          </span>
                          <span className="studentAlertSeverity">
                            {severity}
                            {heldText}
                          </span>
                        </div>

                        <div className="studentAlertText">
                          {item.message ||
                            (fatigueValue > thresholds.student_fatigue_critical
                              ? `Very fatigued (${fatigueValue}%)`
                              : attentionValue < thresholds.student_attention_critical
                                ? `Very inattentive (${attentionValue}%)`
                                : "Needs attention")}
                        </div>

                        <div className="studentAlertMetrics">
                          Fatigue: <strong>{fatigueValue}%</strong> | Attention: <strong>{attentionValue}%</strong>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div className="reportEmpty">No student alerts</div>
                )}
              </div>
            </SectionCard>
          )}

          {settings.showDecisionPanel && (
            <SectionCard title="Suggestions" open={decisionOpen} onToggle={() => setDecisionOpen((v) => !v)}>
              <SubSection title="Class" open={groupDecisionOpen} onToggle={() => setGroupDecisionOpen((v) => !v)}>
                <div className="reportItem decisionItem">
                  <div className="reportLine">{groupDecision.summary}</div>
                  {groupDecision.suggestions.length > 0 && (
                    <div className="decisionSuggestions">
                      {groupDecision.suggestions.map((item, idx) => (
                        <div key={idx} className="decisionSuggestionLine">
                          • {item}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </SubSection>

              <SubSection title="Student" open={individualDecisionOpen} onToggle={() => setIndividualDecisionOpen((v) => !v)}>
                <div className="reportItem decisionItem">
                  <div className="reportLine">{studentDecision.summary}</div>
                  {studentDecision.suggestions.length > 0 && (
                    <div className="decisionSuggestions">
                      {studentDecision.suggestions.map((item, idx) => (
                        <div key={idx} className="decisionSuggestionLine">
                          • {item}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </SubSection>
            </SectionCard>
          )}

          {settings.showInfoPanel && (
            <SectionCard title="Info" open={infoOpen} onToggle={() => setInfoOpen((v) => !v)} className="infoCard" bodyClassName="infoBody">
              <div className="reportsList">
                {settings.reportPreview && recentReports.length > 0 ? recentReports.map((report) => <ReportItem key={report.id} report={report} />) : <ReportItem report={null} />}
              </div>
            </SectionCard>
          )}
        </div>
      </div>

      <div className="bottomBar">
        <div className="bottomLeft">
          <div className="sessionTimer sessionTimerBottom">
            <span>Timer:</span> {formatDuration(remainingSeconds)}
            <div className="timerBubble">This is the timer. You have {formatLongDuration(remainingSeconds)} left in this session.</div>
          </div>
        </div>

        <div className="bottomCenter">
          <button className="btn btnPrimary" onClick={start} disabled={running || status === "camera error"}>
            Start
          </button>

          <button className="btn btnGhost" onClick={stop} disabled={!running}>
            Stop
          </button>

          {settings.reportEnabled && (
            <div className="raportButtonWrap">
              <button className={`btn btnReport ${raportReady && !running ? "ready" : ""}`} onClick={downloadRaport} disabled={!raportReady || running}>
                Raport
              </button>

              {(!raportReady || running) && <div className="timerBubble raportBubble">The report will be available only at the end of the session.</div>}
            </div>
          )}
        </div>

        <div className="bottomRight">
          {developerMode && (
            <div className="boxesMenu">
              <button className={`btn btnToggle boxesMainButton ${boxesOn ? "on" : "off"}`} onClick={toggleBoxes} disabled={status === "camera error"}>
                <span>Boxes: {boxesOn ? "ON" : "OFF"}</span>
                <span
                  className={`boxesArrow ${boxesOptionsOpen ? "open" : ""}`}
                  role="button"
                  tabIndex={0}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    setBoxesOptionsOpen((v) => !v);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      e.stopPropagation();
                      setBoxesOptionsOpen((v) => !v);
                    }
                  }}
                >
                  {boxesOptionsOpen ? "⌃" : "⌄"}
                </span>
              </button>

              {boxesOptionsOpen && (
                <div className="boxesDropdown">
                  <label className={`boxCheck ${showFace ? "active" : ""}`}>
                    <input type="checkbox" checked={showFace} onChange={(e) => setShowFace(e.target.checked)} disabled={!boxesOn} />
                    <span>Face / Side face</span>
                  </label>

                  <label className={`boxCheck ${showHead ? "active" : ""}`}>
                    <input type="checkbox" checked={showHead} onChange={(e) => setShowHead(e.target.checked)} disabled={!boxesOn} />
                    <span>Head</span>
                  </label>
                </div>
              )}
            </div>
          )}

          <button className="btn btnSettings" onClick={() => setSettingsOpen(true)} type="button" aria-label="Open settings">
            ⚙
          </button>

          <button className={`btn btnToggle ${developerMode ? "on" : "off"}`} onClick={() => setDeveloperMode((v) => !v)}>
            Test: {developerMode ? "ON" : "OFF"}
          </button>
        </div>
      </div>

      {settingsOpen && (
        <div className="settingsBackdrop" onMouseDown={() => setSettingsOpen(false)}>
          <div className="settingsModal" onMouseDown={(e) => e.stopPropagation()}>
            <div className="settingsHeader">
              <div>
                <div className="settingsTitle">Settings</div>
              </div>

              <button className="settingsClose" onClick={() => setSettingsOpen(false)} type="button">
                ×
              </button>
            </div>

            <div className="settingsGrid">
              <div className="settingsBlock">
                <div className="settingsBlockTitle">Timer</div>

                <label className="settingsField">
                  <span>Session duration</span>
                  <div className="settingsInlineInput">
                    <input
                      type="number"
                      min="1"
                      max="240"
                      value={settings.sessionMinutes}
                      disabled={running}
                      onChange={(e) => updateSetting("sessionMinutes", Math.max(1, Math.min(240, Number(e.target.value) || 1)))}
                    />
                    <span>min</span>
                  </div>
                </label>

                <div className="settingsHint">Timer changes apply when a new session starts.</div>
              </div>

              <div className="settingsBlock">
                <div className="settingsBlockTitle">Raport</div>

                <label className="settingsToggle">
                  <input type="checkbox" checked={settings.reportEnabled} onChange={(e) => updateSetting("reportEnabled", e.target.checked)} />
                  <span>Show raport button</span>
                </label>

                <label className="settingsToggle">
                  <input type="checkbox" checked={settings.reportPreview} onChange={(e) => updateSetting("reportPreview", e.target.checked)} />
                  <span>Show raport preview in Info</span>
                </label>

                <div className="settingsDivider" />

                <label className="settingsToggle">
                  <input type="radio" name="reportType" checked={(settings.reportType || "teacher") === "teacher"} onChange={() => updateSetting("reportType", "teacher")} />
                  <span>Teacher raport</span>
                </label>

                <label className="settingsToggle">
                  <input type="radio" name="reportType" checked={settings.reportType === "developer"} onChange={() => updateSetting("reportType", "developer")} />
                  <span>Developer raport</span>
                </label>

                <div className="settingsHint">Teacher raport keeps only useful classroom information. Developer raport keeps all technical information and all charts.</div>
              </div>

              <div className="settingsBlock">
                <div className="settingsBlockTitle">Raport charts</div>

                <label className="settingsToggle">
                  <input type="checkbox" checked={settings.chartsIndividual} disabled={settings.reportType === "developer"} onChange={(e) => updateSetting("chartsIndividual", e.target.checked)} />
                  <span>Individual charts</span>
                </label>

                <label className="settingsToggle">
                  <input type="checkbox" checked={settings.chartsGroup} disabled={settings.reportType === "developer"} onChange={(e) => updateSetting("chartsGroup", e.target.checked)} />
                  <span>Group charts</span>
                </label>

                <label className="settingsToggle">
                  <input type="checkbox" checked={settings.chartsAlerts} disabled={settings.reportType === "developer"} onChange={(e) => updateSetting("chartsAlerts", e.target.checked)} />
                  <span>Alert-student charts</span>
                </label>

                <div className="settingsHint">In developer mode all chart groups are exported automatically.</div>
              </div>

              <div className="settingsBlock">
                <div className="settingsBlockTitle">Visible information</div>

                <label className="settingsToggle">
                  <input type="checkbox" checked={settings.showOverlayInfo} onChange={(e) => updateSetting("showOverlayInfo", e.target.checked)} />
                  <span>Show video overlay metrics</span>
                </label>

                <label className="settingsToggle">
                  <input type="checkbox" checked={settings.showStatisticsPanel} onChange={(e) => updateSetting("showStatisticsPanel", e.target.checked)} />
                  <span>Show Statistics panel</span>
                </label>

                <label className="settingsToggle">
                  <input type="checkbox" checked={settings.showAlertPanel} onChange={(e) => updateSetting("showAlertPanel", e.target.checked)} />
                  <span>Show Alert panel</span>
                </label>

                <label className="settingsToggle">
                  <input type="checkbox" checked={settings.showDecisionPanel} onChange={(e) => updateSetting("showDecisionPanel", e.target.checked)} />
                  <span>Show Suggestions panel</span>
                </label>

                <label className="settingsToggle">
                  <input type="checkbox" checked={settings.showInfoPanel} onChange={(e) => updateSetting("showInfoPanel", e.target.checked)} />
                  <span>Show Info panel</span>
                </label>

                <label className="settingsToggle">
                  <input type="checkbox" checked={settings.alertStudentOverlay} onChange={(e) => updateSetting("alertStudentOverlay", e.target.checked)} />
                  <span>Show red alert box around alert students</span>
                </label>

                <div className="settingsHint">Default ON. When disabled, the red persistent box and the left student alert overlay are hidden.</div>
              </div>

              <div className="settingsBlock">
                <div className="settingsBlockTitle">Alerts</div>

                <label className="settingsToggle">
                  <input type="checkbox" checked={settings.soundAlerts} onChange={(e) => updateSetting("soundAlerts", e.target.checked)} />
                  <span>Alert sound</span>
                </label>

                <div className="settingsHint">Visual alerts stay active even when sound is disabled.</div>
              </div>
            </div>

            <div className="settingsFooter">
              <button
                className="btn btnGhost"
                type="button"
                onClick={() =>
                  setSettings({
                    sessionMinutes: DEFAULT_SESSION_MINUTES,
                    soundAlerts: true,
                    showOverlayInfo: true,
                    showStatisticsPanel: true,
                    showAlertPanel: true,
                    showDecisionPanel: true,
                    showInfoPanel: true,
                    alertStudentOverlay: true,
                    reportEnabled: true,
                    reportPreview: true,
                    reportType: DEFAULT_REPORT_CONFIG.reportType,
                    chartsIndividual: DEFAULT_REPORT_CONFIG.chartsIndividual,
                    chartsGroup: DEFAULT_REPORT_CONFIG.chartsGroup,
                    chartsAlerts: DEFAULT_REPORT_CONFIG.chartsAlerts,
                  })
                }
              >
                Reset
              </button>

              <button className="btn btnPrimary" type="button" onClick={() => setSettingsOpen(false)}>
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}