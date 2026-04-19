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

function SectionCard({
  title,
  open,
  onToggle,
  children,
  bodyClassName = "",
  rightSlot = null,
  className = "",
}) {
  return (
    <div className={`card collapsibleCard ${open ? "open" : "closed"} ${className}`}>
      <button className="cardHeaderBtn" onClick={onToggle} type="button">
        <div className="cardHeaderLeft">
          <span className={`cardChevron ${open ? "open" : ""}`}>⌄</span>
          <div className="cardTitle">{title}</div>
        </div>
        {rightSlot ? <div className="cardHeaderRight">{rightSlot}</div> : null}
      </button>

      <div className={`cardBody ${open ? "open" : "closed"} ${bodyClassName}`}>
        {children}
      </div>
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

function formatReportTime(ts) {
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function pluralizeStudent(count) {
  return `${count} student${count === 1 ? "" : "s"}`;
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
    </div>
  );
}

function buildTeacherDecision({
  faces,
  fatigue,
  attention,
  fatigueAlertActive,
  attentionAlertActive,
  studentAlerts,
  thresholds,
}) {
  const suggestions = [];
  const observations = [];

  if (fatigueAlertActive && attentionAlertActive) {
    observations.push(
      `Class fatigue is high (${fatigue}% ≥ ${thresholds.class_fatigue_on}%) and class attention is low (${attention}% ≤ ${thresholds.class_attention_on}%).`
    );
    suggestions.push("Consider a short break or a quick activity change.");
    suggestions.push("Reduce passive lecturing and re-engage the room with questions.");
  } else if (fatigueAlertActive) {
    observations.push(
      `Class fatigue is high (${fatigue}% ≥ ${thresholds.class_fatigue_on}%).`
    );
    suggestions.push("Consider a short pause, stretch break, or lighter activity.");
    suggestions.push("Switch to a more interactive task for a few minutes.");
  } else if (attentionAlertActive) {
    observations.push(
      `Class attention is low (${attention}% ≤ ${thresholds.class_attention_on}%).`
    );
    suggestions.push("Try re-engaging students with direct questions or a quick recap.");
    suggestions.push("Change pace, tone, or task type to recover attention.");
  } else {
    observations.push(
      `No class-level alert. Current class averages are fatigue ${fatigue}% and attention ${attention}%.`
    );
  }

  if (Array.isArray(studentAlerts) && studentAlerts.length > 0) {
    const critical = studentAlerts.filter(
      (s) => String(s.severity || "").toLowerCase() === "critical"
    );
    const warning = studentAlerts.filter(
      (s) => String(s.severity || "").toLowerCase() === "warning"
    );

    if (critical.length > 0) {
      observations.push(
        `${pluralizeStudent(critical.length)} require${critical.length === 1 ? "s" : ""} immediate individual attention.`
      );
      suggestions.push(
        `Check ${critical.length === 1 ? "the flagged student" : "the flagged students"} individually.`
      );
    } else if (warning.length > 0) {
      observations.push(
        `${pluralizeStudent(warning.length)} show${warning.length === 1 ? "s" : ""} signs of inattention or fatigue.`
      );
      suggestions.push("Monitor flagged students and consider a local interaction.");
    }
  } else {
    observations.push("No individual student alert is active.");
  }

  if (
    !fatigueAlertActive &&
    !attentionAlertActive &&
    Array.isArray(studentAlerts) &&
    studentAlerts.length === 0
  ) {
    if (faces === 1) {
      suggestions.push("Continue monitoring the current student.");
    } else if (faces > 1) {
      suggestions.push("Continue the current teaching flow and monitor trends.");
    }
  }

  return {
    summary: observations.join(" "),
    suggestions: [...new Set(suggestions)],
  };
}

export default function App() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const captureRef = useRef(null);
  const rafRef = useRef(null);
  const timerRef = useRef(null);
  const inFlightRef = useRef(false);
  const audioRef = useRef(null);

  const boxesOnRef = useRef(false);
  const showFaceRef = useRef(true);
  const showHeadRef = useRef(true);

  const prevAnyAlertRef = useRef(false);

  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("idle");
  const [boxesOn, setBoxesOn] = useState(false);
  const [showFace, setShowFace] = useState(true);
  const [showHead, setShowHead] = useState(true);
  const [raportReady, setRaportReady] = useState(false);

  const [statsOpen, setStatsOpen] = useState(true);
  const [alertsOpen, setAlertsOpen] = useState(true);
  const [decisionOpen, setDecisionOpen] = useState(true);
  const [infoOpen, setInfoOpen] = useState(true);

  const [faces, setFaces] = useState(0);
  const [heads, setHeads] = useState(0);
  const [fatigue, setFatigue] = useState(0);
  const [attention, setAttention] = useState(0);
  const [alertActive, setAlertActive] = useState(false);
  const [fatigueAlertActive, setFatigueAlertActive] = useState(false);
  const [attentionAlertActive, setAttentionAlertActive] = useState(false);
  const [studentAlerts, setStudentAlerts] = useState([]);
  const [fps, setFps] = useState(0);
  const [recentReports, setRecentReports] = useState([]);
  const [alertEventCount, setAlertEventCount] = useState(0);
  const [validObservations, setValidObservations] = useState(0);

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

  const facesDataRef = useRef([]);
  const smoothFaceRef = useRef(new Map());
  const smoothHeadRef = useRef(new Map());

  const statusPill = useMemo(() => {
    if (status === "running") return "pill pillLive";
    if (status === "camera ready") return "pill pillReady";
    if (status === "camera error") return "pill pillErr";
    return "pill";
  }, [status]);

  const alertMessage = useMemo(() => {
    if (fatigueAlertActive && attentionAlertActive) {
      return `⚠️ High fatigue (${fatigue}%) and low attention (${attention}%)`;
    }
    if (fatigueAlertActive) {
      return `⚠️ High fatigue: ${fatigue}%`;
    }
    if (attentionAlertActive) {
      return `⚠️ Low attention: ${attention}%`;
    }
    if (studentAlerts.length > 0) {
      return `⚠️ ${pluralizeStudent(studentAlerts.length)} need${
        studentAlerts.length === 1 ? "s" : ""
      } attention`;
    }
    return "";
  }, [fatigueAlertActive, attentionAlertActive, studentAlerts, fatigue, attention]);

  const teacherDecision = useMemo(() => {
    return buildTeacherDecision({
      faces,
      fatigue,
      attention,
      fatigueAlertActive,
      attentionAlertActive,
      studentAlerts,
      thresholds,
    });
  }, [
    faces,
    fatigue,
    attention,
    fatigueAlertActive,
    attentionAlertActive,
    studentAlerts,
    thresholds,
  ]);

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
    const anyNow =
      Boolean(fatigueAlertActive) ||
      Boolean(attentionAlertActive) ||
      (Array.isArray(studentAlerts) &&
        studentAlerts.some((s) => String(s.severity || "").toLowerCase() === "critical"));

    const triggered = !prevAnyAlertRef.current && anyNow;

    if (triggered && audioRef.current) {
      try {
        audioRef.current.currentTime = 0;
        const p = audioRef.current.play();
        if (p && typeof p.catch === "function") {
          p.catch(() => {});
        }
      } catch (_err) {}
    }

    prevAnyAlertRef.current = anyNow;
  }, [fatigueAlertActive, attentionAlertActive, studentAlerts]);

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

  function drawOneBox(ctx, ov, boxN, color, mapRef, id, now, label = "", dashed = false) {
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

    if (dashed) {
      ctx.setLineDash([10 * dpr, 6 * dpr]);
    } else {
      ctx.setLineDash([]);
    }

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
      const facesData = Array.isArray(facesDataRef.current) ? facesDataRef.current : [];
      const now = performance.now();

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
          let color = "rgba(38, 182, 222, 0.96)";
          let label = "FACE";

          if (faceKind === "side_face" || faceKind === "sideface") {
            color = "rgba(182, 70, 255, 0.96)";
            label = "SIDE FACE";
          }

          drawOneBox(
            ctx,
            ov,
            faceBox,
            color,
            smoothFaceRef,
            `face_${baseId}`,
            now,
            label
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

      ctx.save();
      ctx.clearRect(0, 0, targetW, targetH);
      ctx.drawImage(video, 0, 0, targetW, targetH);
      ctx.restore();

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

      const data = await res.json().catch(() => null);

      if (!res.ok) {
        console.error("Analyze backend error:", data || res.statusText);
        return;
      }

      const s = data?.stats || data;
      if (!s) return;

      setFaces(Number(s.faces ?? 0));
      setHeads(Number(s.heads ?? 0));
      setFatigue(Math.round(s.class_avg_fatigue_pct ?? 0));
      setAttention(Math.round(s.class_avg_attention_pct ?? 0));
      setAlertActive(Boolean(s.alert_active ?? false));
      setFatigueAlertActive(Boolean(s.fatigue_alert_active ?? false));
      setAttentionAlertActive(Boolean(s.attention_alert_active ?? false));
      setStudentAlerts(Array.isArray(s.student_alerts) ? s.student_alerts : []);
      setFps(Number(s.fps ?? 0));
      setAlertEventCount(Number(s.alert_event_count ?? 0));
      setValidObservations(Number(s.valid_observations ?? 0));

      if (s.thresholds && typeof s.thresholds === "object") {
        setThresholds((prev) => ({
          ...prev,
          ...s.thresholds,
        }));
      }

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
      const res = await fetch(`${BACKEND}/session/start`, { method: "POST" });
      if (!res.ok) {
        const data = await res.json().catch(() => null);
        console.error("Session start backend error:", data || res.statusText);
      }
    } catch (err) {
      console.error("Session start error:", err);
    }

    setRunning(true);
    setStatus("running");
    setRaportReady(false);
    setAlertEventCount(0);
    setValidObservations(0);

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
      const res = await fetch(`${BACKEND}/session/stop`, { method: "POST" });
      const data = await res.json().catch(() => null);

      if (!res.ok) {
        console.error("Session stop backend error:", data || res.statusText);
      } else {
        const summary = data?.summary || {};
        setAlertEventCount(Number(summary.alert_event_count ?? 0));
        setValidObservations(Number(summary.valid_observations ?? 0));
      }

      setRaportReady(true);
    } catch (err) {
      console.error("Session stop error:", err);
      setRaportReady(true);
    }
  }

  function toggleBoxes() {
    setBoxesOn((v) => !v);
    facesDataRef.current = [];
    smoothFaceRef.current.clear();
    smoothHeadRef.current.clear();
  }

  async function downloadRaport() {
    try {
      const res = await fetch(`${BACKEND}/session/report`);

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
            <div className="stageTitle">Class Monitor</div>
            <div className={statusPill}>{status === "running" ? "LIVE" : status}</div>
          </div>

          <div className="videoWrap">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="video"
              style={{ transform: "none" }}
            />
            <canvas ref={overlayRef} className="overlayCanvas" />

            <div className="overlayTopLeft">
              <div className="miniStat">
                <span className="miniKey">Faces</span>
                <span className="miniVal">{faces}</span>
              </div>
              <div className="miniStat">
                <span className="miniKey">Heads</span>
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

            {alertActive && <div className="alertBanner">{alertMessage}</div>}
          </div>

          <canvas ref={captureRef} className="hiddenCanvas" />
        </div>

        <div className="sidebar">
          <SectionCard
            title="Statistics"
            open={statsOpen}
            onToggle={() => setStatsOpen((v) => !v)}
            rightSlot={<span className="sectionBadge">{faces} / {heads}</span>}
          >
            <Stat label="Faces" value={faces} />
            <Stat label="Heads" value={heads} />
            <Stat label="Fatigue Avg" value={`${fatigue}%`} />
            <Stat label="Attention Avg" value={`${attention}%`} />
            <Stat label="Valid observations" value={validObservations} />
            <Stat label="Alert events" value={alertEventCount} />
            <Stat
              label="Class Alert"
              value={
                fatigueAlertActive && attentionAlertActive
                  ? "FATIGUE + ATTENTION"
                  : fatigueAlertActive
                  ? "FATIGUE"
                  : attentionAlertActive
                  ? "ATTENTION"
                  : "OFF"
              }
            />
            <Stat label="Students flagged" value={studentAlerts.length} />
            <Stat label="FPS" value={fps.toFixed(1)} />
          </SectionCard>

          <SectionCard
            title="Student Alerts"
            open={alertsOpen}
            onToggle={() => setAlertsOpen((v) => !v)}
            rightSlot={
              <span className={`sectionBadge ${studentAlerts.length > 0 ? "warn" : ""}`}>
                {studentAlerts.length}
              </span>
            }
          >
            <div className="studentAlertsList">
              {studentAlerts.length > 0 ? (
                studentAlerts.map((item) => (
                  <div
                    key={item.student_id}
                    className={`studentAlertItem ${
                      item.severity === "critical"
                        ? "studentAlertCritical"
                        : "studentAlertWarning"
                    }`}
                  >
                    <div className="studentAlertTop">
                      <span className="studentAlertId">{`Student ${item.student_id}`}</span>
                      <span className="studentAlertSeverity">{item.severity}</span>
                    </div>

                    <div className="studentAlertText">
                      {item.message ||
                        (item.fatigue_pct > thresholds.student_fatigue_critical
                          ? `Very fatigued (${Math.round(item.fatigue_pct)}%)`
                          : item.attention_pct < thresholds.student_attention_critical
                          ? `Very inattentive (${Math.round(item.attention_pct)}%)`
                          : "Needs attention")}
                    </div>

                    <div className="studentAlertMetrics">
                      Fatigue: <strong>{Math.round(item.fatigue_pct)}%</strong> | Attention:{" "}
                      <strong>{Math.round(item.attention_pct)}%</strong>
                    </div>
                  </div>
                ))
              ) : alertActive ? (
                <div className="reportEmpty">Only class-level alert is active</div>
              ) : (
                <div className="reportEmpty">No individual alerts</div>
              )}
            </div>
          </SectionCard>

          <SectionCard
            title="Decision"
            open={decisionOpen}
            onToggle={() => setDecisionOpen((v) => !v)}
            rightSlot={<span className="sectionBadge">AI</span>}
          >
            <div className="reportItem">
              <div className="reportLine">{teacherDecision.summary}</div>
              {teacherDecision.suggestions.length > 0 && (
                <div className="decisionSuggestions">
                  {teacherDecision.suggestions.map((item, idx) => (
                    <div key={idx} className="decisionSuggestionLine">
                      • {item}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </SectionCard>

          <SectionCard
            title="Info"
            open={infoOpen}
            onToggle={() => setInfoOpen((v) => !v)}
            className="infoCard"
            bodyClassName="infoBody"
            rightSlot={<span className="sectionBadge">{recentReports.length}</span>}
          >
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
          </SectionCard>
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
              <span>Face / Side face</span>
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
            onClick={downloadRaport}
            disabled={!raportReady || running}
            title="Exportă raportul sesiunii curente în Excel"
          >
            Raport
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