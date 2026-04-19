from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


# =========================
# Basic geometry utilities
# =========================

@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return self.width() * self.height()

    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def clamp(self, frame_w: int, frame_h: int) -> "Box":
        return Box(
            max(0.0, min(self.x1, frame_w - 1)),
            max(0.0, min(self.y1, frame_h - 1)),
            max(0.0, min(self.x2, frame_w - 1)),
            max(0.0, min(self.y2, frame_h - 1)),
        )

    def expand(self, sx: float, sy: float) -> "Box":
        cx, cy = self.center()
        nw = self.width() * sx
        nh = self.height() * sy
        return Box(
            cx - nw / 2.0,
            cy - nh / 2.0,
            cx + nw / 2.0,
            cy + nh / 2.0,
        )


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area() + b.area() - inter
    if union <= 0:
        return 0.0
    return inter / union


def contains_ratio(outer: Box, inner: Box) -> float:
    ix1 = max(outer.x1, inner.x1)
    iy1 = max(outer.y1, inner.y1)
    ix2 = min(outer.x2, inner.x2)
    iy2 = min(outer.y2, inner.y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    denom = inner.area()
    if denom <= 0:
        return 0.0
    return inter / denom


def center_distance_norm(a: Box, b: Box) -> float:
    ax, ay = a.center()
    bx, by = b.center()
    d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    norm = max(1.0, (a.width() + a.height() + b.width() + b.height()) / 4.0)
    return d / norm


# =========================
# Detection model
# =========================

@dataclass
class Detection:
    label: str
    box: Box
    conf: float
    source: str = ""
    track_id: Optional[int] = None
    meta: Dict = field(default_factory=dict)

    def copy_with(
        self,
        *,
        label: Optional[str] = None,
        box: Optional[Box] = None,
        conf: Optional[float] = None,
        source: Optional[str] = None,
        meta: Optional[Dict] = None,
    ) -> "Detection":
        return Detection(
            label=label if label is not None else self.label,
            box=box if box is not None else self.box,
            conf=conf if conf is not None else self.conf,
            source=source if source is not None else self.source,
            track_id=self.track_id,
            meta=meta if meta is not None else dict(self.meta),
        )


@dataclass
class BoxLogicConfig:
    # Minimum confidence by label
    min_conf_head: float = 0.25
    min_conf_face: float = 0.25
    min_conf_eye: float = 0.20
    min_conf_nose: float = 0.20
    min_conf_mouth: float = 0.20

    # Matching thresholds
    face_inside_head_ratio: float = 0.55
    component_inside_face_ratio: float = 0.45
    component_inside_head_ratio: float = 0.35
    max_face_head_center_dist: float = 0.90

    # Soft recovery rules
    allow_head_from_face: bool = True
    allow_head_from_partial_face: bool = True
    estimated_head_conf_from_face: float = 0.55
    estimated_head_conf_from_partial: float = 0.40

    # Head growth factors when estimated from face
    head_from_face_scale_x: float = 1.90
    head_from_face_scale_y: float = 2.35

    # Partial-face logic
    min_partial_components_for_head_candidate: int = 1
    min_partial_components_for_face_candidate: int = 2

    # Keep unmatched face instead of dropping it
    keep_unmatched_face: bool = True

    # Label aliases
    head_labels: Tuple[str, ...] = ("head", "person_head")
    face_labels: Tuple[str, ...] = ("face", "sideface")
    eye_labels: Tuple[str, ...] = ("eye", "left_eye", "right_eye")
    nose_labels: Tuple[str, ...] = ("nose",)
    mouth_labels: Tuple[str, ...] = ("mouth", "lips")


@dataclass
class ReconciledObject:
    head: Optional[Detection] = None
    face: Optional[Detection] = None
    eyes: List[Detection] = field(default_factory=list)
    noses: List[Detection] = field(default_factory=list)
    mouths: List[Detection] = field(default_factory=list)

    status: str = "unknown"
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)


# =========================
# Main logic
# =========================

class BoxLogic:
    def __init__(self, config: Optional[BoxLogicConfig] = None):
        self.cfg = config or BoxLogicConfig()

    def process(
        self,
        detections: List[Detection],
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
    ) -> Dict[str, List]:
        filtered = self._filter_by_conf(detections)

        heads = [d for d in filtered if d.label in self.cfg.head_labels]
        faces = [d for d in filtered if d.label in self.cfg.face_labels]
        eyes = [d for d in filtered if d.label in self.cfg.eye_labels]
        noses = [d for d in filtered if d.label in self.cfg.nose_labels]
        mouths = [d for d in filtered if d.label in self.cfg.mouth_labels]

        objects = self._pair_heads_faces(heads, faces)
        self._attach_components(objects, eyes, noses, mouths)
        self._recover_missing_heads(objects, frame_w, frame_h)
        self._attach_orphan_components_to_new_objects(objects, eyes, noses, mouths, frame_w, frame_h)
        self._finalize_status(objects)

        final_heads: List[Detection] = []
        final_faces: List[Detection] = []
        final_components: List[Detection] = []

        for obj in objects:
            if obj.head is not None:
                final_heads.append(obj.head)
            if obj.face is not None:
                final_faces.append(obj.face)
            final_components.extend(obj.eyes)
            final_components.extend(obj.noses)
            final_components.extend(obj.mouths)

        return {
            "objects": objects,
            "heads": final_heads,
            "faces": final_faces,
            "components": final_components,
        }

    # -------------------------
    # Stage 1: filtering
    # -------------------------

    def _filter_by_conf(self, detections: List[Detection]) -> List[Detection]:
        out: List[Detection] = []
        for d in detections:
            if d.label in self.cfg.head_labels and d.conf >= self.cfg.min_conf_head:
                out.append(d)
            elif d.label in self.cfg.face_labels and d.conf >= self.cfg.min_conf_face:
                out.append(d)
            elif d.label in self.cfg.eye_labels and d.conf >= self.cfg.min_conf_eye:
                out.append(d)
            elif d.label in self.cfg.nose_labels and d.conf >= self.cfg.min_conf_nose:
                out.append(d)
            elif d.label in self.cfg.mouth_labels and d.conf >= self.cfg.min_conf_mouth:
                out.append(d)
        return out

    # -------------------------
    # Stage 2: head-face pairing
    # -------------------------

    def _pair_heads_faces(
        self,
        heads: List[Detection],
        faces: List[Detection],
    ) -> List[ReconciledObject]:
        objects: List[ReconciledObject] = []
        used_faces = set()

        sorted_heads = sorted(heads, key=lambda d: d.conf, reverse=True)

        for head in sorted_heads:
            best_idx = None
            best_score = -1e9

            for i, face in enumerate(faces):
                if i in used_faces:
                    continue

                inside = contains_ratio(head.box, face.box)
                dist = center_distance_norm(head.box, face.box)

                if inside < self.cfg.face_inside_head_ratio:
                    continue
                if dist > self.cfg.max_face_head_center_dist:
                    continue

                score = (inside * 2.0) + face.conf - dist
                if score > best_score:
                    best_score = score
                    best_idx = i

            obj = ReconciledObject(head=head)

            if best_idx is not None:
                used_faces.add(best_idx)
                obj.face = faces[best_idx]
                obj.notes.append("matched_face_to_head")

            objects.append(obj)

        for i, face in enumerate(faces):
            if i in used_faces:
                continue

            # soft rule: keep unmatched face as its own object
            if self.cfg.keep_unmatched_face:
                objects.append(
                    ReconciledObject(
                        head=None,
                        face=face,
                        notes=["unmatched_face_kept_softly"],
                    )
                )

        return objects

    # -------------------------
    # Stage 3: attach components
    # -------------------------

    def _attach_components(
        self,
        objects: List[ReconciledObject],
        eyes: List[Detection],
        noses: List[Detection],
        mouths: List[Detection],
    ) -> None:
        eye_used = set()
        nose_used = set()
        mouth_used = set()

        for obj in objects:
            anchor_face = obj.face.box if obj.face is not None else None
            anchor_head = obj.head.box if obj.head is not None else None

            if anchor_face is None and anchor_head is None:
                continue

            for i, eye in enumerate(eyes):
                if i in eye_used:
                    continue
                if self._component_matches_object(eye.box, anchor_face, anchor_head):
                    obj.eyes.append(eye)
                    eye_used.add(i)

            for i, nose in enumerate(noses):
                if i in nose_used:
                    continue
                if self._component_matches_object(nose.box, anchor_face, anchor_head):
                    obj.noses.append(nose)
                    nose_used.add(i)

            for i, mouth in enumerate(mouths):
                if i in mouth_used:
                    continue
                if self._component_matches_object(mouth.box, anchor_face, anchor_head):
                    obj.mouths.append(mouth)
                    mouth_used.add(i)

    def _component_matches_object(
        self,
        component: Box,
        face_box: Optional[Box],
        head_box: Optional[Box],
    ) -> bool:
        if face_box is not None:
            if contains_ratio(face_box, component) >= self.cfg.component_inside_face_ratio:
                return True
        if head_box is not None:
            if contains_ratio(head_box, component) >= self.cfg.component_inside_head_ratio:
                return True
        return False

    # -------------------------
    # Stage 4: recover missing head
    # -------------------------

    def _recover_missing_heads(
        self,
        objects: List[ReconciledObject],
        frame_w: Optional[int],
        frame_h: Optional[int],
    ) -> None:
        for obj in objects:
            if obj.head is not None:
                continue

            # Case A: face exists -> estimate head from face
            if obj.face is not None and self.cfg.allow_head_from_face:
                est_box = self._estimate_head_from_face(obj.face.box)
                if frame_w is not None and frame_h is not None:
                    est_box = est_box.clamp(frame_w, frame_h)

                obj.head = Detection(
                    label="head",
                    box=est_box,
                    conf=min(0.99, max(obj.face.conf * 0.85, self.cfg.estimated_head_conf_from_face)),
                    source="box_logic",
                    meta={
                        "synthetic": True,
                        "derived_from": "face",
                    },
                )
                obj.notes.append("synthetic_head_from_face")
                continue

            # Case B: no face, but enough partial components -> estimate soft head
            parts = len(obj.eyes) + len(obj.noses) + len(obj.mouths)
            if (
                obj.face is None
                and self.cfg.allow_head_from_partial_face
                and parts >= self.cfg.min_partial_components_for_head_candidate
            ):
                est_box = self._estimate_head_from_components(obj.eyes, obj.noses, obj.mouths)
                if est_box is not None:
                    if frame_w is not None and frame_h is not None:
                        est_box = est_box.clamp(frame_w, frame_h)

                    obj.head = Detection(
                        label="head",
                        box=est_box,
                        conf=self.cfg.estimated_head_conf_from_partial,
                        source="box_logic",
                        meta={
                            "synthetic": True,
                            "derived_from": "partial_components",
                        },
                    )
                    obj.notes.append("synthetic_head_from_partial_components")

    def _estimate_head_from_face(self, face_box: Box) -> Box:
        return face_box.expand(
            self.cfg.head_from_face_scale_x,
            self.cfg.head_from_face_scale_y,
        )

    def _estimate_head_from_components(
        self,
        eyes: List[Detection],
        noses: List[Detection],
        mouths: List[Detection],
    ) -> Optional[Box]:
        comps = eyes + noses + mouths
        if not comps:
            return None

        x1 = min(c.box.x1 for c in comps)
        y1 = min(c.box.y1 for c in comps)
        x2 = max(c.box.x2 for c in comps)
        y2 = max(c.box.y2 for c in comps)

        base = Box(x1, y1, x2, y2)

        # if we only saw a tiny fragment, estimate a bigger head region softly
        return base.expand(3.0, 3.6)

    # -------------------------
    # Stage 5: orphan components
    # -------------------------

    def _attach_orphan_components_to_new_objects(
        self,
        objects: List[ReconciledObject],
        eyes: List[Detection],
        noses: List[Detection],
        mouths: List[Detection],
        frame_w: Optional[int],
        frame_h: Optional[int],
    ) -> None:
        used_ids = set()
        for obj in objects:
            for d in obj.eyes + obj.noses + obj.mouths:
                used_ids.add(id(d))

        orphan_eyes = [d for d in eyes if id(d) not in used_ids]
        orphan_noses = [d for d in noses if id(d) not in used_ids]
        orphan_mouths = [d for d in mouths if id(d) not in used_ids]

        # Simplu: fiecare grup de componente orfane devine obiect nou dacă are sens minim
        all_orphans = orphan_eyes + orphan_noses + orphan_mouths
        if not all_orphans:
            return

        # Grupare foarte simplă pe proximitate
        groups: List[List[Detection]] = []
        for det in all_orphans:
            placed = False
            for g in groups:
                gx1 = min(x.box.x1 for x in g)
                gy1 = min(x.box.y1 for x in g)
                gx2 = max(x.box.x2 for x in g)
                gy2 = max(x.box.y2 for x in g)
                gbox = Box(gx1, gy1, gx2, gy2).expand(1.8, 1.8)
                if contains_ratio(gbox, det.box) > 0.4 or iou(gbox, det.box) > 0.1:
                    g.append(det)
                    placed = True
                    break
            if not placed:
                groups.append([det])

        for g in groups:
            g_eyes = [d for d in g if d.label in self.cfg.eye_labels]
            g_noses = [d for d in g if d.label in self.cfg.nose_labels]
            g_mouths = [d for d in g if d.label in self.cfg.mouth_labels]

            comp_count = len(g_eyes) + len(g_noses) + len(g_mouths)
            if comp_count < self.cfg.min_partial_components_for_head_candidate:
                continue

            obj = ReconciledObject(
                head=None,
                face=None,
                eyes=g_eyes,
                noses=g_noses,
                mouths=g_mouths,
                notes=["object_created_from_orphan_components"],
            )

            if self.cfg.allow_head_from_partial_face:
                est = self._estimate_head_from_components(g_eyes, g_noses, g_mouths)
                if est is not None:
                    if frame_w is not None and frame_h is not None:
                        est = est.clamp(frame_w, frame_h)
                    obj.head = Detection(
                        label="head",
                        box=est,
                        conf=self.cfg.estimated_head_conf_from_partial,
                        source="box_logic",
                        meta={
                            "synthetic": True,
                            "derived_from": "orphan_components",
                        },
                    )
                    obj.notes.append("synthetic_head_from_orphan_components")

            # optional face candidate if enough components
            if comp_count >= self.cfg.min_partial_components_for_face_candidate:
                x1 = min(c.box.x1 for c in g)
                y1 = min(c.box.y1 for c in g)
                x2 = max(c.box.x2 for c in g)
                y2 = max(c.box.y2 for c in g)
                face_est = Box(x1, y1, x2, y2).expand(1.8, 1.8)
                if frame_w is not None and frame_h is not None:
                    face_est = face_est.clamp(frame_w, frame_h)

                obj.face = Detection(
                    label="face",
                    box=face_est,
                    conf=0.35,
                    source="box_logic",
                    meta={
                        "synthetic": True,
                        "derived_from": "component_group",
                    },
                )
                obj.notes.append("synthetic_face_from_component_group")

            objects.append(obj)

    # -------------------------
    # Stage 6: finalize
    # -------------------------

    def _finalize_status(self, objects: List[ReconciledObject]) -> None:
        for obj in objects:
            parts = len(obj.eyes) + len(obj.noses) + len(obj.mouths)

            if obj.head is not None and obj.face is not None:
                obj.status = "head_face"
                obj.confidence = self._combine_conf(
                    obj.head.conf,
                    obj.face.conf,
                    0.05 * min(parts, 3),
                )
                if obj.head.meta.get("synthetic"):
                    obj.notes.append("head_is_estimated")

            elif obj.head is not None and obj.face is None:
                if parts > 0:
                    obj.status = "head_partial_face"
                    obj.confidence = self._combine_conf(
                        obj.head.conf,
                        0.20,
                        0.05 * min(parts, 3),
                    )
                else:
                    obj.status = "head_only"
                    obj.confidence = obj.head.conf

            elif obj.face is not None and obj.head is None:
                # n-ar trebui să rămână des aici, dar păstrăm soft
                obj.status = "face_only_soft"
                obj.confidence = obj.face.conf
                obj.notes.append("face_without_head_kept_softly")

            elif parts > 0:
                obj.status = "partial_components"
                obj.confidence = min(0.5, 0.15 + 0.1 * parts)

            else:
                obj.status = "unknown"
                obj.confidence = 0.0

    @staticmethod
    def _combine_conf(a: float, b: float, boost: float = 0.0) -> float:
        val = (a + b) / 2.0 + boost
        return max(0.0, min(1.0, val))