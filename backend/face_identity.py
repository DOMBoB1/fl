from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from keras_facenet import FaceNet

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel("ERROR")


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def variance_of_laplacian(img_bgr: np.ndarray) -> float:
    if img_bgr is None or img_bgr.size == 0:
        return 0.0

    if img_bgr.ndim == 3 and img_bgr.shape[2] >= 3:
        b = img_bgr[..., 0].astype(np.float32)
        g = img_bgr[..., 1].astype(np.float32)
        r = img_bgr[..., 2].astype(np.float32)
        gray = 0.114 * b + 0.587 * g + 0.299 * r
    else:
        gray = img_bgr.astype(np.float32)

    if gray.size < 4:
        return 0.0

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    grad_mag = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)
    return float(grad_mag.var())


@dataclass
class Identity:
    identity_id: int
    embedding: np.ndarray
    created_at: float
    last_seen: float
    hits: int = 1


@dataclass
class Candidate:
    embeddings: List[np.ndarray] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    hits: int = 0


class FaceIdentityManager:
    def __init__(
        self,
        sim_threshold: float = 0.72,
        min_face_size: int = 80,
        min_blur_score: float = 80.0,
        candidate_hits: int = 5,
        candidate_ttl_s: float = 3.0,
    ):
        self.embedder = FaceNet()
        self.sim_threshold = sim_threshold
        self.min_face_size = min_face_size
        self.min_blur_score = min_blur_score
        self.candidate_hits = candidate_hits
        self.candidate_ttl_s = candidate_ttl_s

        self.identities: Dict[int, Identity] = {}
        self.track_to_identity: Dict[int, int] = {}
        self.candidates: Dict[int, Candidate] = {}
        self.next_identity_id = 1

    def _valid_face(self, face_bgr: np.ndarray) -> bool:
        if face_bgr is None or face_bgr.size == 0:
            return False
        h, w = face_bgr.shape[:2]
        if min(h, w) < self.min_face_size:
            return False
        if variance_of_laplacian(face_bgr) < self.min_blur_score:
            return False
        return True

    def _get_embedding(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self._valid_face(face_bgr):
            return None
        arr = np.expand_dims(face_bgr.astype("float32"), axis=0)
        emb = self.embedder.embeddings(arr)[0]
        return l2_normalize(emb.astype(np.float32))

    def _match_identity(self, emb: np.ndarray) -> Tuple[Optional[int], float]:
        best_id = None
        best_score = -1.0

        for iid, ident in self.identities.items():
            s = cosine_sim(emb, ident.embedding)
            if s > best_score:
                best_score = s
                best_id = iid

        if best_id is not None and best_score >= self.sim_threshold:
            return best_id, best_score
        return None, best_score

    def _create_identity_from_candidate(self, track_id: int, now: float) -> int:
        cand = self.candidates[track_id]
        avg_emb = np.mean(np.stack(cand.embeddings, axis=0), axis=0)
        avg_emb = l2_normalize(avg_emb)

        iid = self.next_identity_id
        self.next_identity_id += 1

        self.identities[iid] = Identity(
            identity_id=iid,
            embedding=avg_emb,
            created_at=now,
            last_seen=now,
            hits=cand.hits,
        )
        self.track_to_identity[track_id] = iid
        del self.candidates[track_id]
        return iid

    def update_track(self, track_id: int, face_bgr: np.ndarray, now: float) -> Optional[int]:
        emb = self._get_embedding(face_bgr)
        if emb is None:
            return self.track_to_identity.get(track_id)

        if track_id in self.track_to_identity:
            iid = self.track_to_identity[track_id]
            ident = self.identities.get(iid)
            if ident is not None:
                ident.embedding = l2_normalize(0.8 * ident.embedding + 0.2 * emb)
                ident.last_seen = now
                ident.hits += 1
            return iid

        matched_iid, _score = self._match_identity(emb)
        if matched_iid is not None:
            self.track_to_identity[track_id] = matched_iid
            ident = self.identities[matched_iid]
            ident.embedding = l2_normalize(0.85 * ident.embedding + 0.15 * emb)
            ident.last_seen = now
            ident.hits += 1
            return matched_iid

        cand = self.candidates.get(track_id)
        if cand is None:
            cand = Candidate(first_seen=now, last_seen=now, hits=0)
            self.candidates[track_id] = cand

        cand.embeddings.append(emb)
        if len(cand.embeddings) > 8:
            cand.embeddings = cand.embeddings[-8:]
        cand.last_seen = now
        cand.hits += 1

        if cand.hits >= self.candidate_hits:
            return self._create_identity_from_candidate(track_id, now)

        return None

    def force_assign_track_identity(self, track_id: int, identity_id: int, face_bgr, now: float) -> int:
        self.track_to_identity[int(track_id)] = int(identity_id)

        ident = self.identities.get(int(identity_id))
        if ident is not None and face_bgr is not None:
            emb = self._get_embedding(face_bgr)
            if emb is not None:
                ident.embedding = l2_normalize(0.85 * ident.embedding + 0.15 * emb)
                ident.last_seen = now
                ident.hits += 1

        self.candidates.pop(int(track_id), None)
        return int(identity_id)

    def cleanup(self, now: float, active_track_ids: List[int]) -> None:
        active_track_ids = set(active_track_ids)

        for tid in list(self.track_to_identity.keys()):
            if tid not in active_track_ids:
                del self.track_to_identity[tid]

        for tid in list(self.candidates.keys()):
            if (now - self.candidates[tid].last_seen) > self.candidate_ttl_s:
                del self.candidates[tid]

    def unique_count(self) -> int:
        return len(self.identities)