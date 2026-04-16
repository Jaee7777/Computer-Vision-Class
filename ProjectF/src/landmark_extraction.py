"""
CS 5330 — Lightweight Real-Time Head Pose Estimation & Facial Expression Mapping
Author: Jaee Oh

Pipeline:
    Stage 1 — Landmark Extraction      (MediaPipe Face Mesh)
    Stage 2 — Head Pose Estimation     (solvePnP + Rodrigues)
    Stage 3 — Expression Parameterization (blend shape weights)
    Stage 4 — Exaggeration + Avatar Rendering
"""

import cv2
import numpy as np
import mediapipe as mp

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

GAMMA = 0.6  # exaggeration curve: lower = more exaggeration (try 0.5–0.8)
DEAD_ZONE = 0.05  # suppress blend shape values below this threshold
SHOW_MESH = True  # toggle MediaPipe mesh overlay on webcam feed
SHOW_AXES = True  # toggle head pose axes overlay


# ─────────────────────────────────────────────
# STAGE 1 — LANDMARK EXTRACTION
# ─────────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # 468 → 478 points (includes iris)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def get_landmarks(frame):
    """
    Run MediaPipe Face Mesh on a BGR frame.
    Returns a (478, 3) float32 array of (x, y, z) in pixel coords,
    or None if no face is detected.
    """
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float32)
    return pts  # shape (478, 3)


# ─────────────────────────────────────────────
# STAGE 2 — HEAD POSE ESTIMATION
# ─────────────────────────────────────────────

# 3D model points of 6 canonical face landmarks (generic face in mm)
MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],  # Nose tip          → landmark 1
        [0.0, -63.6, -12.5],  # Chin              → landmark 152
        [-43.3, 32.7, -26.0],  # Left eye corner   → landmark 33
        [43.3, 32.7, -26.0],  # Right eye corner  → landmark 263
        [-28.9, -28.9, -24.1],  # Left mouth corner → landmark 61
        [28.9, -28.9, -24.1],  # Right mouth corner→ landmark 291
    ],
    dtype=np.float64,
)

# Corresponding MediaPipe landmark indices
POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def estimate_head_pose(landmarks, frame_shape):
    """
    Estimate yaw, pitch, roll from 6 facial landmarks using solvePnP.
    Returns (yaw, pitch, roll) in degrees, or (0, 0, 0) on failure.
    """
    h, w = frame_shape[:2]

    # Camera intrinsics (approximation — replace with calibrated values if available)
    focal_length = w
    cam_matrix = np.array(
        [
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # 2D image points for the 6 landmarks
    image_points = np.array(
        [[landmarks[i][0], landmarks[i][1]] for i in POSE_LANDMARK_IDS],
        dtype=np.float64,
    )

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    # Rodrigues rotation vector → rotation matrix → Euler angles
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll = 0.0

    # TODO (optional): draw 3D axes onto the frame for debugging
    # draw_axes(frame, rvec, tvec, cam_matrix, dist_coeffs)

    return yaw, pitch, roll


# ─────────────────────────────────────────────
# STAGE 3 — EXPRESSION PARAMETERIZATION
# ─────────────────────────────────────────────


def _dist(a, b):
    return np.linalg.norm(a - b)


def compute_blend_shapes(landmarks):
    """
    Compute 6 normalized blend shape weights from landmark geometry.
    All values are in [0, 1] and scale/distance invariant.

    Returns a dict:
        blink_left, blink_right  — 0 = open, 1 = closed
        brow_left,  brow_right   — 0 = neutral, 1 = raised
        mouth_open               — 0 = closed, 1 = open
        mouth_smile              — 0 = neutral, 1 = smiling
    """
    lm = landmarks  # shorthand

    # Normalisation factor: inter-ocular distance (stable across distances)
    inter_ocular = _dist(lm[33][:2], lm[263][:2])
    if inter_ocular < 1e-6:
        inter_ocular = 1.0

    # ── Blink (Eye Aspect Ratio) ──────────────────────────────────────────
    # Left eye:  corners 33↔133, top↔bottom pairs 159↔145, 158↔153
    # Right eye: corners 362↔263, top↔bottom pairs 386↔374, 385↔380
    # TODO: fill in EAR calculation
    # ear_left  = (|p159-p145| + |p158-p153|) / (2 * |p33-p133|)
    # blink_left = 1 - clamp(ear_left / EAR_OPEN_BASELINE)
    blink_left = 0.0  # placeholder
    blink_right = 0.0  # placeholder

    # ── Brow Raise ───────────────────────────────────────────────────────
    # Distance from brow landmark to eyelid, normalised by inter-ocular
    # TODO: choose brow (e.g. 70) and eyelid (e.g. 159) landmark indices
    brow_left = 0.0  # placeholder
    brow_right = 0.0  # placeholder

    # ── Mouth Open ───────────────────────────────────────────────────────
    # Vertical lip gap (lm[13] upper lip, lm[14] lower lip), normalised
    mouth_open = _dist(lm[13][:2], lm[14][:2]) / inter_ocular
    mouth_open = float(np.clip(mouth_open, 0.0, 1.0))

    # ── Mouth Smile ──────────────────────────────────────────────────────
    # TODO: measure lip corner height relative to lip midpoint
    mouth_smile = 0.0  # placeholder

    return {
        "blink_left": blink_left,
        "blink_right": blink_right,
        "brow_left": brow_left,
        "brow_right": brow_right,
        "mouth_open": mouth_open,
        "mouth_smile": mouth_smile,
    }


# ─────────────────────────────────────────────
# STAGE 4A — EXAGGERATION MAPPING
# ─────────────────────────────────────────────


def exaggerate(value, gamma=GAMMA, dead_zone=DEAD_ZONE):
    """
    Apply nonlinear exaggeration: f(x) = x^gamma
    gamma < 1 boosts mid-range values (cartoon snap).
    Values below dead_zone are zeroed to suppress micro-jitter.
    """
    if value < dead_zone:
        return 0.0
    return float(np.clip(value**gamma, 0.0, 1.0))


def apply_exaggeration(blend_shapes):
    """Apply exaggeration to all blend shape weights."""
    return {k: exaggerate(v) for k, v in blend_shapes.items()}


# ─────────────────────────────────────────────
# STAGE 4B — AVATAR RENDERING
# ─────────────────────────────────────────────


def draw_avatar(canvas, blend_shapes, yaw, pitch, roll):
    """
    Draw a simple 2D stylized avatar driven by blend shapes and head pose.
    canvas: blank BGR image to draw on (e.g. 480×480)

    TODO: replace placeholder shapes with your stylized Disney-inspired design.
    """
    h, w = canvas.shape[:2]
    cx, cy = w // 2, h // 2

    bs = blend_shapes  # shorthand

    # ── Head (ellipse, tilted by roll) ───────────────────────────────────
    cv2.ellipse(canvas, (cx, cy), (120, 150), -roll, 0, 360, (255, 220, 180), -1)

    # ── Eyes ─────────────────────────────────────────────────────────────
    eye_open_h = int(30 * (1.0 - bs["blink_left"]))  # height shrinks on blink
    cv2.ellipse(
        canvas,
        (cx - 45, cy - 20),
        (25, max(eye_open_h, 2)),
        0,
        0,
        360,
        (50, 50, 50),
        -1,
    )

    eye_open_h = int(30 * (1.0 - bs["blink_right"]))
    cv2.ellipse(
        canvas,
        (cx + 45, cy - 20),
        (25, max(eye_open_h, 2)),
        0,
        0,
        360,
        (50, 50, 50),
        -1,
    )

    # ── Brows (shift upward on raise) ────────────────────────────────────
    brow_offset_l = int(20 * bs["brow_left"])
    brow_offset_r = int(20 * bs["brow_right"])
    cv2.line(
        canvas,
        (cx - 65, cy - 55 - brow_offset_l),
        (cx - 25, cy - 50 - brow_offset_l),
        (80, 50, 30),
        5,
    )
    cv2.line(
        canvas,
        (cx + 25, cy - 50 - brow_offset_r),
        (cx + 65, cy - 55 - brow_offset_r),
        (80, 50, 30),
        5,
    )

    # ── Mouth ─────────────────────────────────────────────────────────────
    mouth_h = int(25 * bs["mouth_open"])
    smile = int(15 * bs["mouth_smile"])
    cv2.ellipse(
        canvas,
        (cx, cy + 60),
        (35, max(mouth_h + smile, 5)),
        0,
        0,
        180,
        (180, 80, 80),
        -1,
    )

    # TODO: add nose, pupils, hair, ears, color fills, outlines, etc.


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    avatar_canvas_size = (480, 480, 3)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror

        # ── Stage 1: landmarks ──────────────────────────────────────────
        landmarks = get_landmarks(frame)

        if landmarks is not None:

            # ── Stage 2: head pose ──────────────────────────────────────
            yaw, pitch, roll = estimate_head_pose(landmarks, frame.shape)

            # ── Stage 3: blend shapes ───────────────────────────────────
            blend_shapes = compute_blend_shapes(landmarks)

            # ── Stage 4a: exaggeration ──────────────────────────────────
            exag_shapes = apply_exaggeration(blend_shapes)

            # ── Stage 4b: render avatar ─────────────────────────────────
            avatar = np.ones(avatar_canvas_size, dtype=np.uint8) * 240
            draw_avatar(avatar, exag_shapes, yaw, pitch, roll)

            # ── Debug overlay on webcam frame ───────────────────────────
            cv2.putText(
                frame,
                f"Yaw:{yaw:+.1f} Pitch:{pitch:+.1f} Roll:{roll:+.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            for i, (k, v) in enumerate(blend_shapes.items()):
                cv2.putText(
                    frame,
                    f"{k}: {v:.2f}",
                    (10, 60 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 255),
                    1,
                )

            cv2.imshow("Avatar", avatar)
        else:
            cv2.putText(
                frame,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
