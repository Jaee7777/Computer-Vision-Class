"""
dragon_core.py — shared pipeline for CS5330 dragon face project
Imported by stage3_clean.py and stage4_dragon.py

Contains:
  - CORRESPONDENCES  : landmark → Smaug target map
  - get_landmarks()  : MediaPipe face mesh extraction
  - get_head_pose()  : solvePnP head pose estimation
  - compute_targets(): expression drivers + 3D warp + projection
"""

import cv2
import mediapipe as mp
import numpy as np

# ─────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────

_face_mesh = None

def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _face_mesh

# ─────────────────────────────────────────────
# CORRESPONDENCE MAP
# lm_idx: (name, nx, ny, nz, color)
# nx/ny/nz in IOD units, origin = midpoint of eyes
# color is used on both webcam and dragon panels
# ─────────────────────────────────────────────

CORRESPONDENCES = {
    # Skull (8)
    10:  ("skull_top",    0.0,  -1.10, -0.20, (0, 255, 0)),
    108: ("skull_FL",    -0.5,  -0.85, -0.10, (0, 255, 0)),
    337: ("skull_FR",     0.5,  -0.85, -0.10, (0, 255, 0)),
    234: ("skull_L",     -1.0,  -0.20, -0.30, (0, 255, 0)),
    454: ("skull_R",      1.0,  -0.20, -0.30, (0, 255, 0)),
    162: ("skull_BL",    -0.9,   0.00, -0.40, (0, 255, 0)),
    389: ("skull_BR",     0.9,   0.00, -0.40, (0, 255, 0)),
    151: ("forehead",     0.0,  -0.80, -0.10, (0, 255, 0)),
    # Brows (4)
    70:  ("brow_L_in",   -0.70, -0.62,  0.00, (0, 255, 0)),
    107: ("brow_L_out",  -1.05, -0.65,  0.00, (0, 255, 0)),
    300: ("brow_R_in",    0.70, -0.62,  0.00, (0, 255, 0)),
    336: ("brow_R_out",   1.05, -0.65,  0.00, (0, 255, 0)),
    # Left eye — full contour for blink (6)
    33:  ("eye_L_in",    -0.70, -0.40,  0.00, (0, 200, 255)),
    133: ("eye_L_out",   -1.00, -0.40,  0.00, (0, 200, 255)),
    159: ("eye_L_top",   -0.85, -0.50,  0.00, (0, 200, 255)),
    145: ("eye_L_bot",   -0.85, -0.30,  0.00, (0, 200, 255)),
    158: ("eye_L_tL",    -0.76, -0.48,  0.00, (0, 200, 255)),
    153: ("eye_L_bL",    -0.76, -0.32,  0.00, (0, 200, 255)),
    # Right eye — full contour for blink (6)
    263: ("eye_R_in",     0.70, -0.40,  0.00, (0, 200, 255)),
    362: ("eye_R_out",    1.00, -0.40,  0.00, (0, 200, 255)),
    386: ("eye_R_top",    0.85, -0.50,  0.00, (0, 200, 255)),
    374: ("eye_R_bot",    0.85, -0.30,  0.00, (0, 200, 255)),
    385: ("eye_R_tR",     0.76, -0.48,  0.00, (0, 200, 255)),
    380: ("eye_R_bR",     0.76, -0.32,  0.00, (0, 200, 255)),
    # Nose → snout (5)
    6:   ("snout_bridge", 0.00, -0.10,  0.30, (255, 200, 0)),
    168: ("snout_top",    0.00,  0.10,  0.55, (255, 200, 0)),
    64:  ("snout_L",     -0.80,  0.20,  0.50, (255, 200, 0)),
    294: ("snout_R",      0.80,  0.20,  0.50, (255, 200, 0)),
    1:   ("snout_tip",    0.00,  0.65,  0.90, (255, 200, 0)),
    # Mouth (6)
    61:  ("mouth_L",     -0.80,  0.55,  0.35, (0, 100, 220)),
    291: ("mouth_R",      0.80,  0.55,  0.35, (0, 100, 220)),
    82:  ("upper_lip_L", -0.40,  0.40,  0.60, (0, 100, 220)),
    312: ("upper_lip_R",  0.40,  0.40,  0.60, (0, 100, 220)),
    13:  ("upper_lip",    0.00,  0.40,  0.60, (0, 100, 220)),
    14:  ("lower_lip",    0.00,  0.70,  0.35, (0, 100, 220)),
    # Jaw (7)
    172: ("hinge_L",     -1.10,  0.30, -0.50, (0, 100, 220)),
    397: ("hinge_R",      1.10,  0.30, -0.50, (0, 100, 220)),
    136: ("jaw_LL",      -0.80,  0.65,  0.00, (0, 100, 220)),
    365: ("jaw_RR",       0.80,  0.65,  0.00, (0, 100, 220)),
    176: ("jaw_L",       -0.50,  0.80,  0.10, (0, 100, 220)),
    400: ("jaw_R",        0.50,  0.80,  0.10, (0, 100, 220)),
    152: ("jaw_tip",      0.00,  0.90,  0.20, (0, 100, 220)),
}

# Precomputed name → color lookup
NAME_TO_COLOR = {v[0]: v[4] for v in CORRESPONDENCES.values()}

# ─────────────────────────────────────────────
# EXPRESSION DRIVER CONSTANTS  (tune here)
# ─────────────────────────────────────────────

JAW_GAMMA    = 0.6    # < 1 = snappy cartoon feel
MAX_JAW_DROP = 0.5    # max jaw drop in IOD units
EAR_OPEN     = 0.28   # EAR value for a fully open eye
EYE_HALF     = 0.11   # half eye height in IOD units

# Which target names are driven by each expression
JAW_NAMES  = {"lower_lip","jaw_tip","jaw_L","jaw_R","jaw_LL","jaw_RR",
               "hinge_L","hinge_R","mouth_L","mouth_R"}
EYE_L_TOP  = {"eye_L_top", "eye_L_tL"}
EYE_L_BOT  = {"eye_L_bot", "eye_L_bL"}
EYE_R_TOP  = {"eye_R_top", "eye_R_tR"}
EYE_R_BOT  = {"eye_R_bot", "eye_R_bR"}

# ─────────────────────────────────────────────
# STAGE 1 — LANDMARK EXTRACTION
# ─────────────────────────────────────────────

def get_landmarks(frame):
    """Run MediaPipe on a BGR frame. Returns (N,2) float32 pixel coords or None."""
    h, w    = frame.shape[:2]
    results = _get_face_mesh().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

# ─────────────────────────────────────────────
# STAGE 2 — HEAD POSE
# ─────────────────────────────────────────────

# Generic 3D face model (mm) matching POSE_IDS below
_MODEL_POINTS = np.array([
    [  0.0,   0.0,   0.0],   # nose tip
    [  0.0, -63.6, -12.5],   # chin
    [-43.3,  32.7, -26.0],   # left eye corner
    [ 43.3,  32.7, -26.0],   # right eye corner
    [-28.9, -28.9, -24.1],   # left mouth corner
    [ 28.9, -28.9, -24.1],   # right mouth corner
], dtype=np.float64)
_POSE_IDS = [1, 152, 33, 263, 61, 291]

def get_head_pose(lm, frame_shape):
    """Return (yaw, pitch, roll) in degrees via solvePnP + Rodrigues."""
    h, w    = frame_shape[:2]
    img_pts = np.array([[lm[i][0], lm[i][1]] for i in _POSE_IDS], dtype=np.float64)
    cam     = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_POINTS, img_pts, cam,
                                np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy      = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2( rmat[2,1], rmat[2,2]))
        yaw   = np.degrees(np.arctan2(-rmat[2,0], sy))
        roll  = np.degrees(np.arctan2( rmat[1,0], rmat[0,0]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[1,2], rmat[1,1]))
        yaw   = np.degrees(np.arctan2(-rmat[2,0], sy))
        roll  = 0.0
    # Normalize flipped angles
    if abs(pitch) > 90: pitch = pitch + 180 if pitch < 0 else pitch - 180
    if abs(roll)  > 90: roll  = roll  + 180 if roll  < 0 else roll  - 180
    return yaw, pitch, roll

# ─────────────────────────────────────────────
# STAGE 3 — COMPUTE TARGETS
# ─────────────────────────────────────────────

def _build_R(yaw, pitch, roll):
    """Build 3D rotation matrix from Euler angles (degrees)."""
    yr, pr, rr = np.radians(yaw), np.radians(pitch), np.radians(roll)
    Ry = np.array([[ np.cos(yr), 0, np.sin(yr)],
                   [ 0,          1, 0          ],
                   [-np.sin(yr), 0, np.cos(yr)]])
    Rx = np.array([[1, 0,           0          ],
                   [0, np.cos(pr), -np.sin(pr) ],
                   [0, np.sin(pr),  np.cos(pr) ]])
    Rz = np.array([[ np.cos(rr), -np.sin(rr), 0],
                   [ np.sin(rr),  np.cos(rr),  0],
                   [ 0,           0,            1]])
    return Rz @ Rx @ Ry

def compute_targets(lm, yaw, pitch, roll, frame_shape):
    """
    Map each CORRESPONDENCES landmark to its 2D Smaug target position.

    Steps:
      1. Measure expression scalars (jaw open, blink L/R) from landmarks
      2. Apply scalar offsets to template Y coordinates
      3. Rotate 3D template points by head pose
      4. Perspective-project onto the canvas

    Returns:
      targets      : dict { name: np.array([x, y]) }
      mouth_open   : float 0-1
      eye_open_L   : float 0-1
      eye_open_R   : float 0-1
    """
    h, w   = frame_shape[:2]
    iod    = float(np.linalg.norm(lm[33] - lm[263]))
    center = (lm[33] + lm[263]) / 2.0

    # Jaw
    mouth_open = float(np.clip(
        (np.linalg.norm(lm[13] - lm[14]) / iod - 0.15) / 0.45, 0.0, 1.0))
    jaw_drop = (mouth_open ** JAW_GAMMA) * MAX_JAW_DROP  # in IOD units

    # Blink
    ear_L = float(np.linalg.norm(lm[159]-lm[145])) / (float(np.linalg.norm(lm[33]-lm[133]))+1e-6)
    ear_R = float(np.linalg.norm(lm[386]-lm[374])) / (float(np.linalg.norm(lm[263]-lm[362]))+1e-6)
    eye_open_L = float(np.clip(ear_L / EAR_OPEN, 0.0, 1.0))
    eye_open_R = float(np.clip(ear_R / EAR_OPEN, 0.0, 1.0))
    blink_L = (1.0 - eye_open_L) * EYE_HALF
    blink_R = (1.0 - eye_open_R) * EYE_HALF

    R     = _build_R(yaw, pitch, roll)
    focal = float(w)
    tz    = focal * 2.0   # virtual camera distance
    targets = {}

    for lm_idx, (name, nx, ny, nz, _) in CORRESPONDENCES.items():
        # Apply expression offset to Y
        target_ny = ny
        if   name in JAW_NAMES : target_ny += jaw_drop
        elif name in EYE_L_TOP : target_ny += blink_L
        elif name in EYE_L_BOT : target_ny -= blink_L
        elif name in EYE_R_TOP : target_ny += blink_R
        elif name in EYE_R_BOT : target_ny -= blink_R

        # 3D rotate (negate nx to match mirrored webcam) + perspective project
        p3      = np.array([-nx * iod, target_ny * iod, nz * iod], dtype=np.float64)
        rotated = R @ p3
        z       = max(rotated[2] + tz, 1e-3)
        targets[name] = np.array([
            center[0] + focal * rotated[0] / z,
            center[1] + focal * rotated[1] / z,
        ], dtype=np.float32)

    return targets, mouth_open, eye_open_L, eye_open_R
