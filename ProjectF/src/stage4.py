"""
stage4.py — filled polygon dragon face rendering
Left panel : webcam with colored source landmarks
Right panel: Smaug face with filled surfaces driven by face tracking
"""

import cv2
import numpy as np
from dragon_core import (
    CORRESPONDENCES,
    get_landmarks,
    get_head_pose,
    compute_targets,
)

# ─────────────────────────────────────────────
# SURFACE PATCHES
# (point_names, fill_color, outline_color)
# Drawn back-to-front for correct layering
# ─────────────────────────────────────────────

SURFACES = [
    # Back skull — darkest, furthest back
    (
        ["skull_L", "skull_BL", "hinge_L", "hinge_R", "skull_BR", "skull_R"],
        (30, 55, 30),
        (50, 80, 50),
    ),
    # Upper skull
    (
        ["skull_top", "skull_FL", "skull_L", "skull_R", "skull_FR"],
        (40, 70, 40),
        (60, 100, 60),
    ),
    # Forehead band
    (
        ["skull_FL", "forehead", "skull_FR", "brow_R_in", "brow_L_in"],
        (45, 80, 45),
        (65, 110, 65),
    ),
    # Left cheek
    (
        [
            "skull_L",
            "skull_FL",
            "brow_L_out",
            "eye_L_out",
            "snout_L",
            "hinge_L",
            "skull_BL",
        ],
        (40, 75, 40),
        (60, 105, 60),
    ),
    # Right cheek
    (
        [
            "skull_R",
            "skull_FR",
            "brow_R_out",
            "eye_R_out",
            "snout_R",
            "hinge_R",
            "skull_BR",
        ],
        (40, 75, 40),
        (60, 105, 60),
    ),
    # Mid face — between eyes and snout bridge
    (
        ["brow_L_in", "brow_R_in", "eye_R_in", "snout_bridge", "eye_L_in"],
        (50, 90, 50),
        (70, 120, 70),
    ),
    # Snout top surface
    (
        ["snout_bridge", "snout_top", "snout_L", "snout_tip", "snout_R"],
        (55, 100, 55),
        (80, 140, 80),
    ),
    # Upper jaw — rigid, snout to mouth corners
    (
        [
            "snout_L",
            "mouth_L",
            "upper_lip_L",
            "upper_lip",
            "upper_lip_R",
            "mouth_R",
            "snout_R",
            "snout_tip",
        ],
        (50, 95, 50),
        (75, 130, 75),
    ),
    # Lower jaw — mobile, follows jaw drop
    (
        ["hinge_L", "jaw_LL", "jaw_L", "jaw_tip", "jaw_R", "jaw_RR", "hinge_R"],
        (35, 65, 35),
        (55, 90, 55),
    ),
    # Lower lip / chin
    (
        [
            "mouth_L",
            "lower_lip",
            "mouth_R",
            "jaw_LL",
            "jaw_L",
            "jaw_tip",
            "jaw_R",
            "jaw_RR",
        ],
        (40, 75, 40),
        (60, 105, 60),
    ),
    # Left brow ridge
    (
        ["brow_L_in", "brow_L_out", "eye_L_out", "eye_L_top", "eye_L_in"],
        (30, 55, 30),
        (50, 80, 50),
    ),
    # Right brow ridge
    (
        ["brow_R_in", "brow_R_out", "eye_R_out", "eye_R_top", "eye_R_in"],
        (30, 55, 30),
        (50, 80, 50),
    ),
    # Left eye socket — dark hollow
    (
        ["eye_L_in", "eye_L_tL", "eye_L_top", "eye_L_out", "eye_L_bot", "eye_L_bL"],
        (15, 25, 15),
        (80, 200, 80),
    ),
    # Right eye socket — dark hollow
    (
        ["eye_R_in", "eye_R_tR", "eye_R_top", "eye_R_out", "eye_R_bot", "eye_R_bR"],
        (15, 25, 15),
        (80, 200, 80),
    ),
    # Left eye slit — bright iris visible through socket
    (["eye_L_in", "eye_L_top", "eye_L_out", "eye_L_bot"], (0, 180, 220), (0, 220, 255)),
    # Right eye slit
    (["eye_R_in", "eye_R_top", "eye_R_out", "eye_R_bot"], (0, 180, 220), (0, 220, 255)),
]

# ─────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────


def draw_webcam(frame, lm):
    for pt in lm:
        cv2.circle(frame, tuple(pt.astype(int)), 1, (60, 60, 60), -1)
    for lm_idx, (name, *_, color) in CORRESPONDENCES.items():
        cv2.circle(frame, tuple(lm[lm_idx].astype(int)), 4, color, -1)


def draw_dragon(canvas, targets):
    # Filled surface patches
    for point_names, fill_color, outline_color in SURFACES:
        pts = [targets[n] for n in point_names if n in targets]
        if len(pts) < 3:
            continue
        poly = np.array(pts, dtype=np.int32)
        cv2.fillPoly(canvas, [poly], fill_color)
        cv2.polylines(canvas, [poly], isClosed=True, color=outline_color, thickness=1)

    # Nostril slits
    for side in ("snout_L", "snout_R"):
        if side in targets:
            cv2.ellipse(
                canvas,
                tuple(targets[side].astype(int)),
                (9, 4),
                0,
                0,
                360,
                (10, 20, 10),
                -1,
            )

    # Vertical pupil slits
    for in_pt, out_pt in (("eye_L_in", "eye_L_out"), ("eye_R_in", "eye_R_out")):
        if in_pt in targets and out_pt in targets:
            center = ((targets[in_pt] + targets[out_pt]) / 2).astype(int)
            cv2.ellipse(canvas, tuple(center), (5, 8), 0, 0, 360, (0, 40, 60), -1)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    canvas = np.zeros_like(frame)
    lm = get_landmarks(frame)

    if lm is not None:
        yaw, pitch, roll = get_head_pose(lm, frame.shape)
        targets, mouth, eye_L, eye_R = compute_targets(
            lm, yaw, pitch, roll, frame.shape
        )

        draw_webcam(frame, lm)
        draw_dragon(canvas, targets)

        cv2.putText(
            canvas,
            f"jaw:{mouth*100:.0f}%  yaw:{yaw:+.0f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 200, 120),
            1,
        )

    cv2.imshow("stage4 dragon", np.hstack([frame, canvas]))
    if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows()
