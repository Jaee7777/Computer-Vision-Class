"""
stage3.py — skeleton wireframe with color-coded correspondences
Left panel : webcam with colored source landmarks
Right panel: Smaug skeleton driven by face tracking
"""

import cv2
import numpy as np
from dragon_core import (
    CORRESPONDENCES,
    NAME_TO_COLOR,
    get_landmarks,
    get_head_pose,
    compute_targets,
)

# ─────────────────────────────────────────────
# SKELETON BONES
# ─────────────────────────────────────────────

BONES = [
    # Skull
    ("skull_top", "skull_FL"),
    ("skull_top", "skull_FR"),
    ("skull_FL", "skull_L"),
    ("skull_FR", "skull_R"),
    ("skull_L", "skull_BL"),
    ("skull_R", "skull_BR"),
    ("skull_BL", "hinge_L"),
    ("skull_BR", "hinge_R"),
    ("skull_top", "forehead"),
    # Brows
    ("brow_L_in", "brow_L_out"),
    ("brow_R_in", "brow_R_out"),
    ("brow_L_in", "eye_L_in"),
    ("brow_R_in", "eye_R_in"),
    # Left eye
    ("eye_L_in", "eye_L_tL"),
    ("eye_L_tL", "eye_L_top"),
    ("eye_L_top", "eye_L_out"),
    ("eye_L_in", "eye_L_bL"),
    ("eye_L_bL", "eye_L_bot"),
    ("eye_L_bot", "eye_L_out"),
    # Right eye
    ("eye_R_in", "eye_R_tR"),
    ("eye_R_tR", "eye_R_top"),
    ("eye_R_top", "eye_R_out"),
    ("eye_R_in", "eye_R_bR"),
    ("eye_R_bR", "eye_R_bot"),
    ("eye_R_bot", "eye_R_out"),
    # Snout
    ("eye_L_in", "snout_bridge"),
    ("eye_R_in", "snout_bridge"),
    ("snout_bridge", "snout_top"),
    ("snout_top", "snout_L"),
    ("snout_top", "snout_R"),
    ("snout_L", "snout_tip"),
    ("snout_R", "snout_tip"),
    # Upper jaw
    ("snout_L", "mouth_L"),
    ("snout_R", "mouth_R"),
    ("mouth_L", "upper_lip_L"),
    ("mouth_R", "upper_lip_R"),
    ("upper_lip_L", "upper_lip"),
    ("upper_lip_R", "upper_lip"),
    ("upper_lip", "snout_tip"),
    # Lower jaw
    ("mouth_L", "hinge_L"),
    ("mouth_R", "hinge_R"),
    ("hinge_L", "hinge_R"),
    ("hinge_L", "jaw_LL"),
    ("hinge_R", "jaw_RR"),
    ("jaw_LL", "jaw_L"),
    ("jaw_RR", "jaw_R"),
    ("jaw_L", "jaw_tip"),
    ("jaw_R", "jaw_tip"),
    ("mouth_L", "lower_lip"),
    ("mouth_R", "lower_lip"),
    ("lower_lip", "jaw_tip"),
]

# ─────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────


def draw_webcam(frame, lm):
    for pt in lm:
        cv2.circle(frame, tuple(pt.astype(int)), 1, (60, 60, 60), -1)
    for lm_idx, (name, *_, color) in CORRESPONDENCES.items():
        cv2.circle(frame, tuple(lm[lm_idx].astype(int)), 4, color, -1)


def draw_skeleton(canvas, targets):
    for a, b in BONES:
        if a in targets and b in targets:
            cv2.line(
                canvas,
                tuple(targets[a].astype(int)),
                tuple(targets[b].astype(int)),
                NAME_TO_COLOR.get(a, (150, 150, 150)),
                1,
            )
    for name, pos in targets.items():
        cv2.circle(
            canvas,
            tuple(pos.astype(int)),
            5,
            NAME_TO_COLOR.get(name, (150, 150, 150)),
            -1,
        )


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
        draw_skeleton(canvas, targets)

        cv2.putText(
            canvas,
            f"jaw:{mouth*100:.0f}%  eyeL:{eye_L*100:.0f}%  eyeR:{eye_R*100:.0f}%",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )

    cv2.imshow("stage3", np.hstack([frame, canvas]))
    if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
        break

cap.release()
cv2.destroyAllWindows()
