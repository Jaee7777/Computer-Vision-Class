import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 3D model points in mm (generic face)
MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],  # nose tip        lm 1
        [0.0, -63.6, -12.5],  # chin            lm 152
        [-43.3, 32.7, -26.0],  # left eye corner lm 33
        [43.3, 32.7, -26.0],  # right eye corner lm 263
        [-28.9, -28.9, -24.1],  # left mouth      lm 61
        [28.9, -28.9, -24.1],  # right mouth     lm 291
    ],
    dtype=np.float64,
)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # Use MediaPipe's built-in connection sets for accurate region boundaries
        def conn_ids(conn_set):
            ids = set()
            for a, b in conn_set:
                ids.add(a)
                ids.add(b)
            return ids

        LIPS_IDS = conn_ids(mp_face_mesh.FACEMESH_LIPS)
        OVAL_IDS = conn_ids(mp_face_mesh.FACEMESH_FACE_OVAL)
        EYE_IDS = (
            conn_ids(mp_face_mesh.FACEMESH_LEFT_EYE)
            | conn_ids(mp_face_mesh.FACEMESH_RIGHT_EYE)
            | conn_ids(mp_face_mesh.FACEMESH_LEFT_EYEBROW)
            | conn_ids(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)
        )

        # Use normalized Y positions to split jaw vs upper on the face oval
        mouth_y = (lm[13].y + lm[14].y) / 2.0
        eye_y = (lm[33].y + lm[263].y) / 2.0

        for i, p in enumerate(lm):
            if i in LIPS_IDS or (i in OVAL_IDS and p.y > mouth_y):
                color = (0, 100, 220)  # orange — lips + lower jaw oval
            elif i in EYE_IDS or (i in OVAL_IDS and p.y <= mouth_y):
                color = (0, 255, 0)  # green  — eyes, brows, upper oval
            elif eye_y < p.y < mouth_y and abs(p.x - 0.5) < 0.18:
                color = (255, 200, 0)  # cyan   — nose (center, mid-face)
            else:
                color = (0, 255, 0)  # green  — remaining upper face
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, color, -1)

        # 2D image points for the 6 pose landmarks
        image_points = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in LANDMARK_IDS],
            dtype=np.float64,
        )

        # Camera matrix (approximation)
        cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, image_points, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            # Rodrigues: rotation vector -> rotation matrix -> Euler angles
            rmat, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            if sy > 1e-6:
                pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
                yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
                roll = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
            else:
                pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
                yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
                roll = 0.0

            # Draw pose axes at nose tip
            axis = np.float32([[50, 0, 0], [0, 50, 0], [0, 0, 50]])
            nose_2d, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, cam, dist)
            axis_2d, _ = cv2.projectPoints(axis, rvec, tvec, cam, dist)
            o = tuple(nose_2d[0].ravel().astype(int))
            cv2.line(frame, o, tuple(axis_2d[0].ravel().astype(int)), (0, 0, 255), 2)
            cv2.line(frame, o, tuple(axis_2d[1].ravel().astype(int)), (0, 255, 0), 2)
            cv2.line(frame, o, tuple(axis_2d[2].ravel().astype(int)), (255, 0, 0), 2)

            cv2.putText(
                frame,
                f"yaw:{yaw:+.0f} pitch:{pitch:+.0f} roll:{roll:+.0f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

    cv2.imshow("stage2", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
