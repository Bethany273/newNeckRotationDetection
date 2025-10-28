import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ---------- Parameters ----------
# Expected distance ratio at ~90° yaw versus frontal.
# Tune this empirically for your setup (0.3–0.5 is a common range).
R_AT_90 = 0.55

# Visibility threshold to accept a landmark
VIS_THRESH = 0.5

# Calibration duration in seconds
CALIB_SECONDS = 3.0


timer_started = False
start_time = 0.0
calibration_shoulder = []
calibration_ear = []
calibration_complete = False
avg_shoulder = 0.0
avg_ear = 0.0


def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def linear_ratio_to_angle_deg(ratio, r_at_90):
    # Map ratio: 1.0 -> 0 deg, r_at_90 -> 90 deg (linear interpolation, clamped)
    if r_at_90 >= 1.0:
        return 0.0
    t = (1.0 - ratio) / (1.0 - r_at_90)
    t = clamp(t, 0.0, 1.0)
    return 90.0 * t

def l2(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    raise SystemExit

print("Camera opened successfully. Press 'q' to quit.")
print("Face forward for calibration. Keep shoulders visible.")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        img_h, img_w = image.shape[:2]

        # Draw landmarks for context
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        shoulder_dist = None
        ear_dist = None
        landmarks_ok = False

        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark

                LSH = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                RSH = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                LEAR = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                REAR = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

                visible = (
                    LSH.visibility > VIS_THRESH and RSH.visibility > VIS_THRESH and
                    LEAR.visibility > VIS_THRESH and REAR.visibility > VIS_THRESH
                )

                if visible:
                    lsh_px = (LSH.x * img_w, LSH.y * img_h)
                    rsh_px = (RSH.x * img_w, RSH.y * img_h)
                    lear_px = (LEAR.x * img_w, LEAR.y * img_h)
                    rear_px = (REAR.x * img_w, REAR.y * img_h)

                    shoulder_dist = l2(lsh_px, rsh_px)
                    ear_dist = l2(lear_px, rear_px)
                    landmarks_ok = True

            except Exception as e:
                # Keep running; show a message on screen
                cv2.putText(image, f"Landmark error: {e}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # Calibration logic
        if landmarks_ok and not calibration_complete:
            if not timer_started:
                timer_started = True
                start_time = time.time()
                calibration_shoulder.clear()
                calibration_ear.clear()

            elapsed = time.time() - start_time
            if elapsed < CALIB_SECONDS:
                calibration_shoulder.append(shoulder_dist)
                calibration_ear.append(ear_dist)
                cv2.putText(
                    image,
                    f"Calibrating... Face forward {CALIB_SECONDS - elapsed:.1f}s",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
                )
            else:
                # Finish calibration
                if calibration_shoulder and calibration_ear:
                    avg_shoulder = float(np.mean(calibration_shoulder))
                    avg_ear = float(np.mean(calibration_ear))
                    calibration_complete = True
                    timer_started = False
                    print(f"Calibration complete. Avg shoulder px: {avg_shoulder:.2f}, Avg ear px: {avg_ear:.2f}")
                else:
                    # Restart calibration if we collected nothing
                    timer_started = False

        elif not landmarks_ok and not calibration_complete:
            # Prompt user to get into frame for calibration
            cv2.putText(image, "Ensure face and shoulders visible for calibration",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        # After calibration: compute angles if current frame is valid
        if calibration_complete:
            if landmarks_ok and avg_shoulder > 0 and avg_ear > 0:
                head_ratio = ear_dist / avg_ear
                body_ratio = shoulder_dist / avg_shoulder

                head_angle = linear_ratio_to_angle_deg(head_ratio, R_AT_90)   # 0–90
                body_angle = linear_ratio_to_angle_deg(body_ratio, R_AT_90)   # 0–90

                shoulder_angle = clamp(abs(head_angle - body_angle), 0.0, 90.0)

                cv2.putText(image, f"Shoulder Angle: {shoulder_angle:.1f} deg",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Head Angle: {head_angle:.1f} deg",
                            (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f"Body Angle: {body_angle:.1f} deg",
                            (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Hold still: landmarks not visible",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Display basic instructions if not calibrated yet
        if not calibration_complete:
            cv2.putText(image, "Face forward. Keep ears and shoulders visible.",
                        (30, img_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(image, "Calibration starts automatically.",
                        (30, img_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        # Show image
        cv2.imshow('Head-Body Rotation Estimation', image)

        # Quit on 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()