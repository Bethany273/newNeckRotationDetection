import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ---------- Parameters ----------
R_AT_90 = 0.6     # Expected proportion ratio at 90 deg yaw (for reference)
VIS_THRESH = 0.5    # Visibility threshold for landmarks
CALIB_SECONDS = 3.0 # Calibration duration in seconds
GAMMA = 0.5         # (Unused now, kept if you want nonlinear scaling later)
EPS = 1e-6

# ---------- Globals ----------
proportion_ear_shoulder = 0.0
timer_started = False
start_time = 0.0
calibration_shoulder = []
calibration_ear = []
calibration_nose_offset = []
calibration_complete = False
baseline_nose_offset = 0.0
calibration_nose_x = []
baseline_nose_x = 0.0

# ---------- Utility Functions ----------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def l2(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])


# ---------- Setup ----------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    raise SystemExit

print("Camera opened successfully. Press 'q' to quit.")
print("Face forward for calibration. Keep shoulders visible.")


# ---------- Main Loop ----------
with mp_pose.Pose(min_detection_confidence=0.5,
min_tracking_confidence=0.5,
model_complexity=1) as pose:
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

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        shoulder_dist = None
        ear_dist = None
        landmarks_ok = False
        nose_to_shoulder = 0.0

        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                LSH = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                RSH = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                LEAR = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                REAR = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
                NOSE = landmarks[mp_pose.PoseLandmark.NOSE.value]

                visible = (
                    LSH.visibility > VIS_THRESH and RSH.visibility > VIS_THRESH and
                    LEAR.visibility > VIS_THRESH and REAR.visibility > VIS_THRESH
                )

                if visible:
                    lsh_px = (LSH.x * img_w, LSH.y * img_h)
                    rsh_px = (RSH.x * img_w, RSH.y * img_h)
                    lear_px = (LEAR.x * img_w, LEAR.y * img_h)
                    rear_px = (REAR.x * img_w, REAR.y * img_h)
                    nose_px = (NOSE.x * img_w, NOSE.y * img_h)

                    mid_shoulder_x = (lsh_px[0] + rsh_px[0]) / 2.0
                    ear_dist = l2(lear_px, rear_px)
                    shoulder_dist = l2(lsh_px, rsh_px)
                    nose_to_shoulder = (nose_px[0] - mid_shoulder_x) / (shoulder_dist + EPS)
                    landmarks_ok = True

            except Exception as e:
                cv2.putText(image, f"Landmark error: {e}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


        # ---------- Calibration ----------
        if landmarks_ok and not calibration_complete:
            if not timer_started:
                timer_started = True
                start_time = time.time()
                calibration_shoulder.clear()
                calibration_ear.clear()
                calibration_nose_offset.clear()

            elapsed = time.time() - start_time
            if elapsed < CALIB_SECONDS:
                calibration_shoulder.append(shoulder_dist)
                calibration_ear.append(ear_dist)
                calibration_nose_offset.append(nose_to_shoulder)
                nose_offset_px = nose_px[0] - mid_shoulder_x
                calibration_nose_x.append(nose_offset_px)

                cv2.putText(
                    image,
                    f"Calibrating... Face forward {CALIB_SECONDS - elapsed:.1f}s",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
                )
            else:
                if calibration_shoulder and calibration_ear and calibration_nose_offset:
                    valid_props = []
                    for cs, ce in zip(calibration_shoulder, calibration_ear):
                        if cs and ce and cs > 0 and ce > 0:
                            valid_props.append(ce / cs)
                    if valid_props:
                        proportion_ear_shoulder = float(np.mean(valid_props))
                        baseline_nose_offset = float(np.mean(calibration_nose_offset))
                        calibration_complete = True
                        timer_started = False
                        print(f"Calibration complete. Avg proportion px: {proportion_ear_shoulder:.3f}, ")
                        print(f"Nose offset baseline: {baseline_nose_offset:.3f}")
                    else:
                        timer_started = False
                else:
                    timer_started = False

        elif not landmarks_ok and not calibration_complete:
            cv2.putText(image, "Ensure face and shoulders visible for calibration",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)


        if calibration_complete:
            try:
                if calibration_nose_x:
                    baseline_nose_x = float(np.mean(calibration_nose_x))
                else:
                    baseline_nose_x = 0.0

                if landmarks_ok:
                    delta_offset = nose_to_shoulder - baseline_nose_offset

                    SCALE = 2.0
                    GAMMA = 1.8
                    t = clamp(abs(delta_offset) * SCALE, 0.0, 1.0)
                    t_mapped = t ** GAMMA
                    shoulder_angle = 90.0 * t_mapped
                    if delta_offset < 0:
                        shoulder_angle = -shoulder_angle

                    # compensate for head shift (if ears visible)
                    if all(lm.visibility > VIS_THRESH for lm in [REAR, LEAR]):
                        ear_asymmetry = (LEAR.x - REAR.x)
                        correction_factor = 1.0 - 0.7 * abs(ear_asymmetry)
                        correction_factor = clamp(correction_factor, 0.5, 1.0)
                    else:
                        correction_factor = 1.0

                    shoulder_angle *= correction_factor

                    cv2.putText(image, f"Head Rotation: {shoulder_angle:+.1f} deg",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

                    # compute and show translation (head shift)
                    head_shift_px = nose_px[0] - mid_shoulder_x
                    delta_shift = head_shift_px - baseline_nose_x
                    head_shift_ratio = delta_shift / (shoulder_dist + EPS)
                    real_shift_cm = head_shift_ratio * 40.0

                    cv2.putText(image, f"Head Shift: {real_shift_cm:+.1f} cm",
                                (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "Hold still: landmarks not visible",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print("Error during post-calibration:", e)

        # ---------- Display Instructions ----------
        if not calibration_complete:
            cv2.putText(image, "Face forward. Keep ears and shoulders visible.",
                        (30, img_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(image, "Calibration starts automatically.",
                        (30, img_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.imshow('Head-Body Rotation Estimation', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
