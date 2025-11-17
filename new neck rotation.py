import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ---------- Parameters ----------
R_AT_90 = 0.6     # Expected proportion ratio at 90 deg yaw (for reference)
VIS_THRESH = 0.5    # Visibility threshold for landmarks
CALIB_SECONDS = 3.0 # Calibration duration in seconds        # (Unused now, kept if you want nonlinear scaling later)
EPS = 1e-6
MAX_ROTATION_ANGLE = 80.0  # maximum rotation angle (degrees)

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
rep_max_left = []
rep_max_right = []
MAX_REPS = 5

# ---------- Repetition & Max Rotation Logic ----------

ANGLE_THRESHOLD = 30      # degrees to qualify as full turn
NEUTRAL_THRESHOLD = 10    # degrees considered neutral range
rep_count = 0
direction = "neutral"   # can be 'neutral', 'left', or 'right'
max_right = 0.0
max_left = 0.0
completed_sides = set()
MESSAGE_DURATION = 2.0

# ---------- Utility Functions ----------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def l2(p, q):
    return math.hypot(q[0] - p[0], q[1] - p[1])


def angle_color(angle_abs: float):
    """Return BGR color tuple for the given absolute angle using thresholds:
    red: <60, yellow: 60-70, green: >70
    """
    if angle_abs < 60.0:
        return (0, 0, 255)
    elif angle_abs < 70.0:
        return (0, 255, 255)
    else:
        return (0, 255, 0)


def show_report(avg_left: float, avg_right: float):
    """Create and display a simple report screen summarizing average rotations.
    avg_left and avg_right are absolute degrees.
    The screen waits for a keypress or 15 seconds before closing.
    """
    w, h = 800, 480
    report = np.ones((h, w, 3), dtype=np.uint8) * 240

    title = "Rotation Session Report"
    cv2.putText(report, title, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (50, 50, 50), 2, cv2.LINE_AA)

    # Left
    cv2.putText(report, f"Average Left Rotation: {avg_left:.1f} deg", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2, cv2.LINE_AA)
    color_l = angle_color(avg_left)
    cv2.rectangle(report, (500, 100), (740, 160), color_l, -1)
    # Right
    cv2.putText(report, f"Average Right Rotation: {avg_right:.1f} deg", (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2, cv2.LINE_AA)
    color_r = angle_color(avg_right)
    cv2.rectangle(report, (500, 190), (740, 250), color_r, -1)

    # Threshold legend
    cv2.putText(report, "Legend:", (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2, cv2.LINE_AA)
    cv2.rectangle(report, (30, 350), (80, 390), (0, 0, 255), -1)
    cv2.putText(report, "< 60 deg (red)", (95, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
    cv2.rectangle(report, (260, 350), (310, 390), (0, 255, 255), -1)
    cv2.putText(report, "60-70 deg (yellow)", (325, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
    cv2.rectangle(report, (520, 350), (570, 390), (0, 255, 0), -1)
    cv2.putText(report, "> 70 deg (green)", (585, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)

    cv2.putText(report, "Press any key to exit or wait 15s...", (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2, cv2.LINE_AA)

    cv2.imshow('Rotation Report', report)
    # wait for key or timeout
    key = cv2.waitKey(15000)
    # close report window
    cv2.destroyWindow('Rotation Report')


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

                    SCALE = 1.5
                    GAMMA = 1.8
                    t = clamp(abs(delta_offset) * SCALE, 0.0, 1.0)
                    t_mapped =1 - (1 - t)**GAMMA
                    t_mapped = t_mapped ** 1.5
                    shoulder_angle = MAX_ROTATION_ANGLE * t_mapped
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

                    # Ensure we never exceed configured max rotation
                    shoulder_angle = clamp(shoulder_angle, -MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE)

                    # color the angle text per thresholds (use absolute angle)
                    angle_col = angle_color(abs(shoulder_angle))
                    cv2.putText(image, f"Head Rotation: {shoulder_angle:+.1f} deg",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, angle_col, 2, cv2.LINE_AA)
                    
                    # Initialize globals (only first run)
                    if 'last_message' not in globals():
                        last_message = ""
                        message_time = 0.0

                    current_time = time.time()
                    
                    # Record max angles reached
                    if shoulder_angle > max_right:
                        max_right = shoulder_angle
                    if shoulder_angle < max_left:
                        max_left = shoulder_angle

                    # State transitions
                    if direction == "neutral":
                        if shoulder_angle > ANGLE_THRESHOLD:
                            direction = "right"
                            completed_sides.add("right")
                            cv2.putText(image, "Right rotation complete!", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                            last_message = "Right rotation complete!"
                            message_time = current_time

                        elif shoulder_angle < -ANGLE_THRESHOLD:
                            direction = "left"
                            completed_sides.add("left")
                            cv2.putText(image, "Left rotation complete!", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                            last_message = "Left rotation complete! "
                            message_time = current_time


                    elif direction in ["right", "left"]:
                        if abs(shoulder_angle) < NEUTRAL_THRESHOLD:
                            direction = "neutral"

                    # If both sides completed in this cycle → one repetition
                    if "left" in completed_sides and "right" in completed_sides and direction == "neutral":
                        rep_count += 1
                        last_message = f"L:{abs(max_left):.1f}deg, R:{abs(max_right):.1f}deg"
                        message_time = current_time
                        print(f"Repetition {rep_count} completed! Max L: {max_left:.1f}, Max R: {max_right:.1f}")
                        rep_max_left.append(max_left)
                        rep_max_right.append(max_right)
                        
                        # Display on screen
                        cv2.putText(image, f"Rep {rep_count} done!", (30, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

                        # Reset for next repetition
                        completed_sides.clear()
                        max_left = 0.0
                        max_right = 0.0
                        
                        if rep_count >= MAX_REPS:
                            # compute absolute average rotations for reporting
                            avg_left = abs(sum(rep_max_left) / len(rep_max_left))
                            avg_right = abs(sum(rep_max_right) / len(rep_max_right))

                            print("\n==== SESSION COMPLETE ====")
                            print(f"Average Left Rotation:  {avg_left:.1f}°")
                            print(f"Average Right Rotation: {avg_right:.1f}°")
                            print("==========================")

                            # show dedicated report screen (waits for key or timeout)
                            try:
                                show_report(avg_left, avg_right)
                            except Exception as e:
                                print("Failed to show report screen:", e)

                            break  # exit main loop

                    # Draw persistent message if within time window
                    if current_time - message_time < MESSAGE_DURATION:
                        cv2.putText(image, last_message, (30, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                        
                        
                    # Display counter on screen
                    cv2.putText(image, f"Reps: {rep_count}", (30, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, cv2.LINE_AA)
                    
                                        
                                        
                                        
                                        
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
