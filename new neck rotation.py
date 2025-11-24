import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque

# ---------- Parameters ----------
R_AT_90 = 0.6     # Expected proportion ratio at 90 deg yaw (for reference)
VIS_THRESH = 0.5    # Visibility threshold for landmarks
CALIB_SECONDS = 3.0 # Calibration duration in seconds        # (Unused now, kept if you want nonlinear scaling later)
EPS = 1e-6
MAX_ROTATION_ANGLE = 80.0  # maximum rotation angle (degrees)
SHOULDER_SIGN_INVERT = 1  # can be flipped to -1 if detected shoulder sign is reversed
SENSITIVITY = 7.0  # multiplier to make reported rotation more sensitive (raised further for head)
SHOULDER_GAIN = 7.0  # multiplier to increase shoulder contribution when compensating (increased further)
DEBUG_MODE = False  # Set to True to display detailed shoulder direction diagnostics
SHOULDER_SIGN_HISTORY_LEN = 9  # number of frames to stabilize shoulder sign detection
SHOULDER_SMOOTH_ALPHA = 0.35  # exponential smoothing for shoulder delta (moderate smoothing for stability)

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
calibration_mid_shoulder = []
baseline_mid_shoulder = 0.0
baseline_shoulder_dist = 0.0
calibration_shoulder_xdiff = []
baseline_shoulder_xdiff = 0.0
# not using torso hip landmarks; shoulder-only compensation
# no smoothing; use instantaneous angles for head and shoulders
calibration_head_angles = []
calibration_hip_angles = []  # collect hip angles to establish body baseline
calibration_shoulder_angles = []
baseline_head_angle = 0.0
baseline_hip_angle = 0.0  # body orientation baseline
baseline_shoulder_angle = 0.0
# ring buffer to smooth the detected shoulder rotation direction
sh_sign_history = deque(maxlen=SHOULDER_SIGN_HISTORY_LEN)
prev_sh_delta_smoothed = 0.0
mid_shoulder_direction = "CENTER"

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
                LHI = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                RHI = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

                visible = (
                    LSH.visibility > VIS_THRESH and RSH.visibility > VIS_THRESH and
                    LEAR.visibility > VIS_THRESH and REAR.visibility > VIS_THRESH and
                    LHI.visibility > VIS_THRESH and RHI.visibility > VIS_THRESH
                )

                if visible:
                    lsh_px = (LSH.x * img_w, LSH.y * img_h)
                    rsh_px = (RSH.x * img_w, RSH.y * img_h)
                    lear_px = (LEAR.x * img_w, LEAR.y * img_h)
                    rear_px = (REAR.x * img_w, REAR.y * img_h)
                    nose_px = (NOSE.x * img_w, NOSE.y * img_h)
                    lhi_px = (LHI.x * img_w, LHI.y * img_h)
                    rhi_px = (RHI.x * img_w, RHI.y * img_h)

                    mid_shoulder_x = (lsh_px[0] + rsh_px[0]) / 2.0
                    mid_shoulder_y = (lsh_px[1] + rsh_px[1]) / 2.0
                    mid_hip_x = (lhi_px[0] + rhi_px[0]) / 2.0
                    mid_hip_y = (lhi_px[1] + rhi_px[1]) / 2.0
                    mid_shoulder_y = (lsh_px[1] + rsh_px[1]) / 2.0
                    # Draw midpoints for shoulder and hip and determine simple direction by x-offset
                    MID_DIR_THRESHOLD_PX = 8  # pixels tolerance to avoid jitter
                    mid_sh_pt = (int(mid_shoulder_x), int(mid_shoulder_y))
                    mid_hip_pt = (int(mid_hip_x), int(mid_hip_y))
                    # draw points and connecting line
                    cv2.circle(image, mid_sh_pt, 6, (0, 255, 0), -1)  # mid-shoulder (green)
                    cv2.circle(image, mid_hip_pt, 6, (255, 0, 0), -1)  # mid-hip (blue)
                    cv2.line(image, mid_sh_pt, mid_hip_pt, (200, 200, 200), 1)
                    # determine simple mid-based shoulder direction
                    dx_mid = mid_shoulder_x - mid_hip_x
                    if dx_mid < -MID_DIR_THRESHOLD_PX:
                        mid_shoulder_direction = "LEFT"
                    elif dx_mid > MID_DIR_THRESHOLD_PX:
                        mid_shoulder_direction = "RIGHT"
                    else:
                        mid_shoulder_direction = "CENTER"
                    # show on screen
                    cv2.putText(image, f"MidDir: {mid_shoulder_direction}", (int(mid_shoulder_x)+10, int(mid_shoulder_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                    # (shoulder asymmetry not needed; we'll use midpoint movement)
                    ear_dist = l2(lear_px, rear_px)
                    shoulder_dist = l2(lsh_px, rsh_px)
                    nose_to_shoulder = (nose_px[0] - mid_shoulder_x) / (shoulder_dist + EPS)
                    # nose_to_shoulder still used for head mapping; no torso used here
                    # store shoulder x-diff for potential sign autodetect (r.x - l.x)
                    shoulder_xdiff = (rsh_px[0] - lsh_px[0])
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
                # capture shoulder x-difference for sign autodetection
                calibration_shoulder_xdiff.append(shoulder_xdiff)
                nose_offset_px = nose_px[0] - mid_shoulder_x
                calibration_nose_x.append(nose_offset_px)
                calibration_mid_shoulder.append(mid_shoulder_x)
                # collect angular baselines: head vector and shoulder vector
                head_vec_y = nose_px[1] - mid_shoulder_y
                head_vec_x = nose_px[0] - mid_shoulder_x
                head_ang = math.atan2(head_vec_y, head_vec_x)
                calibration_head_angles.append(head_ang)

                # collect hip angle (body orientation baseline)
                hip_vec_y = rhi_px[1] - lhi_px[1]
                hip_vec_x = rhi_px[0] - lhi_px[0]
                hip_ang = math.atan2(hip_vec_y, hip_vec_x)
                calibration_hip_angles.append(hip_ang)

                sh_vec_y = rsh_px[1] - lsh_px[1]
                sh_vec_x = rsh_px[0] - lsh_px[0]
                sh_ang = math.atan2(sh_vec_y, sh_vec_x)
                calibration_shoulder_angles.append(sh_ang)

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
                        if calibration_mid_shoulder:
                            baseline_mid_shoulder = float(np.mean(calibration_mid_shoulder))
                        if calibration_shoulder:
                            baseline_shoulder_dist = float(np.mean(calibration_shoulder))
                        if calibration_shoulder_xdiff:
                            baseline_shoulder_xdiff = float(np.mean(calibration_shoulder_xdiff))
                            # if the average x-diff is negative, landmark left/right are flipped -> invert shoulder sign
                            if baseline_shoulder_xdiff < 0.0:
                                SHOULDER_SIGN_INVERT = -1
                                print("Auto-inverted shoulder sign based on calibration x-diff")

                        # compute circular mean of collected angles
                        if calibration_head_angles:
                            sin_sum = float(np.mean([math.sin(a) for a in calibration_head_angles]))
                            cos_sum = float(np.mean([math.cos(a) for a in calibration_head_angles]))
                            baseline_head_angle = math.atan2(sin_sum, cos_sum)
                        if calibration_hip_angles:
                            sin_sum = float(np.mean([math.sin(a) for a in calibration_hip_angles]))
                            cos_sum = float(np.mean([math.cos(a) for a in calibration_hip_angles]))
                            baseline_hip_angle = math.atan2(sin_sum, cos_sum)
                        if calibration_shoulder_angles:
                            sin_sum = float(np.mean([math.sin(a) for a in calibration_shoulder_angles]))
                            cos_sum = float(np.mean([math.cos(a) for a in calibration_shoulder_angles]))
                            baseline_shoulder_angle = math.atan2(sin_sum, cos_sum)
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
                # baseline_nose_x unused; use normalized nose_to_shoulder baseline instead
                if landmarks_ok:
                    # head offset: normalized nose-to-shoulder difference from calibration baseline
                    head_offset = 0.0
                    if baseline_shoulder_dist and (baseline_nose_offset is not None):
                        head_offset = (nose_to_shoulder - baseline_nose_offset)

                    # shoulder offset: midpoint movement relative to baseline (camera frame) normalized by shoulder width
                    shoulder_offset = 0.0
                    if baseline_mid_shoulder and baseline_shoulder_dist:
                        shoulder_offset = (mid_shoulder_x - baseline_mid_shoulder) / (baseline_shoulder_dist + EPS)

                    # geometric mapping: use arctan of normalized offset to produce an angle
                    # atan is smoother and maps 0..inf to 0..pi/2; we scale by ANGLE_K
                    ANGLE_K = 3.5
                    def map_offset_to_angle(off):
                        # off is normalized by shoulder width; clamp small values
                        val = math.atan(ANGLE_K * off)
                        # normalize to [0, pi/2] then map to 0..MAX_ROTATION_ANGLE
                        angle = (val / (math.pi / 2.0)) * MAX_ROTATION_ANGLE
                        return angle

                    # compute absolute angular orientation of head and hips (in degrees)
                    head_vec_x = nose_px[0] - mid_shoulder_x
                    head_vec_y = nose_px[1] - mid_shoulder_y
                    head_ang_deg = math.degrees(math.atan2(head_vec_y, head_vec_x))

                    hip_vec_x = rhi_px[0] - lhi_px[0]
                    hip_vec_y = rhi_px[1] - lhi_px[1]
                    hip_ang_deg = math.degrees(math.atan2(hip_vec_y, hip_vec_x))

                    base_head_deg = math.degrees(baseline_head_angle) if baseline_head_angle else 0.0
                    base_hip_deg = math.degrees(baseline_hip_angle) if baseline_hip_angle else 0.0

                    def angle_diff(a, b):
                        d = (a - b + 180.0) % 360.0 - 180.0
                        return d

                    head_delta = angle_diff(head_ang_deg, base_head_deg)
                    hip_delta = angle_diff(hip_ang_deg, base_hip_deg)
                    
                    # compute shoulder rotation using two metrics:
                    # 1. x-diff change (right_x - left_x): when rotating clockwise (right), x-diff decreases (shoulders compress)
                    #    when rotating anticlockwise (left), x-diff increases (shoulders expand)
                    # Note: camera is mirrored, so we invert the x-diff sign to match user perception
                    current_sh_xdiff = rsh_px[0] - lsh_px[0]
                    # invert sign: positive (expanding) from camera = anticlockwise from user perspective (negative)
                    sh_xdiff_delta = -(current_sh_xdiff - baseline_shoulder_xdiff)
                    
                    # shoulder distance change: negative when distance decreases (rotating)
                    sh_dist_delta = (shoulder_dist - baseline_shoulder_dist) / (baseline_shoulder_dist + EPS)
                    
                    # combine both metrics: x-diff gives direction, distance change amplifies rotation magnitude
                    SH_XDIFF_GAIN = 0.8  # gain for x-diff contribution (reduced further for less sensitivity)
                    SH_DIST_GAIN = 0.4  # gain for distance-change contribution (reduced further to dampen shoulder)
                    
                    # base angle from x-diff (gives direction)
                    sh_ang_from_xdiff = sh_xdiff_delta * SH_XDIFF_GAIN
                    
                    # distance-based amplification: abs(distance_delta) amplifies the x-diff sign
                    # when shoulders rotate inward (distance decreases), abs(sh_dist_delta) is positive and amplifies
                    dist_amplify = max(1.0, 1.0 + abs(sh_dist_delta) * SH_DIST_GAIN)
                    
                    sh_ang_deg = sh_ang_from_xdiff * dist_amplify
                    
                    # shoulder rotation relative to body (hip): subtract hip rotation from shoulder rotation
                    # this removes body/torso rotation, isolating pure shoulder rotation within the body frame
                    sh_delta = sh_ang_deg - hip_delta
                    
                    # DEBUG: Print shoulder direction diagnostics if DEBUG_MODE enabled
                    if DEBUG_MODE:
                        print(f"\n--- Shoulder Direction Debug ---")
                        print(f"Right Shoulder X: {rsh_px[0]:.1f}, Left Shoulder X: {lsh_px[0]:.1f}")
                        print(f"Current x-diff (R-L): {current_sh_xdiff:.1f}")
                        print(f"Baseline x-diff: {baseline_shoulder_xdiff:.1f}")
                        print(f"X-diff delta (-(curr-base)): {sh_xdiff_delta:.1f}")
                        print(f"Shoulder x-diff angle (with gain): {sh_ang_from_xdiff:.1f}°")
                        print(f"Distance delta: {sh_dist_delta:.3f}, Amplify: {dist_amplify:.3f}")
                        print(f"Shoulder angle (before hip subtract): {sh_ang_deg:.1f}°")
                        print(f"Hip angle delta: {hip_delta:.1f}°")
                        print(f"Shoulder delta (after hip subtract): {sh_delta:.1f}°")
                        print(f"Sign interpretation: {'RIGHT' if sh_delta > 0 else 'LEFT' if sh_delta < 0 else 'CENTER'}")

                    # adjust shoulder contribution based on ear vs expected ear distance
                    shoulder_adjust = 0.0
                    try:
                        if proportion_ear_shoulder and shoulder_dist and ear_dist:
                            expected_ear = proportion_ear_shoulder * shoulder_dist
                            ratio_diff = (ear_dist - expected_ear) / (expected_ear + EPS)
                            ADJUST_GAIN = 3.5
                            MAX_ADJ = 8.0
                            shoulder_adjust = clamp(ratio_diff * ADJUST_GAIN, -MAX_ADJ, MAX_ADJ)
                    except Exception:
                        shoulder_adjust = 0.0

                    # apply a small exponential smoothing to shoulder delta to reduce sign noise
                    sh_delta_smoothed = (SHOULDER_SMOOTH_ALPHA * sh_delta) + ((1.0 - SHOULDER_SMOOTH_ALPHA) * prev_sh_delta_smoothed)
                    prev_sh_delta_smoothed = sh_delta_smoothed

                    # Ensure shoulder sign matches midpoint-based direction (mid_shoulder_direction)
                    # If mid-shoulder suggests RIGHT but computed shoulder delta is negative, flip sign, and vice versa.
                    try:
                        if mid_shoulder_direction == "RIGHT" and sh_delta_smoothed < 0.0:
                            sh_delta_smoothed = -sh_delta_smoothed
                        elif mid_shoulder_direction == "LEFT" and sh_delta_smoothed > 0.0:
                            sh_delta_smoothed = -sh_delta_smoothed
                    except Exception:
                        # if mid_shoulder_direction is not defined or other error, ignore
                        pass

                    # simple direct calculation: neck rotation = head rotation - shoulder rotation
                    # this naturally compensates for body rotation
                    neck_angle = head_delta - sh_delta_smoothed

                    # apply sensitivity multiplier
                    neck_angle = neck_angle * SENSITIVITY

                    # Ensure we never exceed configured max rotation
                    neck_angle = clamp(neck_angle, -MAX_ROTATION_ANGLE, MAX_ROTATION_ANGLE)

                    # color the angle text per thresholds (use absolute neck angle)
                    # NOTE: head/neck rotation display/commented out per user request
                    # angle_col = angle_color(abs(neck_angle))
                    # cv2.putText(image, f"Head Rotation: {neck_angle:+.1f} deg",
                    #             (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, angle_col, 2, cv2.LINE_AA)
                    # show head and shoulder delta values
                    cv2.putText(image, f"Head Δ: {head_delta:+.1f}°", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Shoulder Δ: {sh_delta_smoothed:+.1f}°", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)
                    
                    # DEBUG: Display shoulder direction details on screen if enabled
                    if DEBUG_MODE:
                        y_offset = 145
                        cv2.putText(image, f"X-diff: {current_sh_xdiff:.1f} (base: {baseline_shoulder_xdiff:.1f})", 
                                    (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1, cv2.LINE_AA)
                        cv2.putText(image, f"Sh Δ (raw): {sh_ang_deg:.1f}°, Hip Δ: {hip_delta:.1f}°", 
                                    (30, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 255), 1, cv2.LINE_AA)
                        direction_str = "RIGHT→" if sh_delta > 2 else "←LEFT" if sh_delta < -2 else "CENTER"
                        cv2.putText(image, f"Direction: {direction_str}", 
                                    (30, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2, cv2.LINE_AA)
                    

                    # Initialize globals (only first run)
                    if 'last_message' not in globals():
                        last_message = ""
                        message_time = 0.0

                    current_time = time.time()
                    
                    # Record max neck angles reached
                    if neck_angle > max_right:
                        max_right = neck_angle
                    if neck_angle < max_left:
                        max_left = neck_angle

                    # State transitions
                    if direction == "neutral":
                        if neck_angle > ANGLE_THRESHOLD:
                            direction = "right"
                            completed_sides.add("right")
                            cv2.putText(image, "Right rotation complete!", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                            last_message = "Right rotation complete!"
                            message_time = current_time

                        elif neck_angle < -ANGLE_THRESHOLD:
                            direction = "left"
                            completed_sides.add("left")
                            cv2.putText(image, "Left rotation complete!", (30, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                            last_message = "Left rotation complete! "
                            message_time = current_time


                    elif direction in ["right", "left"]:
                        if abs(neck_angle) < NEUTRAL_THRESHOLD:
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
