import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()


def angle_between_vectors(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1, v2)
    dot_product = np.clip(np.abs(dot_product), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            try:
                landmarks = results.pose_landmarks.landmark
                img_h, img_w, _ = image.shape

                # Get key points
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
                right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
                nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
                
                #calculating face direction vector 
                midppoint_ears = np.array([
                    (left_ear.x + right_ear.x)/2, 
                    (left_ear.y + right_ear.y)/2, 
                    (left_ear.z + right_ear.z)/2
                    ])
                
                nose = np.array(
                    [nose.x, 
                     nose.y,
                     nose.z
                     ])
                
                vec_face= nose-midppoint_ears
                
                vec_face_proj = np.array([
                    vec_face[0], 
                    0, 
                    vec_face[2]
                ])

                

                # Create vectors from normalized coordinates (3D for better accuracy)
                vec_shoulders = np.array([
                    left_shoulder.x - right_shoulder.x,
                    left_shoulder.y - right_shoulder.y,
                    left_shoulder.z - right_shoulder.z
                ])


                reference_vec = np.array([0,1,0])
                normal_vec_shoulders = np.cross(vec_shoulders, reference_vec)

                normal_vec_shoulders = np.array([
                    normal_vec_shoulders[0],
                    0,
                    normal_vec_shoulders[2]
                ])
                
                angle_deg = angle_between_vectors(normal_vec_shoulders, vec_face_proj)
                # Display angle on image
                cv2.putText(image, f"Head-Body Rotation: {angle_deg:.2f} deg", 
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                print(f"Head-Body Rotation Angle: {angle_deg:.2f} degrees")

            except Exception as e:
                print(f"Error processing landmarks: {e}")

        else:
            cv2.putText(image, "No pose detected",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Head-Body Rotation Estimation', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
