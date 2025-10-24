import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

video_path = "kabadi.mp4"  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Error: Cannot open video at {video_path}")
    exit()
else:
    print("✅ Video opened successfully")

BONUS_HEIGHT_THRESHOLD = 0.02 
GROUND_Y = None
raider_id = None
previous_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        if GROUND_Y is None:
            GROUND_Y = max(left_ankle.y, right_ankle.y) + 0.01
        left_height = GROUND_Y - left_ankle.y
        right_height = GROUND_Y - right_ankle.y
        avg_height = (left_height + right_height) / 2
        movement_score = abs(left_height - previous_positions.get("left", left_height)) + \
                         abs(right_height - previous_positions.get("right", right_height))
        previous_positions["left"] = left_height
        previous_positions["right"] = right_height
        if raider_id is None or movement_score > previous_positions.get("score", 0):
            raider_id = "raider"
            previous_positions["score"] = movement_score
        decision = "Bonus ✅" if avg_height > BONUS_HEIGHT_THRESHOLD else "No Bonus ❌"
        color = (0, 255, 0) if avg_height > BONUS_HEIGHT_THRESHOLD else (0, 0, 255)
        cv2.putText(frame, "RAIDER DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Left Foot Height: {left_height:.4f}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Right Foot Height: {right_height:.4f}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, decision, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Kabaddi DRS - MediaPipe", frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
