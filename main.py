import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from collections import deque


VIDEO_PATH = "kabaddi.mp4"
CROP_W = 300
CROP_H = 480
SLOW_MS = 80
DIST_THRESHOLD_PX = 20
SIGNIFICANT_HEIGHT_DIFF = 20   
CSV_OUT = "detections.csv"
WINDOW_NAME = "Kabaddi DRS - Simple Bonus Detect"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

selected_click = None
pause_frame = False
frame_for_pause = None
GROUND_LINE = None
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) else 30
frame_idx = 0


with open(CSV_OUT, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "time_s", "left_vertical_dist", "right_vertical_dist", "height_diff", "bonus_flag", "bonus_reason"])

def detect_black_line(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.Canny(th, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=30)
    if lines is None:
        return None
    best = None
    best_len = 0
    for l in lines:
        x1,y1,x2,y2 = l[0]
        length = np.hypot(x2-x1, y2-y1)
        angle = abs((y2-y1)/(x2-x1+1e-6))
        score = length / (1 + angle*10)
        if score > best_len:
            best_len = score
            best = (x1,y1,x2,y2)
    return best

def crop_region(full_frame, cx, cy, w=CROP_W, h=CROP_H):
    H, W = full_frame.shape[:2]
    x1 = int(max(0, cx - w//2))
    y1 = int(max(0, cy - h//2))
    x2 = int(min(W, x1 + w))
    y2 = int(min(H, y1 + h))
    if (x2 - x1) < w:
        x1 = max(0, x2 - w)
    if (y2 - y1) < h:
        y1 = max(0, y2 - h)
    return x1, y1, x2, y2, full_frame[y1:y2, x1:x2].copy()

def draw_line_overlay(frame, line, color=(0,255,255)):
    if line is None: 
        return
    x1,y1,x2,y2 = line
    cv2.line(frame, (x1,y1), (x2,y2), color, 3)

def get_vertical_distances(left_y, right_y, line_y):
    left_vertical = line_y - left_y
    right_vertical = line_y - right_y
    return left_vertical, right_vertical

def detect_bonus_simple(left_y, right_y, line_y):
    """
    Bonus if the vertical difference between ankles is significant.
    """
    left_vertical, right_vertical = get_vertical_distances(left_y, right_y, line_y)
    height_diff = abs(left_vertical - right_vertical)
    if height_diff >= SIGNIFICANT_HEIGHT_DIFF:
        bonus_flag = True
        reason = f"Foot height diff {height_diff:.1f}px >= threshold {SIGNIFICANT_HEIGHT_DIFF}. BONUS."
    else:
        bonus_flag = False
        reason = f"Foot height diff {height_diff:.1f}px < threshold. NO BONUS."
    return bonus_flag, reason, left_vertical, right_vertical, height_diff

def on_mouse(event, x, y, flags, param):
    global selected_click, pause_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_click = (x, y)
        pause_frame = True
        print(f"[CLICK] selected at {selected_click} - pausing for analysis")

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

ret0, frame0 = cap.read()
if not ret0:
    raise RuntimeError("Cannot open video or video is empty")
initial_line = detect_black_line(frame0)
if initial_line is not None:
    GROUND_LINE = initial_line
    print("[INFO] Auto-detected black line:", GROUND_LINE)
else:
    H0, W0 = frame0.shape[:2]
    y = int(H0 * 0.85)
    GROUND_LINE = (0, y, W0, y)
    print("[WARN] Could not auto-detect line. Using fallback at y=", y)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0


while True:
    if not pause_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_for_pause = frame.copy()
    else:
        frame = frame_for_pause.copy()

    H, W = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_idx % 30 == 0 and not pause_frame:
        new_line = detect_black_line(frame)
        if new_line is not None:
            GROUND_LINE = new_line
    cv2.putText(frame, "Click on raider to analyze (pauses). Press 'c' to continue, 'q' to quit", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,0), 2, cv2.LINE_AA)
    draw_line_overlay(frame, GROUND_LINE, (0,255,255))

    if selected_click and pause_frame:
        cx, cy = selected_click
        x1, y1, x2, y2, crop = crop_region(frame, cx, cy, CROP_W, CROP_H)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = pose.process(crop_rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            la = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            ra = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_x_full = int(la.x * (x2-x1) + x1)
            left_y_full = int(la.y * (y2-y1) + y1)
            right_x_full = int(ra.x * (x2-x1) + x1)
            right_y_full = int(ra.y * (y2-y1) + y1)

            
            x1l,y1l,x2l,y2l = GROUND_LINE
            line_avg_y = (y1l + y2l) / 2.0

           
            bonus_flag, bonus_reason, left_vertical_dist, right_vertical_dist, height_diff = detect_bonus_simple(
                left_y_full, right_y_full, line_avg_y
            )
            
           
            left_color = (0,255,255) if bonus_flag else (0,255,0)
            right_color = (0,255,255) if bonus_flag else (0,255,0)
            cv2.circle(frame, (left_x_full, left_y_full), 8, left_color, -1)
            cv2.circle(frame, (right_x_full, right_y_full), 8, right_color, -1)
            cv2.putText(frame, "L", (left_x_full-20, left_y_full-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)
            cv2.putText(frame, "R", (right_x_full+10, right_y_full-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)
            
            y_offset = 70
            cv2.putText(frame, f"L_vertical: {left_vertical_dist:.1f}px", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,100,100), 2)
            y_offset += 30
            cv2.putText(frame, f"R_vertical: {right_vertical_dist:.1f}px", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,100,100), 2)
            y_offset += 30
            cv2.putText(frame, f"Height diff: {height_diff:.1f}px", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,200,100), 2)
            y_offset += 30
            bonus_color = (0, 255, 0) if bonus_flag else (0, 0, 255)
            cv2.putText(frame, f"BONUS: {'YES' if bonus_flag else 'NO'}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, bonus_color, 3)
            y_offset += 40
            reason_lines = [bonus_reason[i:i+50] for i in range(0, len(bonus_reason), 50)]
            for reason_line in reason_lines:
                cv2.putText(frame, reason_line, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 2)
                y_offset += 25
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            crop_vis = crop.copy()
            mp.solutions.drawing_utils.draw_landmarks(crop_vis, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            preview = cv2.resize(crop_vis, (int(CROP_W*0.5), int(CROP_H*0.5)))
            ph, pw = preview.shape[:2]
            if (10+ph) <= H and (W-10-pw) >= 0:
                frame[10:10+ph, W-10-pw:W-10] = preview
            # WRITE to CSV
            with open(CSV_OUT, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([frame_idx, round(frame_idx / fps, 3), 
                                 round(left_vertical_dist, 3), round(right_vertical_dist, 3), 
                                 round(height_diff, 3), int(bonus_flag), bonus_reason])
        else:
            cv2.putText(frame, "No pose detected in selected crop. Try different point.", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)
    
    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(SLOW_MS) & 0xFF
    if key == ord('q'):
        print("[QUIT] Exiting.")
        break
    elif key == ord('c'):
        selected_click = None
        pause_frame = False
        print("[CONTINUE] Resuming video.")
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Detection results saved to {CSV_OUT}")

