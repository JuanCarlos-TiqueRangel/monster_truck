import cv2
import numpy as np

video_path = "new_car_obstacle_1.mp4"
out_path = "obstacle_1.png"

cap = cv2.VideoCapture(video_path)
ret, base = cap.read()
if not ret:
    raise RuntimeError("Could not read video")

acc = base.copy()
base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

stride = 14.5
motion_thresh = 25  # 15–35 typical; lower = more sensitive
kernel = np.ones((7, 7), np.uint8)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % stride != 0:
        continue

    # --- 1) Motion mask (captures body + wheels because they move) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, base_gray)
    motion = (diff > motion_thresh).astype(np.uint8) * 255
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=1)
    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel, iterations=2)
    motion = cv2.dilate(motion, kernel, iterations=2)

    # # --- 2) Red-likeness mask in Lab (more robust to lighting than HSV) ---
    # lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # a = lab[:, :, 1]  # higher = more red
    # a_blur = cv2.GaussianBlur(a, (5, 5), 0)
    # _, red_like = cv2.threshold(a_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # Dilate red to make sure it touches nearby robot pixels (helps connect to wheels)
    # red_seed = cv2.dilate(red_like, kernel, iterations=2)


    # --- 2) Blue + black mask for your new car ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blue parts
    lower_blue = np.array([95, 50, 40])
    upper_blue = np.array([135, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Black body / wheels
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 70])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    car_color_mask = cv2.bitwise_or(blue_mask, black_mask)

    # Dilate so it connects to nearby moving robot pixels
    color_seed = cv2.dilate(car_color_mask, kernel, iterations=2)



    # --- 3) Keep ONLY the moving connected component(s) that overlap the red seed ---
    num, labels, stats, _ = cv2.connectedComponentsWithStats(motion, connectivity=8)

    mask = np.zeros_like(motion)
    for k in range(1, num):  # skip background 0
        component = (labels == k).astype(np.uint8) * 255
        if cv2.countNonZero(cv2.bitwise_and(component, color_seed)) > 0:
            mask = cv2.bitwise_or(mask, component)

    # Final cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)

    # Paste robot (including wheels) onto accumulator
    acc[mask > 0] = frame[mask > 0]

cap.release()
cv2.imwrite(out_path, acc)
print("Saved:", out_path)

cv2.imshow("Accumulated car frames", acc)
cv2.waitKey(0)
cv2.destroyAllWindows()