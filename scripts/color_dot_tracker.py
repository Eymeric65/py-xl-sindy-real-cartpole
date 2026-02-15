import cv2
import numpy as np

# 0 is usually the default camera. If you have a built-in webcam, this might be 1.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW is important for Windows performance

# FORCE HIGH FPS SETTINGS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 70)

# Check what we actually got
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

# Define color ranges in HSV
# Green color range
green_lower = np.array([35, 50, 50])
green_upper = np.array([85, 255, 255])

# Yellow color range
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([35, 255, 255])

# Minimum area to filter out noise
min_area = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create masks for green and yellow
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours for green dots
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in green_contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Get center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw circle and label
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
                cv2.putText(frame, f"Green ({cx}, {cy})", (cx - 50, cy - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Find contours for yellow dots
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in yellow_contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Get center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw circle and label
                cv2.circle(frame, (cx, cy), 10, (0, 255, 255), 2)
                cv2.putText(frame, f"Yellow ({cx}, {cy})", (cx - 50, cy - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Display the result
    cv2.imshow('Green & Yellow Dot Tracking', frame)
    
    # Optional: Show masks for debugging
    cv2.imshow('Green Mask', green_mask)
    cv2.imshow('Yellow Mask', yellow_mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
