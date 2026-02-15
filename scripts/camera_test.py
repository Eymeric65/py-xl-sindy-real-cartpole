import cv2

# 0 is usually the default camera. If you have a built-in webcam, this might be 1.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW is important for Windows performance

# FORCE HIGH FPS SETTINGS
# 60 FPS at 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#cap.set(cv2.CAP_PROP_FPS, 60)

# OR: 120 FPS at 320x240 (Smoothest for fast tracking)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 70)

# Check what we actually got
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Your Tracking Logic Goes Here
    
    cv2.imshow('PS3 Eye Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()