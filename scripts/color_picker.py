import cv2
import numpy as np
import time
try:
    import pseyepy
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pseyepy'))
    import pseyepy

# Global variables to store HSV values
h_min = 0
s_min = 0
v_min = 0
h_max = 179
s_max = 255
v_max = 255

# Camera control settings
current_exposure = 20
current_gain = 16
current_wb_red = 128
current_wb_blue = 128
auto_wb = False

def nothing(x):
    """Callback function for trackbars (does nothing)"""
    pass

# Initialize PS3Eye camera
print("Initializing PS3Eye camera...")
resolution = pseyepy.Camera.RES_SMALL  # 320x240
fps = 60

try:
    cam = pseyepy.Camera(ids=[0], 
                        resolution=resolution, 
                        fps=fps, 
                        colour=True)
    
    # Set manual settings
    cam.auto_gain = False
    cam.auto_exposure = False
    cam.auto_whitebalance = auto_wb
    cam.exposure = current_exposure
    cam.gain = current_gain
    
    if not auto_wb:
        cam.whitebalance_red = current_wb_red
        cam.whitebalance_blue = current_wb_blue
    
    print("Camera initialized!")
    
except Exception as e:
    print(f"Camera initialization failed: {e}")
    import sys
    sys.exit(1)

print(f"Resolution: 320x240, Target FPS: {fps}")

# Create windows
cv2.namedWindow('Original')
cv2.namedWindow('Mask')
cv2.namedWindow('Result')
cv2.namedWindow('HSV Controls')

# Create trackbars for HSV adjustment
cv2.createTrackbar('H Min', 'HSV Controls', 0, 179, nothing)
cv2.createTrackbar('H Max', 'HSV Controls', 179, 179, nothing)
cv2.createTrackbar('S Min', 'HSV Controls', 0, 255, nothing)
cv2.createTrackbar('S Max', 'HSV Controls', 255, 255, nothing)
cv2.createTrackbar('V Min', 'HSV Controls', 0, 255, nothing)
cv2.createTrackbar('V Max', 'HSV Controls', 255, 255, nothing)

# Create trackbar for morphology kernel size
cv2.createTrackbar('Kernel', 'HSV Controls', 5, 20, nothing)

print("\n=== COLOR PICKER TOOL (PS3Eye) ===")
print("Instructions:")
print("1. Adjust the trackbars to isolate your target color")
print("2. Watch the 'Mask' window - white areas are detected")
print("\nCamera Controls:")
print("  'e'/'E' - decrease/increase Exposure")
print("  'g'/'G' - decrease/increase Gain")
print("  'r'/'R' - decrease/increase WB Red")
print("  'b'/'B' - decrease/increase WB Blue")
print("  'w'     - toggle auto white balance")
print("\nPresets:")
print("  'p' - print current HSV values")
print("  's' - save current values")
print("  '1' - load Green preset")
print("  '2' - load Yellow preset")
print("  '0' - reset to full range")
print("  'q' - quit\n")

# Preset values
green_preset = {'h_min': 35, 'h_max': 85, 's_min': 50, 's_max': 255, 'v_min': 50, 'v_max': 255}
yellow_preset = {'h_min': 20, 'h_max': 35, 's_min': 100, 's_max': 255, 'v_min': 100, 'v_max': 255}

def load_preset(preset):
    """Load a preset into trackbars"""
    cv2.setTrackbarPos('H Min', 'HSV Controls', preset['h_min'])
    cv2.setTrackbarPos('H Max', 'HSV Controls', preset['h_max'])
    cv2.setTrackbarPos('S Min', 'HSV Controls', preset['s_min'])
    cv2.setTrackbarPos('S Max', 'HSV Controls', preset['s_max'])
    cv2.setTrackbarPos('V Min', 'HSV Controls', preset['v_min'])
    cv2.setTrackbarPos('V Max', 'HSV Controls', preset['v_max'])

frame_count = 0
last_time = time.time()

try:
    while True:
        # Read frame from PS3Eye
        frame, timestamp = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Calculate FPS
        current_time = time.time()
        fps_actual = 1.0 / (current_time - last_time) if frame_count > 0 else 0
        last_time = current_time
        frame_count += 1
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get current trackbar positions
        h_min = cv2.getTrackbarPos('H Min', 'HSV Controls')
        h_max = cv2.getTrackbarPos('H Max', 'HSV Controls')
        s_min = cv2.getTrackbarPos('S Min', 'HSV Controls')
        s_max = cv2.getTrackbarPos('S Max', 'HSV Controls')
        v_min = cv2.getTrackbarPos('V Min', 'HSV Controls')
        v_max = cv2.getTrackbarPos('V Max', 'HSV Controls')
        kernel_size = cv2.getTrackbarPos('Kernel', 'HSV Controls')
        
        # Ensure kernel size is odd and at least 1
        kernel_size = max(1, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create HSV range
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        
        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations if kernel > 1
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask to original frame
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Find and draw contours on the result
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_with_contours = result.copy()
        cv2.drawContours(result_with_contours, contours, -1, (0, 255, 0), 2)
        
        # Add text overlay with current values
        y_offset = 20
        line_height = 20
        
        text = f"H:[{h_min}-{h_max}] S:[{s_min}-{s_max}] V:[{v_min}-{v_max}]"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_offset += line_height
        
        cv2.putText(frame, f"Contours: {len(contours)}  FPS: {fps_actual:.1f}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_offset += line_height
        
        # Camera settings overlay
        cv2.putText(frame, f"Exp:{current_exposure} Gain:{current_gain}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_offset += line_height
        
        if auto_wb:
            cv2.putText(frame, "WB: AUTO", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(frame, f"WB R:{current_wb_red} B:{current_wb_blue}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Display windows
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result_with_contours)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        # Camera controls
        elif key == ord('e'):
            current_exposure = max(0, current_exposure - 5)
            cam.exposure = current_exposure
            print(f"Exposure: {current_exposure}")
        elif key == ord('E'):
            current_exposure = min(255, current_exposure + 5)
            cam.exposure = current_exposure
            print(f"Exposure: {current_exposure}")
        
        elif key == ord('g'):
            current_gain = max(0, current_gain - 4)
            cam.gain = current_gain
            print(f"Gain: {current_gain}")
        elif key == ord('G'):
            current_gain = min(255, current_gain + 4)
            cam.gain = current_gain
            print(f"Gain: {current_gain}")
        
        elif key == ord('r') and not auto_wb:
            current_wb_red = max(0, current_wb_red - 8)
            cam.whitebalance_red = current_wb_red
            print(f"White Balance Red: {current_wb_red}")
        elif key == ord('R') and not auto_wb:
            current_wb_red = min(255, current_wb_red + 8)
            cam.whitebalance_red = current_wb_red
            print(f"White Balance Red: {current_wb_red}")
        
        elif key == ord('b') and not auto_wb:
            current_wb_blue = max(0, current_wb_blue - 8)
            cam.whitebalance_blue = current_wb_blue
            print(f"White Balance Blue: {current_wb_blue}")
        elif key == ord('B') and not auto_wb:
            current_wb_blue = min(255, current_wb_blue + 8)
            cam.whitebalance_blue = current_wb_blue
            print(f"White Balance Blue: {current_wb_blue}")
        
        elif key == ord('w'):
            auto_wb = not auto_wb
            cam.auto_whitebalance = auto_wb
            if not auto_wb:
                cam.whitebalance_red = current_wb_red
                cam.whitebalance_blue = current_wb_blue
            print(f"Auto White Balance: {'ON' if auto_wb else 'OFF'}")
        
        # HSV presets and functions
        elif key == ord('p'):
            # Print current values
            print("\n=== CURRENT HSV VALUES ===")
            print(f"Lower: np.array([{h_min}, {s_min}, {v_min}])")
            print(f"Upper: np.array([{h_max}, {s_max}, {v_max}])")
            print(f"Kernel size: {kernel_size}")
            print(f"Contours found: {len(contours)}")
            print(f"Camera: Exp={current_exposure}, Gain={current_gain}, WB_R={current_wb_red}, WB_B={current_wb_blue}")
        
        elif key == ord('1'):
            # Load green preset
            load_preset(green_preset)
            print("Loaded GREEN preset")
        elif key == ord('2'):
            # Load yellow preset
            load_preset(yellow_preset)
            print("Loaded YELLOW preset")
        elif key == ord('0'):
            # Reset to full range
            cv2.setTrackbarPos('H Min', 'HSV Controls', 0)
            cv2.setTrackbarPos('H Max', 'HSV Controls', 179)
            cv2.setTrackbarPos('S Min', 'HSV Controls', 0)
            cv2.setTrackbarPos('S Max', 'HSV Controls', 255)
            cv2.setTrackbarPos('V Min', 'HSV Controls', 0)
            cv2.setTrackbarPos('V Max', 'HSV Controls', 255)
            print("Reset to full range")
        elif key == ord('s'):
            # Save current values
            print("\n=== SAVE CURRENT VALUES ===")
            print(f"lower = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper = np.array([{h_max}, {s_max}, {v_max}])")
            print(f"exposure = {current_exposure}")
            print(f"gain = {current_gain}")
            print(f"whitebalance_red = {current_wb_red}")
            print(f"whitebalance_blue = {current_wb_blue}")
            print("Values saved to console!")

finally:
    cam.end()
    cv2.destroyAllWindows()
    print("\nCamera closed.")
