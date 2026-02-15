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

# ============================================================================
# PS3EYE CAMERA CALIBRATION TOOL
# ============================================================================

# Initial settings
current_exposure = 20
current_gain = 16
current_brightness = 0
current_contrast = 0
current_wb_red = 128      # White balance red gain (0-255)
current_wb_blue = 128     # White balance blue gain (0-255)
use_color = True  # Start with color to see white balance effect
auto_wb = False  # Start with manual control

# Resolution and FPS
resolution = pseyepy.Camera.RES_SMALL  # 320x240
fps = 60  # Moderate FPS for calibration

print("=== PS3Eye Camera Calibration Tool ===")
print("\nInitializing camera...")

try:
    cam = pseyepy.Camera(ids=[0], 
                        resolution=resolution, 
                        fps=fps, 
                        colour=use_color)
    
    print("Camera initialized!")
    
    # Set initial manual settings
    cam.auto_gain = False
    cam.auto_exposure = False
    cam.auto_whitebalance = auto_wb
    cam.exposure = current_exposure
    cam.gain = current_gain
    
    # Set manual white balance if not auto
    if not auto_wb:
        cam.whitebalance_red = current_wb_red
        cam.whitebalance_blue = current_wb_blue
    
    print("\nCamera ready for calibration")
    
except Exception as e:
    print(f"Camera initialization failed: {e}")
    import sys
    sys.exit(1)

print("\n" + "="*60)
print("CONTROLS:")
print("="*60)
print("Exposure:       'e' decrease   'E' increase   (range: 0-255)")
print("Gain:           'g' decrease   'G' increase   (range: 0-255)")
print("WB Red:         'r' decrease   'R' increase   (range: 0-255)")
print("WB Blue:        'b' decrease   'B' increase   (range: 0-255)")
print("Auto WB:        'w' toggle auto white balance")
print("Mode:           'm' toggle color/grayscale")
print("Histogram:      'h' toggle histogram display")
print("Save:           's' save current settings to file")
print("Quit:           'q' exit")
print("="*60)

cv2.namedWindow('Camera View')
cv2.namedWindow('Histogram')

show_histogram = True
frame_count = 0
last_time = time.time()

def calculate_histogram(frame):
    """Calculate histogram for grayscale or color image"""
    if len(frame.shape) == 2:
        # Grayscale
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
        return [hist], ['gray']
    else:
        # Color (BGR)
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
        return [hist_b, hist_g, hist_r], ['blue', 'green', 'red']

def draw_histogram(hists, labels, is_color):
    """Draw histogram visualization"""
    hist_height = 200
    hist_width = 512
    hist_img = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] if is_color else [(255, 255, 255)]
    
    for hist, color in zip(hists, colors):
        # Normalize
        cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)
        
        # Draw lines
        for i in range(1, 256):
            x1 = int((i-1) * hist_width / 256)
            x2 = int(i * hist_width / 256)
            y1 = hist_height - int(hist[i-1][0])
            y2 = hist_height - int(hist[i][0])
            cv2.line(hist_img, (x1, y1), (x2, y2), color, 1)
    
    # Draw grid
    for i in range(0, hist_width, hist_width // 8):
        cv2.line(hist_img, (i, 0), (i, hist_height), (50, 50, 50), 1)
    for i in range(0, hist_height, hist_height // 4):
        cv2.line(hist_img, (0, i), (hist_width, i), (50, 50, 50), 1)
    
    # Add labels
    cv2.putText(hist_img, "0", (5, hist_height - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    cv2.putText(hist_img, "255", (hist_width - 30, hist_height - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    
    return hist_img

def print_settings():
    """Print current camera settings"""
    print(f"\nCurrent Settings:")
    print(f"  Mode: {'COLOR' if use_color else 'GRAYSCALE'}")
    print(f"  Exposure: {current_exposure}")
    print(f"  Gain: {current_gain}")
    print(f"  Brightness: {current_brightness}")
    print(f"  Contrast: {current_contrast}")
    print(f"  Auto White Balance: {'ON' if auto_wb else 'OFF'}")
    if not auto_wb:
        print(f"  WB Red: {current_wb_red}")
        print(f"  WB Blue: {current_wb_blue}")

def save_settings():
    """Save settings to file"""
    filename = "camera_settings.txt"
    with open(filename, 'w') as f:
        f.write(f"# PS3Eye Camera Settings\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"mode={'color' if use_color else 'grayscale'}\n")
        f.write(f"exposure={current_exposure}\n")
        f.write(f"gain={current_gain}\n")
        f.write(f"brightness={current_brightness}\n")
        f.write(f"contrast={current_contrast}\n")
        f.write(f"auto_whitebalance={'True' if auto_wb else 'False'}\n")
        f.write(f"whitebalance_red={current_wb_red}\n")
        f.write(f"whitebalance_blue={current_wb_blue}\n")
    print(f"\nSettings saved to {filename}")

try:
    while True:
        # Read frame
        frame, timestamp = cam.read()
        
        # Convert to BGR if color mode, or grayscale to BGR for display
        if use_color:
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Calculate FPS
        current_time = time.time()
        fps_actual = 1.0 / (current_time - last_time) if frame_count > 0 else 0
        last_time = current_time
        frame_count += 1
        
        # Calculate image statistics
        if use_color:
            gray_for_stats = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_for_stats = frame
        
        mean_val = np.mean(gray_for_stats)
        std_val = np.std(gray_for_stats)
        min_val = np.min(gray_for_stats)
        max_val = np.max(gray_for_stats)
        
        # Add info overlay
        y_offset = 20
        line_height = 18
        
        cv2.putText(display_frame, f"FPS: {fps_actual:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += line_height
        
        cv2.putText(display_frame, f"Mode: {'COLOR' if use_color else 'GRAY'}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(display_frame, f"Exposure: {current_exposure}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(display_frame, f"Gain: {current_gain}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        if auto_wb:
            cv2.putText(display_frame, f"White Balance: AUTO", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(display_frame, f"WB Red: {current_wb_red}  Blue: {current_wb_blue}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Image statistics
        y_offset += 5
        cv2.putText(display_frame, f"Mean: {mean_val:.1f}  Std: {std_val:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(display_frame, f"Range: [{min_val:.0f}, {max_val:.0f}]", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Quality indicator
        if mean_val < 50:
            quality = "TOO DARK"
            quality_color = (0, 0, 255)
        elif mean_val > 200:
            quality = "TOO BRIGHT"
            quality_color = (0, 0, 255)
        elif std_val < 20:
            quality = "LOW CONTRAST"
            quality_color = (0, 165, 255)
        else:
            quality = "GOOD"
            quality_color = (0, 255, 0)
        
        cv2.putText(display_frame, quality, 
                   (display_frame.shape[1] - 120, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        
        cv2.imshow('Camera View', display_frame)
        
        # Histogram
        if show_histogram:
            hists, labels = calculate_histogram(frame)
            hist_img = draw_histogram(hists, labels, use_color)
            cv2.imshow('Histogram', hist_img)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        # Exposure controls
        elif key == ord('e'):
            current_exposure = max(0, current_exposure - 5)
            cam.exposure = current_exposure
            print(f"Exposure: {current_exposure}")
        elif key == ord('E'):
            current_exposure = min(255, current_exposure + 5)
            cam.exposure = current_exposure
            print(f"Exposure: {current_exposure}")
        
        # Gain controls
        elif key == ord('g'):
            current_gain = max(0, current_gain - 4)
            cam.gain = current_gain
            print(f"Gain: {current_gain}")
        elif key == ord('G'):
            current_gain = min(255, current_gain + 4)
            cam.gain = current_gain
            print(f"Gain: {current_gain}")
        
        # White balance red control
        elif key == ord('r') and not auto_wb:
            current_wb_red = max(0, current_wb_red - 8)
            cam.whitebalance_red = current_wb_red
            print(f"White Balance Red: {current_wb_red}")
        elif key == ord('R') and not auto_wb:
            current_wb_red = min(255, current_wb_red + 8)
            cam.whitebalance_red = current_wb_red
            print(f"White Balance Red: {current_wb_red}")
        
        # White balance blue control
        elif key == ord('b') and not auto_wb:
            current_wb_blue = max(0, current_wb_blue - 8)
            cam.whitebalance_blue = current_wb_blue
            print(f"White Balance Blue: {current_wb_blue}")
        elif key == ord('B') and not auto_wb:
            current_wb_blue = min(255, current_wb_blue + 8)
            cam.whitebalance_blue = current_wb_blue
            print(f"White Balance Blue: {current_wb_blue}")
        
        # White balance toggle
        elif key == ord('w'):
            auto_wb = not auto_wb
            cam.auto_whitebalance = auto_wb
            if not auto_wb:
                cam.whitebalance_red = current_wb_red
                cam.whitebalance_blue = current_wb_blue
            print(f"Auto White Balance: {'ON' if auto_wb else 'OFF'}")
        
        # Mode toggle
        elif key == ord('m'):
            use_color = not use_color
            print(f"\nSwitching to {'COLOR' if use_color else 'GRAYSCALE'} mode...")
            cam.end()
            time.sleep(0.2)
            
            cam = pseyepy.Camera(ids=[0], 
                                resolution=resolution, 
                                fps=fps, 
                                colour=use_color)
            
            cam.auto_gain = False
            cam.auto_exposure = False
            cam.auto_whitebalance = auto_wb
            cam.exposure = current_exposure
            cam.gain = current_gain
            
            if not auto_wb:
                cam.whitebalance_red = current_wb_red
                cam.whitebalance_blue = current_wb_blue
            
            print(f"Switched to {'COLOR' if use_color else 'GRAYSCALE'}")
        
        # Histogram toggle
        elif key == ord('h'):
            show_histogram = not show_histogram
            if not show_histogram:
                cv2.destroyWindow('Histogram')
            print(f"Histogram: {'ON' if show_histogram else 'OFF'}")
        
        # Print settings
        elif key == ord('p'):
            print_settings()
        
        # Save settings
        elif key == ord('s'):
            save_settings()

finally:
    cam.end()
    cv2.destroyAllWindows()
    print("\nFinal settings:")
    print_settings()
    print("\nCamera closed.")
