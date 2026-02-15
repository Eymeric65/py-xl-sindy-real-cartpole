import cv2
import numpy as np
try:
    import pseyepy
except ImportError:
    import sys
    import os
    # Fallback: add pseyepy to path if not installed
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pseyepy'))
    import pseyepy

# Global variables
points = []
templates = []
template_size = 15
search_area = 80
colors = [(0, 255, 0), (0, 255, 255)]  # Green and Yellow in BGR
labels = ["Point 1", "Point 2"]
tracking = False
last_positions = []
debug_mode = True
match_threshold = 0.4
use_grayscale = True

# Camera parameters
current_exposure = 20
current_gain = 16

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to capture initial clicks"""
    global points, templates, tracking, last_positions
    
    if event == cv2.EVENT_LBUTTONDOWN and not tracking:
        if len(points) < 2:
            points.append((x, y))
            print(f"Point {len(points)} selected at: ({x}, {y})")
            
            if len(points) == 2:
                # Extract templates around clicked points
                frame = param
                for i, (px, py) in enumerate(points):
                    # Extract template region
                    half_size = template_size // 2
                    y1 = max(0, py - half_size)
                    y2 = min(frame.shape[0], py + half_size)
                    x1 = max(0, px - half_size)
                    x2 = min(frame.shape[1], px + half_size)
                    
                    template = frame[y1:y2, x1:x2].copy()
                    templates.append(template)
                    last_positions.append((px, py))
                    
                    print(f"Template {i+1} extracted: {template.shape}")
                
                tracking = True
                print("\nTracking started! Press 'r' to reset, 'q' to quit")
                
                # Show captured templates in debug windows
                for i, template in enumerate(templates):
                    cv2.imshow(f'Template {i+1} - {labels[i]}', template)

def find_template_in_region(frame, template, center, search_size):
    """
    Search for template in a region around the center point
    Returns the best match position or None if not found
    """
    cx, cy = center
    half_search = search_size // 2
    
    # Define search region
    y1 = max(0, cy - half_search)
    y2 = min(frame.shape[0], cy + half_search)
    x1 = max(0, cx - half_search)
    x2 = min(frame.shape[1], cx + half_search)
    
    # Extract search region
    search_region = frame[y1:y2, x1:x2]
    
    if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
        return None
    
    # Convert to grayscale if needed (more robust to lighting changes)
    if use_grayscale:
        search_region_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY) if len(search_region.shape) == 3 else search_region
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        result = cv2.matchTemplate(search_region_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    else:
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Check if match is good enough
    if max_val > match_threshold:
        # Convert local coordinates to global
        match_x = x1 + max_loc[0] + template.shape[1] // 2
        match_y = y1 + max_loc[1] + template.shape[0] // 2
        return (match_x, match_y, max_val)
    
    return None

# Initialize PS3Eye camera
print("Initializing PS3Eye camera...")
try:
    # Maximum safe FPS: 187 at 320x240 resolution
    cam = pseyepy.Camera(ids=[0], resolution=pseyepy.Camera.RES_LARGE, fps=60, colour=True)
    #cam = pseyepy.Camera(ids=[0], resolution=pseyepy.Camera.RES_SMALL, fps=187, colour=True)
    print(f"Camera initialized successfully!")
    print(f"Running at MAXIMUM FPS: 187fps @ 320x240")
    
    # Disable auto settings for manual control (except white balance)
    cam.auto_gain = False
    cam.auto_exposure = False
    cam.auto_whitebalance = True  # Enable automatic white balance
    
    # Set initial exposure and gain
    cam.exposure = current_exposure
    cam.gain = current_gain
    
    print(f"Initial settings: Exposure={current_exposure}, Gain={current_gain}")
except Exception as e:
    print(f"Failed to initialize camera: {e}")
    print("Make sure pseyepy is properly installed and PS3Eye camera is connected")
    sys.exit(1)

print("\n=== PS3EYE CLICK-TO-TRACK TRACKER ===")
print("Instructions:")
print("1. Click on the first marker/dot")
print("2. Click on the second marker/dot")
print("3. Tracking will start automatically")
print("\nCamera Controls:")
print("  'e'/'E' - Decrease/Increase exposure (0-255)")
print("  'g'/'G' - Decrease/Increase gain (0-63)")
print("\nTracking Controls:")
print("  'd' - Toggle debug mode")
print("  '+/-' - Adjust match threshold")
print("  'x' - Toggle grayscale matching")
print("  'r' - Reset and select new points")
print("  'q' - Quit\n")
print(f"Current: Exposure={current_exposure}, Gain={current_gain}, Threshold={match_threshold}, Grayscale={use_grayscale}")

# Create window and set mouse callback
cv2.namedWindow('PS3Eye Tracker')

# Read first frame for initialization
frame, ts = cam.read()
# Convert RGB to BGR for OpenCV compatibility
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
first_frame = frame.copy()

cv2.setMouseCallback('PS3Eye Tracker', mouse_callback, first_frame)

# Show first frame with instructions
frame_display = first_frame.copy()
cv2.putText(frame_display, "Click on the two markers to track", (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imshow('PS3Eye Tracker', frame_display)

try:
    while True:
        # Read frame from PS3Eye (returns RGB format)
        frame, ts = cam.read()
        
        # Convert RGB to BGR for OpenCV compatibility
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame_display = frame.copy()
        
        if not tracking:
            # Waiting for user to click points
            cv2.setMouseCallback('PS3Eye Tracker', mouse_callback, frame)
            
            # Show current selections
            for i, (px, py) in enumerate(points):
                cv2.circle(frame_display, (px, py), 10, colors[i], 2)
                cv2.putText(frame_display, labels[i], (px - 30, py - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
            
            # Show instructions
            if len(points) == 0:
                cv2.putText(frame_display, "Click on FIRST marker", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif len(points) == 1:
                cv2.putText(frame_display, "Click on SECOND marker", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # Tracking mode
            for i, (template, last_pos) in enumerate(zip(templates, last_positions)):
                # Search for template around last known position
                result = find_template_in_region(frame, template, last_pos, search_area)
                
                if result is not None:
                    match_x, match_y, confidence = result
                    last_positions[i] = (match_x, match_y)
                    
                    # Draw circle and label
                    cv2.circle(frame_display, (match_x, match_y), 10, colors[i], 2)
                    cv2.putText(frame_display, f"{labels[i]} ({match_x}, {match_y})",
                               (match_x - 60, match_y - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
                    
                    # Draw search area and template size in debug mode
                    if debug_mode:
                        half_search = search_area // 2
                        cv2.rectangle(frame_display, 
                                     (match_x - half_search, match_y - half_search),
                                     (match_x + half_search, match_y + half_search),
                                     colors[i], 1)
                        
                        half_template = template_size // 2
                        cv2.rectangle(frame_display,
                                     (match_x - half_template, match_y - half_template),
                                     (match_x + half_template, match_y + half_template),
                                     colors[i], 2)
                    
                    # Show confidence
                    cv2.putText(frame_display, f"Conf: {confidence:.2f}",
                               (match_x - 60, match_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
                else:
                    # Lost tracking
                    cv2.putText(frame_display, f"{labels[i]}: LOST",
                               (10, 60 + i * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw line between the two points if both are tracked
            if len(last_positions) == 2:
                cv2.line(frame_display, last_positions[0], last_positions[1], (255, 0, 255), 2)
                
                # Calculate distance
                dx = last_positions[1][0] - last_positions[0][0]
                dy = last_positions[1][1] - last_positions[0][1]
                distance = np.sqrt(dx**2 + dy**2)
                
                mid_x = (last_positions[0][0] + last_positions[1][0]) // 2
                mid_y = (last_positions[0][1] + last_positions[1][1]) // 2
                cv2.putText(frame_display, f"Dist: {distance:.1f}px",
                           (mid_x - 50, mid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Show camera settings and debug info
        if debug_mode:
            cv2.putText(frame_display, "DEBUG MODE ON", (10, frame_display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            mode = "Gray" if use_grayscale else "Color"
            cv2.putText(frame_display, f"Exp:{current_exposure} Gain:{current_gain} Thresh:{match_threshold:.2f} {mode}",
                       (10, frame_display.shape[0] - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        else:
            cv2.putText(frame_display, f"Exp:{current_exposure} Gain:{current_gain}",
                       (10, frame_display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('PS3Eye Tracker', frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset tracking
            points = []
            templates = []
            last_positions = []
            tracking = False
            # Close template windows
            try:
                cv2.destroyWindow('Template 1 - Point 1')
                cv2.destroyWindow('Template 2 - Point 2')
            except:
                pass
            print("\nReset! Click on new points to track.")
        elif key == ord('d'):
            # Toggle debug mode
            debug_mode = not debug_mode
            status = "ON" if debug_mode else "OFF"
            print(f"Debug mode: {status}")
        elif key == ord('+') or key == ord('='):
            # Increase threshold
            match_threshold = min(0.95, match_threshold + 0.05)
            print(f"Match threshold: {match_threshold:.2f} (higher = stricter)")
        elif key == ord('-') or key == ord('_'):
            # Decrease threshold
            match_threshold = max(0.1, match_threshold - 0.05)
            print(f"Match threshold: {match_threshold:.2f} (lower = more permissive)")
        elif key == ord('x'):
            # Toggle grayscale
            use_grayscale = not use_grayscale
            mode = "grayscale" if use_grayscale else "color"
            print(f"Switched to {mode} matching")
        elif key == ord('e'):
            # Decrease exposure
            current_exposure = max(0, current_exposure - 5)
            cam.exposure = current_exposure
            print(f"Exposure: {current_exposure}")
        elif key == ord('E'):
            # Increase exposure
            current_exposure = min(255, current_exposure + 5)
            cam.exposure = current_exposure
            print(f"Exposure: {current_exposure}")
        elif key == ord('g'):
            # Decrease gain
            current_gain = max(0, current_gain - 2)
            cam.gain = current_gain
            print(f"Gain: {current_gain}")
        elif key == ord('G'):
            # Increase gain
            current_gain = min(63, current_gain + 2)
            cam.gain = current_gain
            print(f"Gain: {current_gain}")

finally:
    # Clean up
    cam.end()
    cv2.destroyAllWindows()
    print("\nCamera closed successfully")
