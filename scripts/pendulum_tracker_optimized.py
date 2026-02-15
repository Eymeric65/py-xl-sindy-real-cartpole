import cv2
import numpy as np
import time
from collections import deque
try:
    import pseyepy
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pseyepy'))
    import pseyepy

# ============================================================================
# OPTIMIZED PENDULUM TRACKER - HIGH PERFORMANCE
# ============================================================================

# Configuration
TEMPLATE_SIZE = 15          # Small template for speed
SEARCH_AREA = 60            # Tight search area based on expected motion
MATCH_THRESHOLD = 0.35      # Lower threshold for reliability
FPS_TARGET = 187            # Maximum PS3Eye FPS

# Camera settings (optimized for fast motion)
EXPOSURE = 15               # Very low exposure to minimize motion blur
GAIN = 20                   # Moderate gain

# Tracking state
pivot_point = None          # Cart position (optional)
bob_template = None
last_bob_pos = None
tracking = False
debug_mode = True

# Performance monitoring
frame_times = deque(maxlen=100)
tracking_times = deque(maxlen=100)

# Data recording
positions_history = deque(maxlen=1000)
timestamps_history = deque(maxlen=1000)

def mouse_callback(event, x, y, flags, param):
    """Click to select bob position"""
    global bob_template, last_bob_pos, tracking, pivot_point
    
    if event == cv2.EVENT_LBUTTONDOWN and not tracking:
        frame = param
        
        if pivot_point is None:
            # First click: set pivot (cart position)
            pivot_point = (x, y)
            print(f"Pivot set at: ({x}, {y})")
            print("Now click on the pendulum bob (red marker)")
        else:
            # Second click: set bob and start tracking
            last_bob_pos = (x, y)
            
            # Extract template
            half_size = TEMPLATE_SIZE // 2
            y1 = max(0, y - half_size)
            y2 = min(frame.shape[0], y + half_size)
            x1 = max(0, x - half_size)
            x2 = min(frame.shape[1], x + half_size)
            
            bob_template = frame[y1:y2, x1:x2].copy()
            tracking = True
            
            print(f"Bob selected at: ({x}, {y})")
            print(f"Template size: {bob_template.shape}")
            print("\nTracking started!")
            print("Press 's' to save data, 'r' to reset, 'q' to quit")
            
            cv2.imshow('Bob Template', bob_template)

def find_bob(frame, template, last_pos, search_area):
    """Fast template matching with sub-pixel accuracy"""
    cx, cy = last_pos
    half_search = search_area // 2
    
    # Define search region
    y1 = max(0, cy - half_search)
    y2 = min(frame.shape[0], cy + half_search)
    x1 = max(0, cx - half_search)
    x2 = min(frame.shape[1], cx + half_search)
    
    search_region = frame[y1:y2, x1:x2]
    
    if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
        return None
    
    # Template matching
    result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val < MATCH_THRESHOLD:
        return None
    
    # Convert to global coordinates
    match_x = x1 + max_loc[0] + template.shape[1] // 2
    match_y = y1 + max_loc[1] + template.shape[0] // 2
    
    return (match_x, match_y, max_val)

def calculate_pendulum_angle(pivot, bob):
    """Calculate angle from vertical (degrees)"""
    dx = bob[0] - pivot[0]
    dy = bob[1] - pivot[1]
    
    # Angle from vertical (0° = straight down)
    angle = np.degrees(np.arctan2(dx, dy))
    
    # Length
    length = np.sqrt(dx**2 + dy**2)
    
    return angle, length

def estimate_velocity(positions, timestamps):
    """Estimate angular velocity using recent positions"""
    if len(positions) < 2:
        return 0.0, 0.0
    
    # Use last 5 frames for velocity estimation
    n = min(5, len(positions))
    
    pos_array = np.array(list(positions)[-n:])
    time_array = np.array(list(timestamps)[-n:])
    
    if len(time_array) < 2:
        return 0.0, 0.0
    
    # Linear velocity (pixels/second)
    dt = time_array[-1] - time_array[0]
    if dt < 1e-6:
        return 0.0, 0.0
    
    dx = pos_array[-1, 0] - pos_array[0, 0]
    dy = pos_array[-1, 1] - pos_array[0, 1]
    
    vx = dx / dt
    vy = dy / dt
    
    return vx, vy

# Initialize camera
print("Initializing PS3Eye camera...")
print(f"Target: {FPS_TARGET} FPS @ 320x240, Grayscale mode")

try:
    # Grayscale mode for maximum performance
    cam = pseyepy.Camera(ids=[0], 
                        resolution=pseyepy.Camera.RES_SMALL, 
                        fps=FPS_TARGET, 
                        colour=False)  # Grayscale = fastest
    
    print("Camera initialized!")
    
    # Manual settings for fast motion
    cam.auto_gain = False
    cam.auto_exposure = False
    cam.auto_whitebalance = False
    cam.exposure = EXPOSURE
    cam.gain = GAIN
    
    print(f"Settings: Exposure={EXPOSURE}, Gain={GAIN}")
    
except Exception as e:
    print(f"Camera initialization failed: {e}")
    import sys
    sys.exit(1)

print("\n=== OPTIMIZED PENDULUM TRACKER ===")
print("Setup:")
print("1. Click on PIVOT POINT (cart center)")
print("2. Click on PENDULUM BOB (red marker)")
print("\nControls:")
print("  'd' - Toggle debug visualization")
print("  's' - Save tracking data to file")
print("  'r' - Reset tracking")
print("  'q' - Quit")
print("\n")

cv2.namedWindow('Pendulum Tracker')

# Read first frame
frame, ts = cam.read()
first_frame = frame.copy()
cv2.setMouseCallback('Pendulum Tracker', mouse_callback, first_frame)

# Show first frame
cv2.putText(first_frame, "Click on PIVOT POINT (cart)", (10, 20),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
cv2.imshow('Pendulum Tracker', first_frame)

last_time = time.time()
frame_count = 0

try:
    while True:
        loop_start = time.time()
        
        # Read frame (already grayscale)
        frame, ts = cam.read()
        frame_display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored annotations
        
        # Frame timing
        current_time = time.time()
        frame_dt = current_time - last_time
        frame_times.append(frame_dt)
        last_time = current_time
        
        if not tracking:
            # Setup mode
            cv2.setMouseCallback('Pendulum Tracker', mouse_callback, frame)
            
            if pivot_point:
                cv2.circle(frame_display, pivot_point, 5, (0, 255, 0), -1)
                cv2.putText(frame_display, "Now click on PENDULUM BOB", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(frame_display, "Click on PIVOT POINT", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            # Tracking mode
            track_start = time.time()
            
            result = find_bob(frame, bob_template, last_bob_pos, SEARCH_AREA)
            
            if result is not None:
                bob_x, bob_y, confidence = result
                last_bob_pos = (bob_x, bob_y)
                
                # Record data
                positions_history.append((bob_x, bob_y))
                timestamps_history.append(ts)
                
                # Calculate pendulum properties
                angle, length = calculate_pendulum_angle(pivot_point, last_bob_pos)
                vx, vy = estimate_velocity(positions_history, timestamps_history)
                
                # Draw pivot
                cv2.circle(frame_display, pivot_point, 6, (0, 255, 0), -1)
                cv2.circle(frame_display, pivot_point, 8, (0, 255, 0), 2)
                
                # Draw bob
                cv2.circle(frame_display, last_bob_pos, 8, (0, 0, 255), -1)
                cv2.circle(frame_display, last_bob_pos, 10, (0, 0, 255), 2)
                
                # Draw pendulum rod
                cv2.line(frame_display, pivot_point, last_bob_pos, (255, 0, 255), 2)
                
                # Debug visualization
                if debug_mode:
                    # Search area
                    half = SEARCH_AREA // 2
                    cv2.rectangle(frame_display,
                                (bob_x - half, bob_y - half),
                                (bob_x + half, bob_y + half),
                                (255, 255, 0), 1)
                    
                    # Template area
                    half_t = TEMPLATE_SIZE // 2
                    cv2.rectangle(frame_display,
                                (bob_x - half_t, bob_y - half_t),
                                (bob_x + half_t, bob_y + half_t),
                                (0, 255, 255), 1)
                
                # Display info
                cv2.putText(frame_display, f"Angle: {angle:6.2f} deg", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame_display, f"Length: {length:6.1f} px", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame_display, f"Vel: ({vx:6.1f}, {vy:6.1f}) px/s", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame_display, f"Conf: {confidence:.3f}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Lost tracking
                cv2.putText(frame_display, "LOST TRACKING", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Draw last known position
                cv2.circle(frame_display, last_bob_pos, 10, (0, 0, 255), 2)
            
            track_time = time.time() - track_start
            tracking_times.append(track_time)
        
        # Performance stats
        if len(frame_times) > 0:
            avg_fps = 1.0 / np.mean(frame_times)
            avg_track_ms = np.mean(tracking_times) * 1000 if len(tracking_times) > 0 else 0
            
            cv2.putText(frame_display, f"FPS: {avg_fps:.1f}", (frame_display.shape[1] - 120, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame_display, f"Track: {avg_track_ms:.2f}ms", (frame_display.shape[1] - 120, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if debug_mode and tracking:
            cv2.putText(frame_display, "DEBUG ON", (frame_display.shape[1] - 120, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow('Pendulum Tracker', frame_display)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset
            pivot_point = None
            bob_template = None
            last_bob_pos = None
            tracking = False
            positions_history.clear()
            timestamps_history.clear()
            try:
                cv2.destroyWindow('Bob Template')
            except:
                pass
            print("\nReset! Click to setup again.")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('s') and len(positions_history) > 0:
            # Save data
            filename = f"pendulum_data_{int(time.time())}.csv"
            with open(filename, 'w') as f:
                f.write("timestamp,bob_x,bob_y,angle,length\n")
                for i, ((bx, by), t) in enumerate(zip(positions_history, timestamps_history)):
                    angle, length = calculate_pendulum_angle(pivot_point, (bx, by))
                    f.write(f"{t},{bx},{by},{angle},{length}\n")
            print(f"Saved {len(positions_history)} data points to {filename}")
        
        frame_count += 1

finally:
    cam.end()
    cv2.destroyAllWindows()
    
    # Final statistics
    if len(frame_times) > 0:
        print(f"\n=== Performance Statistics ===")
        print(f"Total frames: {frame_count}")
        print(f"Average FPS: {1.0/np.mean(frame_times):.1f}")
        if len(tracking_times) > 0:
            print(f"Average tracking time: {np.mean(tracking_times)*1000:.2f}ms")
            print(f"Min tracking time: {np.min(tracking_times)*1000:.2f}ms")
            print(f"Max tracking time: {np.max(tracking_times)*1000:.2f}ms")
    
    print("\nCamera closed.")
