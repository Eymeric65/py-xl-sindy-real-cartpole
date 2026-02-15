import cv2
import numpy as np
import time
import json
try:
    import pseyepy
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pseyepy'))
    import pseyepy

# ============================================================================
# CARTPOLE BRIGHTNESS TRACKER - Finds bright dots on black background
# ============================================================================

# Configuration
BRIGHTNESS_THRESHOLD = 50  # Threshold for bright dots (0-255)
MIN_CLUSTER_SIZE = 2        # Minimum pixels in a cluster
MAX_CLUSTER_SIZE = 40      # Maximum pixels in a cluster

# Camera settings
FPS_TARGET = 187
EXPOSURE = 15
GAIN = 0

# Tracking state
last_cart_pos = None
last_pole_pos = None
tracking = False
debug_mode = True
setup_complete = False

# Data recording
cart_positions = []
pole_positions = []
timestamps = []
angles = []

def mouse_callback(event, x, y, flags, param):
    """Click to select cart and pole initial positions"""
    global last_cart_pos, last_pole_pos, tracking, setup_complete
    
    if event == cv2.EVENT_LBUTTONDOWN and not setup_complete:
        if last_cart_pos is None:
            last_cart_pos = (x, y)
            print(f"Cart set at ({x}, {y}). Now click POLE TIP.")
        elif last_pole_pos is None:
            last_pole_pos = (x, y)
            setup_complete = True
            tracking = True
            print(f"Pole set at ({x}, {y}). TRACKING STARTED!")
            print("Controls: 'd'=debug, 's'=save, 'r'=reset, 'q'=quit")

def find_bright_clusters(frame):
    """Find bright clusters in frame and return their centroids"""
    # Threshold bright pixels
    _, binary = cv2.threshold(frame, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_CLUSTER_SIZE <= area <= MAX_CLUSTER_SIZE:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy, area))
    
    return centroids

def assign_clusters_to_markers(centroids, last_cart, last_pole):
    """Assign detected clusters to cart or pole based on proximity to last positions"""
    if len(centroids) == 0:
        return None, None
    
    if len(centroids) == 1:
        # Only one cluster found - assign to closest marker
        cx, cy, area = centroids[0]
        dist_cart = np.sqrt((cx - last_cart[0])**2 + (cy - last_cart[1])**2)
        dist_pole = np.sqrt((cx - last_pole[0])**2 + (cy - last_pole[1])**2)
        
        if dist_cart < dist_pole:
            return (cx, cy), None
        else:
            return None, (cx, cy)
    
    # Multiple clusters - assign uniquely to cart and pole
    # Find closest centroid to cart
    cart_pos = None
    cart_idx = None
    min_cart_dist = float('inf')
    
    for i, (cx, cy, area) in enumerate(centroids):
        dist = np.sqrt((cx - last_cart[0])**2 + (cy - last_cart[1])**2)
        if dist < min_cart_dist:
            min_cart_dist = dist
            cart_pos = (cx, cy)
            cart_idx = i
    
    # Find closest centroid to pole (excluding cart's centroid)
    pole_pos = None
    min_pole_dist = float('inf')
    
    for i, (cx, cy, area) in enumerate(centroids):
        if i == cart_idx:
            continue
        dist = np.sqrt((cx - last_pole[0])**2 + (cy - last_pole[1])**2)
        if dist < min_pole_dist:
            min_pole_dist = dist
            pole_pos = (cx, cy)
    
    return cart_pos, pole_pos

def calculate_pole_angle(cart_pos, pole_pos):
    """Calculate pole angle from vertical (0° = up, + = clockwise)"""
    dx = pole_pos[0] - cart_pos[0]
    dy = cart_pos[1] - pole_pos[1]  # Flip Y
    angle = np.degrees(np.arctan2(dx, dy))
    length = np.sqrt(dx**2 + dy**2)
    return angle, length

def estimate_velocities(cart_hist, pole_hist, time_hist):
    """Estimate cart and pole velocities"""
    if len(cart_hist) < 3:
        return 0.0, 0.0
    
    n = min(5, len(cart_hist))
    cart_array = np.array(cart_hist[-n:])
    pole_array = np.array(pole_hist[-n:])
    time_array = np.array(time_hist[-n:])
    
    dt = time_array[-1] - time_array[0]
    if dt < 1e-6:
        return 0.0, 0.0
    
    cart_vx = (cart_array[-1, 0] - cart_array[0, 0]) / dt
    
    angles_list = [calculate_pole_angle(cart_array[i], pole_array[i])[0] for i in range(n)]
    angle_velocity = (angles_list[-1] - angles_list[0]) / dt
    
    return cart_vx, angle_velocity

def save_data(filename='cartpole_data.json'):
    """Save tracking data to JSON"""
    data = {
        'cart_positions': [[float(x), float(y)] for x, y in cart_positions],
        'pole_positions': [[float(x), float(y)] for x, y in pole_positions],
        'timestamps': [float(t) for t in timestamps],
        'angles': [float(a) for a in angles],
        'config': {'fps': FPS_TARGET, 'exposure': EXPOSURE, 'gain': GAIN}
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(timestamps)} frames to {filename}")

# ============================================================================
# MAIN
# ============================================================================

print(f"Initializing PS3Eye: {FPS_TARGET}fps, Exp={EXPOSURE}, Gain={GAIN}")

try:
    cam = pseyepy.Camera(ids=[0], resolution=pseyepy.Camera.RES_SMALL, 
                        fps=FPS_TARGET, colour=False)
    cam.auto_gain = cam.auto_exposure = cam.auto_whitebalance = False
    cam.exposure = EXPOSURE
    cam.gain = GAIN
    print("Camera ready!")
except Exception as e:
    print(f"Camera init failed: {e}")
    exit(1)

print("\n=== CARTPOLE TRACKER ===")
print("1. Click CART marker\n2. Click POLE TIP marker")

cv2.namedWindow('Cartpole Tracker')
frame, _ = cam.read()
first_frame = frame.copy()
cv2.setMouseCallback('Cartpole Tracker', mouse_callback, first_frame)
cv2.putText(first_frame, "1. Click CART marker", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
cv2.imshow('Cartpole Tracker', first_frame)

last_time = time.time()
frame_count = 0

try:
    while True:
        frame, timestamp = cam.read()
        
        if tracking:
            # Find bright clusters
            centroids = find_bright_clusters(frame)
            
            # Assign clusters to cart and pole
            cart_result, pole_result = assign_clusters_to_markers(centroids, last_cart_pos, last_pole_pos)
            
            # Update positions
            if cart_result:
                cart_x, cart_y = cart_result
                last_cart_pos = (cart_x, cart_y)
            else:
                cart_x, cart_y = last_cart_pos
            
            if pole_result:
                pole_x, pole_y = pole_result
                last_pole_pos = (pole_x, pole_y)
            else:
                pole_x, pole_y = last_pole_pos
            
            # Calculate and record
            angle, length = calculate_pole_angle(last_cart_pos, last_pole_pos)
            cart_positions.append(last_cart_pos)
            pole_positions.append(last_pole_pos)
            timestamps.append(timestamp)
            angles.append(angle)
            
            cart_vx, angle_vel = estimate_velocities(cart_positions, pole_positions, timestamps)
            
            # Display
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            if debug_mode:
                # Show thresholded image
                _, binary = cv2.threshold(frame, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
                binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                
                # Draw all detected centroids in blue
                for cx, cy, area in centroids:
                    cv2.circle(binary_bgr, (cx, cy), 5, (255, 0, 0), 1)
                
                # Show binary view in corner
                h, w = binary_bgr.shape[:2]
                small = cv2.resize(binary_bgr, (w//3, h//3))
                display[0:h//3, 0:w//3] = small
            
            cv2.circle(display, (cart_x, cart_y), 8, (0, 255, 0), 2)
            cv2.circle(display, (pole_x, pole_y), 8, (0, 0, 255), 2)
            cv2.line(display, (cart_x, cart_y), (pole_x, pole_y), (255, 255, 255), 2)
            
            cv2.putText(display, f"Clusters:{len(centroids)} Angle:{angle:5.1f} Len:{length:5.1f}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display, f"Cart:({cart_x},{cart_y}) Pole:({pole_x},{pole_y})", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(display, f"CartVx:{cart_vx:5.1f} AngleVel:{angle_vel:5.1f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            frame_count += 1
            if time.time() - last_time >= 1.0:
                fps = frame_count / (time.time() - last_time)
                print(f"FPS: {fps:.1f} | Frames: {len(timestamps)}")
                last_time = time.time()
                frame_count = 0
            
            cv2.imshow('Cartpole Tracker', display)
        else:
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if last_cart_pos is not None:
                cv2.circle(display, last_cart_pos, 10, (0, 255, 0), 2)
                cv2.putText(display, "2. Click POLE TIP", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            else:
                cv2.putText(display, "1. Click CART", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            cv2.imshow('Cartpole Tracker', display)
            cv2.setMouseCallback('Cartpole Tracker', mouse_callback, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            last_cart_pos = last_pole_pos = None
            tracking = False
            setup_complete = False
            cart_positions = pole_positions = timestamps = angles = []
            print("Reset tracking")
        elif key == ord('d'):
            debug_mode = not debug_mode
        elif key == ord('s') and timestamps:
            save_data(f'cartpole_data_{time.strftime("%Y%m%d_%H%M%S")}.json')

except KeyboardInterrupt:
    print("\nInterrupted")
finally:
    print("Cleaning up...")
    cv2.destroyAllWindows()
    time.sleep(0.1)  # Give time for windows to close
    try:
        cam.end()
    except:
        pass  # Ignore cleanup errors
    
    if timestamps and input("Save data? (y/n): ").lower() == 'y':
        save_data(f'cartpole_data_{time.strftime("%Y%m%d_%H%M%S")}.json')
    print("Done!")
