import cv2
import numpy as np
import pandas as pd
import time
try:
    import pseyepy
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pseyepy'))
    import pseyepy

# ============================================================================
# MINIMAL CARTPOLE TRACKER FOR SINDY - Records position + force data
# ============================================================================

# Camera settings
FPS_TARGET = 187
EXPOSURE = 50
GAIN = 0
BRIGHTNESS_THRESHOLD = 50
MIN_CLUSTER_SIZE = 2
MAX_CLUSTER_SIZE = 100

# Arduino connection for force output
import serial
try:
    arduino = serial.Serial('COM3', 115200, timeout=10)  # Adjust COM port
    print("Arduino connected on COM3")
except Exception as e:
    arduino = None
    print(f"Arduino not found - running without force control: {e}")

# State
last_cart_pos = None
last_pole_pos = None
recording = False
current_force = 0  # Current force being applied

# For velocity computation
prev_cart_x = None
prev_angle = None
prev_time = None
angle_unwrapped = 0.0  # Unwrapped angle to handle wrapping

# Data storage - SINDy format with velocities computed in real-time
data = {
    'time': [],       # Time (s)
    'cart_x': [],     # Cart horizontal position (px)
    'cart_vx': [],    # Cart velocity (px/s)
    'angle': [],      # Pole angle unwrapped (rad)
    'angle_vel': [],  # Angular velocity (rad/s)
    'force': []       # Applied force (sent to Arduino)
}

def find_bright_clusters(frame):
    """Find bright dot centroids"""
    _, binary = cv2.threshold(frame, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_CLUSTER_SIZE <= area <= MAX_CLUSTER_SIZE:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
    return centroids

def assign_markers(centroids, last_cart, last_pole):
    """Assign clusters to cart and pole by proximity"""
    if len(centroids) == 0:
        return None, None
    if len(centroids) == 1:
        return centroids[0], None
    
    # Find closest to cart
    dists = [np.hypot(c[0]-last_cart[0], c[1]-last_cart[1]) for c in centroids]
    cart_idx = np.argmin(dists)
    cart_pos = centroids[cart_idx]
    
    # Find closest to pole (excluding cart)
    pole_centroids = [c for i, c in enumerate(centroids) if i != cart_idx]
    if pole_centroids:
        dists = [np.hypot(c[0]-last_pole[0], c[1]-last_pole[1]) for c in pole_centroids]
        pole_pos = pole_centroids[np.argmin(dists)]
    else:
        pole_pos = None
    
    return cart_pos, pole_pos

def calculate_angle(cart_pos, pole_pos):
    """Calculate angle from vertical (radians)"""
    dx = pole_pos[0] - cart_pos[0]
    dy = cart_pos[1] - pole_pos[1]  # Flip Y
    angle = np.arctan2(dx, dy)  # radians, 0 = up
    length = np.hypot(dx, dy)
    return angle, length

def send_force(force_value):
    """Send force command to Arduino in format T{value}"""
    global current_force
    current_force = force_value
    if arduino:
        command = f"T{int(force_value)}\n"
        arduino.write(command.encode())
        print(f"Sent to Arduino: {command.strip()}")
    return current_force

def mouse_callback(event, x, y, flags, param):
    """Click to set initial positions"""
    global last_cart_pos, last_pole_pos, recording
    
    if event == cv2.EVENT_LBUTTONDOWN and not recording:
        if last_cart_pos is None:
            last_cart_pos = (x, y)
            print(f"Cart: {last_cart_pos}. Click pole tip.")
        elif last_pole_pos is None:
            last_pole_pos = (x, y)
            recording = True
            print(f"Pole: {last_pole_pos}. Recording! Press 's' to save, 'q' to quit.")

def export_dataframe(filename='cartpole_data.csv'):
    """Export data as pandas DataFrame for SINDy"""
    df = pd.DataFrame(data)
    
    df.to_csv(filename, index=False)
    print(f"\nExported {len(df)} frames to {filename}")
    print(f"Duration: {df['time'].iloc[-1] - df['time'].iloc[0]:.2f}s")
    print(f"Columns: {list(df.columns)}")
    return df

# ============================================================================
# MAIN LOOP
# ============================================================================

print(f"Initializing camera: {FPS_TARGET}fps")
cam = pseyepy.Camera(ids=[0], resolution=pseyepy.Camera.RES_SMALL, 
                    fps=FPS_TARGET, colour=False)
cam.auto_gain = cam.auto_exposure = cam.auto_whitebalance = False
cam.exposure = EXPOSURE
cam.gain = GAIN
print("Camera ready!")

print("\n=== CARTPOLE SINDY RECORDER ===")
print("1. Click CART marker")
print("2. Click POLE TIP marker")
print("3. Control force: Up/Down arrows (+/-5), '0' to zero")
print("4. Press 's' to save, 'r' to reset, 'q' to quit")
print("\nForce commands sent to Arduino as 'T{value}'")

cv2.namedWindow('Tracker')
frame, _ = cam.read()
cv2.setMouseCallback('Tracker', mouse_callback)

start_time = None
frame_count = 0
fps_time = time.time()

try:
    while True:
        frame, timestamp = cam.read()
        
        if recording:
            if start_time is None:
                start_time = timestamp
                prev_time = timestamp
            
            # Track markers
            centroids = find_bright_clusters(frame)
            cart_pos, pole_pos = assign_markers(centroids, last_cart_pos, last_pole_pos)
            
            if cart_pos:
                last_cart_pos = cart_pos
            if pole_pos:
                last_pole_pos = pole_pos
            
            # Calculate state
            angle_raw, length = calculate_angle(last_cart_pos, last_pole_pos)
            
            # Unwrap angle to avoid modulo jumps (-π to π)
            if prev_angle is not None:
                # Calculate difference in raw angle
                angle_diff = angle_raw - (prev_angle - angle_unwrapped)
                # Detect and correct wrapping
                if angle_diff > np.pi:
                    angle_unwrapped -= 2 * np.pi
                elif angle_diff < -np.pi:
                    angle_unwrapped += 2 * np.pi
            
            angle_current = angle_raw + angle_unwrapped
            
            # Compute velocities
            current_time = timestamp - start_time
            if prev_cart_x is not None and prev_angle is not None:
                dt = timestamp - prev_time
                if dt > 0:
                    cart_vx = (last_cart_pos[0] - prev_cart_x) / dt
                    angle_vel = (angle_current - prev_angle) / dt
                else:
                    cart_vx = 0.0
                    angle_vel = 0.0
            else:
                cart_vx = 0.0
                angle_vel = 0.0
            
            # Record data
            data['time'].append(current_time)
            data['cart_x'].append(last_cart_pos[0])
            data['cart_vx'].append(cart_vx)
            data['angle'].append(angle_current)
            data['angle_vel'].append(angle_vel)
            data['force'].append(current_force)
            
            # Update previous values
            prev_cart_x = last_cart_pos[0]
            prev_angle = angle_current
            prev_time = timestamp
            
            # Display
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.circle(display, last_cart_pos, 8, (0, 255, 0), 2)
            cv2.circle(display, last_pole_pos, 8, (0, 0, 255), 2)
            cv2.line(display, last_cart_pos, last_pole_pos, (255, 255, 255), 2)
            
            cv2.putText(display, f"Frames:{len(data['time'])} Angle:{np.degrees(angle_current):.1f}deg CartX:{last_cart_pos[0]}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, f"Force: T{int(current_force)} (Up/Down arrows to change)", 
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                print(f"FPS: {frame_count/(time.time()-fps_time):.1f} | Frames: {len(data['time'])}")
                fps_time = time.time()
                frame_count = 0
            
            cv2.imshow('Tracker', display)
        else:
            # Setup mode
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if last_cart_pos:
                cv2.circle(display, last_cart_pos, 10, (0, 255, 0), 2)
                cv2.putText(display, "2. Click POLE TIP", (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            else:
                cv2.putText(display, "1. Click CART", (10, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            cv2.imshow('Tracker', display)
            cv2.setMouseCallback('Tracker', mouse_callback)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            break
        elif key == ord('q') and len(data['time']) > 0:
            df = export_dataframe(f'cartpole_data_{time.strftime("%Y%m%d_%H%M%S")}.csv')
            break
        elif key == ord('r'):
            # Reset
            last_cart_pos = last_pole_pos = None
            recording = False
            start_time = None
            prev_cart_x = prev_angle = prev_time = None
            angle_unwrapped = 0.0
            for k in data:
                data[k] = []
            send_force(0)
            print("Reset!")
        elif key == ord('z'):  # Up arrow
            send_force(15)
        elif key == ord('s'):  # Down arrow
            send_force(-15)
        elif key == ord('0'):
            send_force(0)

except KeyboardInterrupt:
    print("\nInterrupted")
finally:
    print("Cleaning up...")
    if arduino:
        send_force(0)  # Reset force to 0
        arduino.close()
    cv2.destroyAllWindows()
    time.sleep(0.1)
    try:
        cam.end()
    except:
        pass
    
    # Auto-save if data exists
    if len(data['time']) > 0:
        response = input("Save data? (y/n): ").lower()
        if response == 'y':
            export_dataframe(f'cartpole_data_{time.strftime("%Y%m%d_%H%M%S")}.csv')
    
    print("Done!")
