import cv2
import numpy as np

# Global variables
points = []
templates = []
template_size = 30
search_area = 80
rod_template_width = 60   # Width for rod template (longer)
rod_template_height = 20  # Height for rod template (thinner)
tracking = False
last_positions = []
debug_mode = False
rotation_step = 15  # Check every 15 degrees

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to capture initial clicks"""
    global points, templates, tracking, last_positions
    
    if event == cv2.EVENT_LBUTTONDOWN and not tracking:
        if len(points) < 2:
            points.append((x, y))
            
            if len(points) == 1:
                print(f"Pivot point selected at: ({x}, {y})")
            else:
                print(f"Rod point selected at: ({x}, {y})")
            
            if len(points) == 2:
                # Extract templates
                frame = param
                
                # Pivot template (square)
                px, py = points[0]
                half_size = template_size // 2
                y1 = max(0, py - half_size)
                y2 = min(frame.shape[0], py + half_size)
                x1 = max(0, px - half_size)
                x2 = min(frame.shape[1], px + half_size)
                pivot_template = frame[y1:y2, x1:x2].copy()
                templates.append(pivot_template)
                last_positions.append((px, py))
                
                # Rod template (rectangular)
                rx, ry = points[1]
                half_w = rod_template_width // 2
                half_h = rod_template_height // 2
                y1 = max(0, ry - half_h)
                y2 = min(frame.shape[0], ry + half_h)
                x1 = max(0, rx - half_w)
                x2 = min(frame.shape[1], rx + half_w)
                rod_template = frame[y1:y2, x1:x2].copy()
                templates.append(rod_template)
                last_positions.append((rx, ry))
                
                print(f"Pivot template: {pivot_template.shape}")
                print(f"Rod template: {rod_template.shape}")
                print("\nTracking started! Press 'r' to reset, 'q' to quit")
                
                tracking = True
                cv2.imshow('Pivot Template', pivot_template)
                cv2.imshow('Rod Template', rod_template)

def rotate_image(image, angle):
    """Rotate image by angle around its center"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated

def find_template_in_region(frame, template, center, search_size):
    """Standard template matching in a region"""
    cx, cy = center
    half_search = search_size // 2
    
    y1 = max(0, cy - half_search)
    y2 = min(frame.shape[0], cy + half_search)
    x1 = max(0, cx - half_search)
    x2 = min(frame.shape[1], cx + half_search)
    
    search_region = frame[y1:y2, x1:x2]
    
    if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
        return None
    
    result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val > 0.5:
        match_x = x1 + max_loc[0] + template.shape[1] // 2
        match_y = y1 + max_loc[1] + template.shape[0] // 2
        return (match_x, match_y, max_val, 0)  # angle=0 for non-rotated
    
    return None

def find_template_with_rotation(frame, template, center, search_size, rotation_step=15):
    """Template matching with rotation testing"""
    cx, cy = center
    half_search = search_size // 2
    
    y1 = max(0, cy - half_search)
    y2 = min(frame.shape[0], cy + half_search)
    x1 = max(0, cx - half_search)
    x2 = min(frame.shape[1], cx + half_search)
    
    search_region = frame[y1:y2, x1:x2]
    
    if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
        return None
    
    best_match = None
    best_score = 0.5  # Minimum threshold
    best_angle = 0
    
    # Try different rotations
    for angle in range(-90, 91, rotation_step):
        rotated_template = rotate_image(template, angle)
        
        # Check if rotated template fits in search region
        if (rotated_template.shape[0] > search_region.shape[0] or 
            rotated_template.shape[1] > search_region.shape[1]):
            continue
        
        result = cv2.matchTemplate(search_region, rotated_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_match = max_loc
            best_angle = angle
    
    if best_match is not None:
        match_x = x1 + best_match[0] + template.shape[1] // 2
        match_y = y1 + best_match[1] + template.shape[0] // 2
        return (match_x, match_y, best_score, best_angle)
    
    return None

# Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 70)

print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print("\n=== PENDULUM TRACKER (ROTATING TEMPLATE) ===")
print("Instructions:")
print("1. Click on the PIVOT point first")
print("2. Click on the ROD (middle of the bar)")
print("3. Tracking will start automatically")
print("4. Press 'd' to toggle debug mode")
print("5. Press 'r' to reset and select new points")
print("6. Press 'q' to quit\n")

cv2.namedWindow('Pendulum Tracker')

# Read first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read from camera")
    exit()

cv2.setMouseCallback('Pendulum Tracker', mouse_callback, first_frame)

# Show first frame with instructions
frame_display = first_frame.copy()
cv2.putText(frame_display, "Click on pivot and rod", (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imshow('Pendulum Tracker', frame_display)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_display = frame.copy()
    
    if not tracking:
        cv2.setMouseCallback('Pendulum Tracker', mouse_callback, frame)
        
        for i, (px, py) in enumerate(points):
            color = (0, 255, 0) if i == 0 else (255, 0, 255)
            cv2.circle(frame_display, (px, py), 8, color, 2)
        
        if len(points) == 0:
            cv2.putText(frame_display, "Click on PIVOT point", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif len(points) == 1:
            cv2.putText(frame_display, "Click on ROD", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    else:
        # Track pivot point (no rotation)
        pivot_result = find_template_in_region(frame, templates[0], last_positions[0], search_area)
        
        if pivot_result is not None:
            pivot_x, pivot_y, conf, _ = pivot_result
            last_positions[0] = (pivot_x, pivot_y)
            
            # Draw pivot
            cv2.circle(frame_display, (pivot_x, pivot_y), 8, (0, 255, 0), -1)
            cv2.circle(frame_display, (pivot_x, pivot_y), 12, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Pivot", (pivot_x + 15, pivot_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if debug_mode:
                half = search_area // 2
                cv2.rectangle(frame_display,
                            (pivot_x - half, pivot_y - half),
                            (pivot_x + half, pivot_y + half),
                            (0, 255, 0), 1)
        else:
            cv2.putText(frame_display, "Pivot: LOST", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Track rod point (with rotation)
        rod_result = find_template_with_rotation(frame, templates[1], last_positions[1], 
                                                  search_area, rotation_step)
        
        if rod_result is not None:
            rod_x, rod_y, conf, angle = rod_result
            last_positions[1] = (rod_x, rod_y)
            
            # Draw rod point
            cv2.circle(frame_display, (rod_x, rod_y), 8, (255, 0, 255), -1)
            cv2.circle(frame_display, (rod_x, rod_y), 12, (255, 0, 255), 2)
            cv2.putText(frame_display, f"Rod (rot: {angle:.0f}°)", (rod_x + 15, rod_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            if debug_mode:
                half = search_area // 2
                cv2.rectangle(frame_display,
                            (rod_x - half, rod_y - half),
                            (rod_x + half, rod_y + half),
                            (255, 0, 255), 1)
            
            # Draw line from pivot to rod if both are tracked
            if pivot_result is not None:
                cv2.line(frame_display, last_positions[0], last_positions[1], 
                        (0, 255, 255), 2)
                
                # Calculate pendulum angle from vertical
                dx = rod_x - pivot_x
                dy = rod_y - pivot_y
                pendulum_angle = np.degrees(np.arctan2(dx, dy))
                distance = np.sqrt(dx**2 + dy**2)
                
                cv2.putText(frame_display, f"Pendulum Angle: {pendulum_angle:.1f}°", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_display, f"Length: {distance:.1f}px", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame_display, "Rod: LOST", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    if debug_mode:
        cv2.putText(frame_display, "DEBUG MODE ON", (10, frame_display.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imshow('Pendulum Tracker', frame_display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        points = []
        templates = []
        last_positions = []
        tracking = False
        try:
            cv2.destroyWindow('Pivot Template')
            cv2.destroyWindow('Rod Template')
        except:
            pass
        print("\nReset! Click on new points.")
    elif key == ord('d'):
        debug_mode = not debug_mode
        status = "ON" if debug_mode else "OFF"
        print(f"Debug mode: {status}")

cap.release()
cv2.destroyAllWindows()
