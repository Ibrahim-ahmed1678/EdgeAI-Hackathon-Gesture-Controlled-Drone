import cv2
import numpy as np

def create_tactical_display(frame, detections, swarm_status):
    """Add cool HUD overlay"""
    h, w = frame.shape[:2]
    
    # Create overlay
    overlay = frame.copy()
    
    # Draw grid
    for i in range(0, w, 50):
        cv2.line(overlay, (i, 0), (i, h), (0, 255, 0), 1, cv2.LINE_AA)
    for i in range(0, h, 50):
        cv2.line(overlay, (0, i), (w, i), (0, 255, 0), 1, cv2.LINE_AA)
    
    # Blend
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Status panel
    cv2.rectangle(frame, (10, 10), (300, 150), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (300, 150), (0, 255, 0), 2)
    
    y_offset = 35
    cv2.putText(frame, "SWARM VISION SYSTEM", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    y_offset += 30
    status = "TRACKING" if detections else "SEARCHING"
    color = (0, 255, 0) if detections else (0, 165, 255)
    cv2.putText(frame, f"Status: {status}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    y_offset += 25
    cv2.putText(frame, f"Drones Active: {swarm_status['active']}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(frame, f"Targets: {swarm_status['targets']}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame
# ```

# ---

# ## **CRITICAL SUCCESS TIPS**

# 1. **Test EVERYTHING incrementally** - Don't wait until hour 24 to test integration
# 2. **Have fallback demos** - Record good runs in case live demo fails
# 3. **Practice your pitch** - 3 minutes explaining what it does and WHY it matters
# 4. **Real-world application** - "This could be used for search & rescue, finding lost hikers"
# 5. **Error handling** - Add try/except everywhere, the demo WILL have issues

# ## **Demo Script** (Practice This)
# ```
# "What you're seeing is a drone swarm with collective vision. 

# [Show CV feed detecting person]

# When our computer vision detects a target, the swarm autonomously 
# repositions to orbit and track it. 

# [Swarm moves into orbit]

# If the target is lost, they return to search pattern. 

# [Target disappears, swarm returns to grid]

# This combines real-time edge AI on the Jetson with distributed 
# drone control via MAVLink. Applications include search & rescue, 
# perimeter security, and autonomous surveillance."