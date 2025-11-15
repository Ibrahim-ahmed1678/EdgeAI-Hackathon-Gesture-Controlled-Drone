from vision_detector import DroneVision
from swarm_controller import DroneSwarm
import threading
import time
import cv2

class SwarmVisionSystem:
    def __init__(self):
        self.vision = DroneVision()
        self.swarm = DroneSwarm(num_drones=3)
        self.running = True
        self.current_target = None
        
    def vision_thread(self):
        """Run CV in separate thread"""
        while self.running:
            ret, frame = self.vision.cap.read()
            if not ret:
                continue
            
            target = self.vision.get_target_position(frame)
            
            if target:
                # Convert pixel coordinates to world coordinates
                # This is simplified - you'd need camera calibration for real
                frame_center_x = frame.shape[1] / 2
                frame_center_y = frame.shape[0] / 2
                
                # Rough conversion (adjust based on camera FOV and altitude)
                target_x = (target['center'][0] - frame_center_x) / 50
                target_y = (target['center'][1] - frame_center_y) / 50
                
                self.current_target = (target_x, target_y)
            else:
                self.current_target = None
            
            # Display
            frame = self.vision.draw_detections(frame, target)
            cv2.imshow('Swarm Vision', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
    
    def swarm_control_thread(self):
        """Control swarm based on vision"""
        # Initial setup
        print("Setting up swarm...")
        self.swarm.arm_all()
        self.swarm.takeoff_all(altitude=10)
        time.sleep(5)
        
        # Start in grid formation
        self.swarm.formation_grid(spacing=5)
        time.sleep(5)
        
        mode = "search"
        last_target_time = time.time()
        
        while self.running:
            if self.current_target:
                # Target detected - orbit it
                if mode != "track":
                    print("TARGET DETECTED - Switching to orbit mode")
                    mode = "track"
                
                target_x, target_y = self.current_target
                self.swarm.orbit_target(target_x, target_y, radius=8, altitude=10)
                last_target_time = time.time()
                
            else:
                # No target - return to search pattern after timeout
                if mode == "track" and (time.time() - last_target_time > 5):
                    print("Target lost - Returning to search pattern")
                    mode = "search"
                    self.swarm.formation_circle(radius=10)
            
            time.sleep(0.5)
        
        # Cleanup
        print("Landing swarm...")
        self.swarm.land_all()
    
    def run(self):
        """Start the system"""
        # Start threads
        vision_t = threading.Thread(target=self.vision_thread)
        swarm_t = threading.Thread(target=self.swarm_control_thread)
        
        vision_t.start()
        time.sleep(2)  # Let vision start first
        swarm_t.start()
        
        # Wait for completion
        vision_t.join()
        swarm_t.join()
        
        self.vision.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Swarm Vision System...")
    system = SwarmVisionSystem()
    system.run()