import cv2
from ultralytics import YOLO
import numpy as np

class DroneVision:
    def __init__(self):
        # Use YOLOv8n (nano - fastest for edge)
        self.model = YOLO('yolov8n.pt')
        self.cap = cv2.VideoCapture(0)
        self.target_detected = False
        self.target_position = None
        
    def detect_objects(self, frame):
        """Run detection on frame"""
        results = self.model(frame, conf=0.5)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Filter for person class (class 0 in COCO)
                if cls == 0:  # Person
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'confidence': conf
                    })
        
        return detections
    
    def get_target_position(self, frame):
        """Get highest confidence person detection"""
        detections = self.detect_objects(frame)
        
        if detections:
            # Sort by confidence
            best = max(detections, key=lambda x: x['confidence'])
            self.target_detected = True
            self.target_position = best['center']
            return best
        else:
            self.target_detected = False
            self.target_position = None
            return None
    
    def draw_detections(self, frame, detections):
        """Visualize detections"""
        if detections:
            x1, y1, x2, y2 = detections['bbox']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(detections['center'][0]), int(detections['center'][1])), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Target: {detections['confidence']:.2f}", 
                       (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def run(self):
        """Main loop"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            target = self.get_target_position(frame)
            frame = self.draw_detections(frame, target)
            
            cv2.imshow('Drone Vision', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    vision = DroneVision()
    vision.run()

    print("Target Detected:", vision.target_detected)