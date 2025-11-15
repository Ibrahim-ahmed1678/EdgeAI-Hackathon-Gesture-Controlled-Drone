from flask import Flask, Response
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
model.to("cuda")
app = Flask(__name__)

# --- CSI CAMERA OPENCV PIPELINE (modify width/height if needed) ---
def gstreamer_pipeline(
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=30,
        flip_method=0):
    return (
            "nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
            f"format=NV12, framerate={framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink"
            )


def detect_save_me_gesture(keypoints_xy, margin=20):
    """
    keypoints_xy: numpy array of shape (17, 2) -> (x, y) for each keypoint
    margin: pixels above the nose to be considered "above head"
    Returns True if both wrists are above the head.
    """
    if keypoints_xy is None or keypoints_xy.shape[0] < 11:
        return False

    # COCO-style indices for YOLO pose:
    # 0: nose, 9: left wrist, 10: right wrist
    nose_x, nose_y = keypoints_xy[0]
    lw_x, lw_y = keypoints_xy[9]
    rw_x, rw_y = keypoints_xy[10]

    # y is smaller => higher on the image
    hands_above_head = (lw_y < nose_y - margin) and (rw_y < nose_y - margin)
    return hands_above_head

# Open camera
camera = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            continue



        # Run pose inference
        results = model(frame, verbose=False)

        # Take first result (current frame)
        res = results[0]

        # Draw only skeleton/keypoints, no boxes, no labels, no confidences
        annotated_frame = res.plot(
                boxes=False,    # hide boxes
                labels=False,   # hide class labels
                conf=False      # hide confidence numbers
                )

        gesture_text = ""

        # If we have at least one person with keypoints
        if res.keypoints is not None and len(res.keypoints) > 0:
            # Take the first detected person
            kps = res.keypoints.xy[0].cpu().numpy()  # shape (17, 2)

            if detect_save_me_gesture(kps):
                gesture_text = "SAVE ME"

        # Overlay our custom text instead of YOLO labels
        if gesture_text:
            h, w, _ = annotated_frame.shape
            cv2.putText(
                    annotated_frame,
                    gesture_text,
                    (int(w * 0.3), int(h * 0.1)),  # near top-center
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA
                    )
            cv2.putText(
                    annotated_frame,
                    "Press 'q' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                    )

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)

        frame_bytes = buffer.tobytes()

        # Yield frame as multipart for MJPEG stream
        yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )

@app.route('/video')
def video():
    return Response(generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h2>CSI Camera Stream</h2><img src='/video'>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

