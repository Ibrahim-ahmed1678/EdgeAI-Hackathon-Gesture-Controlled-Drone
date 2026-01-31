import cv2
from ultralytics import YOLO
import numpy as np

def detect_save_me_gesture(keypoints_xy, margin=20):
    """
    Hands above head gesture.
    keypoints_xy: numpy array of shape (17, 2) -> (x, y) for each keypoint.
    margin: pixels above the nose to be considered "above head".
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

def detect_danger_gesture(keypoints_xy, margin_above_nose=20, shoulder_scale=0.8):
    """
    Arms crossed in an X near chest: show 'DANGER'.

    Logic:
    - Use shoulder distance as a scale reference.
    - Each wrist is close to the opposite shoulder.
    - Hands are not significantly above the nose (to avoid SAVE ME).
    """
    if keypoints_xy is None or keypoints_xy.shape[0] < 11:
        return False

    # Indices:
    # 0: nose, 5: left shoulder, 6: right shoulder, 9: left wrist, 10: right wrist
    nose_x, nose_y = keypoints_xy[0]
    ls_x, ls_y = keypoints_xy[5]   # left shoulder
    rs_x, rs_y = keypoints_xy[6]   # right shoulder
    lw_x, lw_y = keypoints_xy[9]   # left wrist
    rw_x, rw_y = keypoints_xy[10]  # right wrist

    # Shoulder distance as scale
    shoulder_dist = np.linalg.norm(np.array([ls_x, ls_y]) - np.array([rs_x, rs_y]))
    if shoulder_dist <= 1e-3:
        return False

    # Distances from wrists to opposite shoulders
    dist_lw_to_rs = np.linalg.norm(np.array([lw_x, lw_y]) - np.array([rs_x, rs_y]))
    dist_rw_to_ls = np.linalg.norm(np.array([rw_x, rw_y]) - np.array([ls_x, ls_y]))

    close_cross = (dist_lw_to_rs < shoulder_scale * shoulder_dist) and \
                  (dist_rw_to_ls < shoulder_scale * shoulder_dist)

    # Hands not high above nose (chest area, not above head)
    hands_not_high = (lw_y > nose_y - margin_above_nose) and (rw_y > nose_y - margin_above_nose)

    return close_cross and hands_not_high

def detect_need_food_wrist(keypoints_xy,
                           mouth_offset_scale=0.3,
                           mouth_radius_scale=0.4):
    """
    'NEED FOOD' using wrists near mouth/chin.

    Approximation:
    - Estimate a mouth point slightly below the nose:
        mouth_y = nose_y + mouth_offset_scale * shoulder_dist
        mouth_x = nose_x
    - If either wrist is within mouth_radius_scale * shoulder_dist of this point,
      we say the hand is at the mouth.
    """
    if keypoints_xy is None or keypoints_xy.shape[0] < 11:
        return False

    # Indices:
    # 0: nose, 5: left shoulder, 6: right shoulder, 9: left wrist, 10: right wrist
    nose_x, nose_y = keypoints_xy[0]
    ls_x, ls_y = keypoints_xy[5]
    rs_x, rs_y = keypoints_xy[6]
    lw_x, lw_y = keypoints_xy[9]
    rw_x, rw_y = keypoints_xy[10]

    # Shoulder distance as scale reference
    shoulder_dist = np.linalg.norm(np.array([ls_x, ls_y]) - np.array([rs_x, rs_y]))
    if shoulder_dist <= 1e-3:
        return False

    # Approximate mouth position: below nose toward chin
    mouth_x = nose_x
    mouth_y = nose_y + mouth_offset_scale * shoulder_dist  # y increases downward

    mouth = np.array([mouth_x, mouth_y])
    lw = np.array([lw_x, lw_y])
    rw = np.array([rw_x, rw_y])

    dist_lw_to_mouth = np.linalg.norm(lw - mouth)
    dist_rw_to_mouth = np.linalg.norm(rw - mouth)

    threshold = mouth_radius_scale * shoulder_dist

    hand_near_mouth = (dist_lw_to_mouth < threshold) or (dist_rw_to_mouth < threshold)
    return hand_near_mouth

def main():
    # Load YOLOv8 nano pose model
    model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Mirror the frame so it behaves like a mirror
        frame = cv2.flip(frame, 1)

        # Run pose inference
        results = model(frame, verbose=False)
        res = results[0]

        # Draw only skeleton/keypoints, no boxes, no labels, no confidences
        annotated_frame = res.plot(
            boxes=False,    # hide bounding boxes
            labels=False,   # hide class labels
            conf=False      # hide confidence numbers
        )

        gesture_text = ""

        if res.keypoints is not None and len(res.keypoints) > 0:
            # For now, just consider the first detected person
            kps = res.keypoints.xy[0].cpu().numpy()  # shape (17, 2)

            # Priority: SAVE ME > DANGER > NEED FOOD
            if detect_save_me_gesture(kps):
                gesture_text = "SAVE ME"
            elif detect_danger_gesture(kps):
                gesture_text = "DANGER"
            elif detect_need_food_wrist(kps):
                gesture_text = "NEED FOOD"

        # Show gesture text if any
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

        cv2.imshow("YOLO Pose - Custom Gestures (Mirrored)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()