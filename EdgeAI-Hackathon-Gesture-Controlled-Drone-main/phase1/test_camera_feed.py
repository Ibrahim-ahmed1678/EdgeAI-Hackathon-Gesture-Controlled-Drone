import cv2

# Test camera capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Can't open camera!")
    exit()

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Test Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()