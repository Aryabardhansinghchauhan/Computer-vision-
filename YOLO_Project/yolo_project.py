from ultralytics import YOLO
import cv2
import time

# Load YOLO model
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)

# Create resizable window
cv2.namedWindow("YOLO Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Object Detection", 800, 600)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Fix overlapping issue
    annotated_frame = results[0].plot().copy()

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Count objects (extra feature ⭐)
    objects_detected = len(results[0].boxes)

    # Display FPS
    cv2.putText(annotated_frame, f"FPS: {int(fps)}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Display object count
    cv2.putText(annotated_frame, f"Objects: {objects_detected}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)

    # Show output
    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()