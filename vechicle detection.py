import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' or 'yolov8m.pt' for better accuracy

# Open video file or capture from webcam
video_path = "C:/Users/priya/Downloads/WhatsApp Video 2025-03-04 at 10.12.11 AM.mp4"  # Change to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define video writer to save output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Draw bounding boxes and count vehicles
    vehicle_count = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID

            # Filter for vehicle classes (Car, Bus, Truck, Motorcycle)
            vehicle_classes = [2, 3, 5, 7]  # COCO Classes for vehicles
            if cls in vehicle_classes and conf > 0.5:
                vehicle_count += 1
                label = f"Vehicle {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display vehicle count on the frame
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show output
    cv2.imshow("YOLOv8 Vehicle Detection", frame)
    out.write(frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
