from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov5s.pt")  # or 'yolov5su.pt' as suggested

# Run prediction on image
results = model("scratch3.png")  # Just pass the path string

# Display results
for r in results:
    r.show()  # Shows image with detections
    r.save(filename="defect_detection_output.jpg")  # Save output image

    # Count detected objects
    counts = {}
    names = model.names  # class names

    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            counts[label] = counts.get(label, 0) + 1

    # Print counts
    for label, count in counts.items():
        print(f"{label}: {count}")
