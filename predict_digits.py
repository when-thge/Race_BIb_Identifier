import cv2 as cv
import os
import sys
from ultralytics import YOLO

def run_live_cam(model_path):
    # Load the specialized FunRun model
    model = YOLO(model_path)
    
    # Initialize camera
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("Live Camera Active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the current frame
        results = model(frame, verbose=False)[0]
        
        # Extract and sort digit detections
        digit_detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            digit_val = int(box.cls[0].item())
            digit_detections.append({'x': x1, 'val': str(digit_val)})
        
        # Sort digits from Left to Right
        digit_detections.sort(key=lambda d: d['x'])
        bib_number = "".join([d['val'] for d in digit_detections])
        
        # Draw on frame
        annotated_frame = results.plot()
        if bib_number:
            cv.putText(annotated_frame, f"Bib: {bib_number}", (50, 50), 
                        cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Show the frame
        cv.imshow("FunRun Live Detection", annotated_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def predict_digits(input_path, model_path='runs/detect/funrun_digit_model/weights/best.pt'):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    # Handle Camera Mode
    if input_path == "--cam":
        run_live_cam(model_path)
        return

    model = YOLO(model_path)
    
    # Handle single file or directory
    images = []
    if os.path.isdir(input_path):
        for f in os.listdir(input_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(input_path, f))
    else:
        images.append(input_path)

    print(f"Processing {len(images)} images...")

    for img_path in images:
        results = model(img_path, verbose=False)[0]
        digit_detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            digit_val = int(box.cls[0].item())
            digit_detections.append({'x': x1, 'val': str(digit_val)})
        
        digit_detections.sort(key=lambda d: d['x'])
        bib_number = "".join([d['val'] for d in digit_detections]) if digit_detections else "???"
            
        print(f"Image: {os.path.basename(img_path)} | Detected Bib: {bib_number}")
        res_img = results.plot()
        cv.imwrite(f"digit_result_{os.path.basename(img_path)}", res_img)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Images/Folder: python3 predict_digits.py <image_path_or_dir>")
        print("  Live Camera:   python3 predict_digits.py --cam")
    else:
        target = sys.argv[1]
        predict_digits(target)
