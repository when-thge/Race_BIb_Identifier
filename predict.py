from ultralytics import YOLO
import cv2
import numpy as np
import os
import argparse

class RaceBibIdentifier:
    def __init__(self, bib_model_path='yolo26_bibs.pt', digit_model_path='yolo26_digits.pt'):
        print(f"Loading Bib Detector: {bib_model_path}")
        self.bib_model = YOLO(bib_model_path)
        print(f"Loading Digit Detector: {digit_model_path}")
        self.digit_model = YOLO(digit_model_path)
        # Mapping: class index i -> digit "i"
        self.digit_map = {i: str(i) for i in range(10)}

    def identify(self, image_path, output_path=None):
        img = cv2.imread(image_path)
        if img is None: return None
        h, w = img.shape[:2]
        
        # 1. Detect Bibs
        bib_results = self.bib_model(img, verbose=False)[0]
        
        detections = []
        for box in bib_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 2. Crop Bib
            margin = 5
            cx1, cy1 = max(0, x1-margin), max(0, y1-margin)
            cx2, cy2 = min(w, x2+margin), min(h, y2+margin)
            bib_crop = img[cy1:cy2, cx1:cx2]
            
            # 3. Detect Digits in Crop
            digit_results = self.digit_model(bib_crop, verbose=False)[0]
            
            found_digits = []
            for d_box in digit_results.boxes:
                dx1, dy1, dx2, dy2 = map(int, d_box.xyxy[0])
                cls = int(d_box.cls[0])
                found_digits.append({'x': dx1, 'val': self.digit_map.get(cls, "?")})
            
            found_digits.sort(key=lambda x: x['x'])
            number = "".join([d['val'] for d in found_digits])
            
            detections.append({'box': (x1, y1, x2, y2), 'number': number})
            
            # Draw
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Bib: {number}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        if output_path:
            cv2.imwrite(output_path, img)
        return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.jpg')
    args = parser.parse_args()
    
    # Use best available or fallback to base
    b_path = 'yolo26_bibs.pt' if os.path.exists('yolo26_bibs.pt') else 'yolo26n.pt'
    d_path = 'yolo26_digits.pt' if os.path.exists('yolo26_digits.pt') else 'yolo26n.pt'
    
    identifier = RaceBibIdentifier(bib_model_path=b_path, digit_model_path=d_path)
    res = identifier.identify(args.image, args.output)
    
    if res:
        for i, d in enumerate(res):
            print(f"Bib {i+1}: Number={d['number']}")
    else:
        print("No bibs detected.")
