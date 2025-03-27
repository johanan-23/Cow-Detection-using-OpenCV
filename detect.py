import cv2
import numpy as np

class CowDetection:
    def __init__(self, model_pb, config_pbtxt, coco_label_txt):
        # Load the trained model
        self.net = cv2.dnn.readNetFromTensorflow(model_pb, config_pbtxt)
        self.labels = self._read_label(coco_label_txt)
        
    def predict(self, image, min_confidence=0.3, max_iou=0.3):
        # Preprocess the image
        blob = cv2.dnn.blobFromImage(image, size=(300, 300) , swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Extract results: image_id, label, confidence, x1, y1, x2, y2
        result = np.array([det[1:] for det in detections[0, 0] if det[2] > min_confidence])

        # Validate the result to ensure it is not empty and has the expected structure
        if result is None or len(result) == 0:
            return [], [], []  # Return empty lists if no detections are found

        if len(result.shape) == 1:  # If result is 1D, reshape it to 2D
            result = result.reshape(-1, result.shape[0])

        label_ids = result[:, 0].astype(int)
        scores = result[:, 1]
        boxes = np.clip(result[:, 2:], 0, 1)

        # Convert to pixel values (scaled to the image size)
        height, width = image.shape[:2]
        boxes = boxes * [width, height, width, height]

        # Perform Non-Maximum Suppression (NMS)
        boxes = boxes.astype(np.int32)
        confidences = scores.tolist()
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences, min_confidence, max_iou)

        if len(indices) > 0:
            indices = indices.flatten()
            boxes = boxes[indices]
            labels = self.labels[label_ids[indices]]
            scores = scores[indices]
        else:
            boxes = np.array([])
            labels = np.array([])
            scores = np.array([])

        return boxes, labels, scores
    
    def draw(self, frame, bbox, labels, scores):
        # Draw bounding boxes and labels on the frame
        for (x1, y1, x2, y2), label, conf in zip(bbox, labels, scores):
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # Put label and confidence score
            label_text = f"{label} {conf:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return frame
    
    @staticmethod
    def _read_label(label_txt_file):
        with open(label_txt_file, "r") as f:
            labels = [line.strip("\n") for line in f]
        return np.array(labels)
