import cv2
from detect import CowDetection
import config as cfg

if __name__ == "__main__":
    # Initialize video capture (use camera if cfg.USE_CAMERA is True, else use video file)
    if cfg.USE_CAMERA:
        media = cv2.VideoCapture(0)  # 0 is the default camera index
    else:
        media = cv2.VideoCapture(cfg.FILE_PATH)

    if not media.isOpened():
        print("Error: Could not open video source.")
        exit()

    # Initialize the CowDetection model
    model = CowDetection(cfg.MODEL_PATH, cfg.CONFIG_PATH, cfg.LABEL_PATH)

    # Get the video frame width and height to calculate aspect ratio
    frame_width = int(media.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(media.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = frame_width / frame_height

    # Create OpenCV window with a specific size
    cv2.namedWindow(cfg.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.WINDOW_NAME, 500, int(500 / aspect_ratio))

    while True:
        # Read a frame from the video or camera
        ret, frame = media.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Get predictions and bounding boxes
        bbox, labels, scores = model.predict(frame, min_confidence=cfg.MIN_CONF, max_iou=cfg.MAX_IOU)

        # Draw the bounding boxes and labels on the frame
        frame = model.draw(frame, bbox, labels, scores)

        # Show the frame in the window
        cv2.imshow(cfg.WINDOW_NAME, frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close the window
    media.release()
    cv2.destroyAllWindows()
