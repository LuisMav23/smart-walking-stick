import time
import cv2
import torch
import socketio
from picamera import PiCamera
from picamera.array import PiRGBArray

# Initialize Socket.IO client (synchronous)
sio = socketio.Client()

@sio.event
def connect():
    print("Connected to server")

@sio.event
def disconnect():
    print("Disconnected from server")

# Load YOLOv5 model (custom weights)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')
model.conf = 0.7  # confidence threshold

# Camera settings
RESOLUTION = (640, 480)
FRAMERATE = 15

def main():
    # Connect to server
    try:
        sio.connect('http://localhost:3000')
    except Exception as e:
        print(f"Failed to connect: {e}")

    # Initialize PiCamera
    camera = PiCamera()
    camera.resolution = RESOLUTION
    camera.framerate = FRAMERATE
    raw_capture = PiRGBArray(camera, size=RESOLUTION)
    time.sleep(0.1)  # allow camera to warm up

    print("Starting video capture...")
    try:
        for frame_obj in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
            frame = frame_obj.array

            # YOLOv5 inference
            results = model(frame)
            boxes = results.xyxy[0]
            annotated = frame.copy()

            if boxes.shape[0] > 0:
                # pick largest box
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                idx = torch.argmax(areas)
                x1, y1, x2, y2, conf, cls = boxes[idx].tolist()
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                label = model.names[int(cls)]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Encode and send
            ret, buf = cv2.imencode('.jpg', annotated)
            if ret and sio.connected:
                sio.emit('frame', buf.tobytes())

            # Display locally
            cv2.imshow('YOLOv5 on PiCamera', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # clear buffer and sleep to maintain FPS
            raw_capture.truncate(0)
            time.sleep(1 / FRAMERATE)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        camera.close()
        cv2.destroyAllWindows()
        if sio.connected:
            sio.disconnect()
        print("Clean shutdown complete")

if _name_ == '_main_':
    main()