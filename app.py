import sys
import pathlib

# Patch PosixPath for Windows compatibility
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import torch
import asyncio
import socketio
import threading
from playsound import playsound

sio = socketio.AsyncClient()

@sio.event
async def connect():
    print("Connected to server")

@sio.event
async def disconnect():
    print("Disconnected")

# Load model (replace with your actual path)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')

# Set confidence threshold
model.conf = 0.8

async def main():
    cap = cv2.VideoCapture(0)
    last_detected_class = None

    # target frame delay for 15 fps (i.e., 1/15 seconds per frame)
    frame_delay = 1 / 30
    i = 360

    while True:
        
        if not sio.connected and i % 360 == 0:
            try:
                await sio.connect('http://localhost:3000')
            except Exception as e:
                print("Reconnection attempt failed:", e)
            finally:
                i = 0
        i += 1

        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model(frame)

        # Process detections to select the closest object (assumed as the largest bounding box)
        boxes = results.xyxy[0]  # Each row: [x1, y1, x2, y2, conf, cls]
        annotated_frame = frame.copy()
        if boxes.shape[0] > 0:
            # Compute areas of all bounding boxes
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            max_idx = torch.argmax(areas)
            box = boxes[max_idx]
            x1, y1, x2, y2, conf, cls = box.tolist()
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            class_label = model.names[int(cls)]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, class_label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Update the last detected class (assume last_detected_class is defined outside the loop)
            last_detected_class = class_label

            # On every 30th frame, play the audio for the last detected class and then clear it
            if last_detected_class:
                audio_file = f"voices/{last_detected_class.lower()}_detected.mp3"
                threading.Thread(target=playsound, args=(audio_file,), daemon=True).start()
                last_detected_class = None

        # Encode and send the annotated frame if connected
        ret, buf = cv2.imencode('.jpg', annotated_frame)
        jpg_bytes = buf.tobytes()
        if sio.connected:
            await sio.emit('frame', jpg_bytes)

        # Wait to maintain 15 fps
        await asyncio.sleep(frame_delay)

        # Display the processed frame
        cv2.imshow('YOLOv5 Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        asyncio.run(sio.disconnect())
