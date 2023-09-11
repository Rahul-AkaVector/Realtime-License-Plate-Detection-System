from ultralytics import YOLO
import cv2
import time

from deeplearning import number_plate_to_text

# Load YOLO model
model = YOLO('yolov8n.pt')

# Open webcam capture
cap = cv2.VideoCapture(0)  # 0 represents the default webcam

# Define class indices for cars and trucks
# l1 = 2.0
# l2 = 'board'  # Replace with the correct class index for trucks
valid_vehicles = [2.0, 3.0, 5.0, 7.0]
while cap.isOpened():
    ret, frame = cap.read()

    # print(frame)

    if not ret:
        break

    # Perform object detection
    results = model.predict(source=frame, show=False)
    result = results[0]
    box = result.boxes.data
    box_data = [0,0,0,0,0,0]
    # print(box)
    if len(box) >= 1:
        # obj_class_loc = 0
        for i in range(len(box)):
            # print(box[i].tolist()[5])
            # print(valid_vehicles)
            if box[i].tolist()[5] in valid_vehicles:
                # print(valid_vehicles)
                box_data = box[i].tolist()
                break

    xmin, ymin, xmax, ymax, score, vehicle = box_data

    nframe = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

    if vehicle in valid_vehicles:
        text = number_plate_to_text(nframe)
        if text is not None:
            if text[0:2].isalpha() and text[2:4].isdigit() and text[4:6].isalpha() and text[6:10].isdigit():
                print(text)
                cv2.imshow('Detected Image', nframe)
                cv2.waitKey(10000)
                cv2.destroyWindow('Detected Image')

    # Display the original frame
    cv2.imshow('Webcam Feed', frame)
    # cv2.waitKey(5000)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

