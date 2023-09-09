# from ultralytics import YOLO
# # model = YOLO('yolov8n.pt')
#
#
# model = YOLO('plate.pt')
# results = model.predict(source='1', show=True)

# ------------------------------------------------

# from ultralytics import YOLO
# import cv2
# import time
#
# # Load YOLO model
# model = YOLO('plate.pt')
#
# # Open webcam capture
# cap = cv2.VideoCapture(1)  # 0 represents the default webcam
#
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     # Perform object detection
#     results = model.predict(source=frame, show=False)
#     detected_objects = results[0] if results[0] is not None else []
#
#     if len(detected_objects) > 0:
#
#         # Convert the frame to grayscale
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # Save the grayscale frame
#         cv2.imwrite('detected_image.jpg', gray_frame)
#
#         # Convert the frame back to BGR for displaying
#         frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
#
#         # Display the frame with detected objects
#         cv2.imshow('Detected Image', frame)
#         time.sleep(10)
#         # cv2.waitKey(0) # Wait for a key press
#
#     # Wait for 10 seconds before resuming
#
#     # Display the original frame
#     cv2.imshow('Webcam Feed', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# --------------------------------------------------

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
    # detected_objects = results.pred[0] if results.pred[0] is not None else []
    # print(results)
    # detected_objects = results[0]
    # print(detected_objects)

    # Filter detected objects for cars and trucks
    # car_truck_objects = [obj for obj in detected_objects if obj[5] == car_class_index or obj[5] == truck_class_index]
    # car_truck_objects = [obj for obj in detected_objects if obj[0] in (car_class_label, truck_class_label)]
    # car_truck_objects = [obj for obj in detected_objects if car_class_label in obj[1] or truck_class_label in obj[1]]
    # print(detected_objects.names)
    # car_truck_objects = [obj for obj in detected_objects if car_class_label in obj or truck_class_label in obj]
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
    # print("\nxmin", xmin, "\nymin", ymin, "\nxmax", xmax, "\nymax", ymax, "\nscore", score, "\nvehicle", vehicle)
    # print("BoxData : ",box_data[0:4])

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
