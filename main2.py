# import tkinter as tk
# import cv2
# # import time
# from ultralytics import YOLO
# from deeplearning import number_plate_to_text
# from PIL import Image, ImageTk
#
# class CarDetectionApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Car Detection App")
#
#         self.cap = None
#         self.is_running = False
#         self.model = YOLO('yolov8n.pt')
#         self.valid_vehicles = [2.0, 3.0, 5.0, 7.0]
#         self.current_camera = 0  # Default to main camera (0)
#
#         self.create_widgets()
#
#     def create_widgets(self):
#         # Dark mode background
#         self.root.configure(bg='black')
#
#         # Frames to hold the video feed and detected image
#         self.video_frame = tk.Frame(self.root, width=640, height=480, bg='black')
#         self.video_frame.grid(row=0, column=0, padx=10, pady=10)
#         self.detected_frame = tk.Frame(self.root, width=640, height=480, bg='black')
#         self.detected_frame.grid(row=0, column=1, padx=10, pady=10)
#
#         # Camera label in the middle
#         self.camera_label = tk.Label(self.root, text="Camera: 0", bg='black', fg='white')
#         self.camera_label.grid(row=1, column=0, columnspan=2, pady=10)
#
#         # Buttons split into three columns at the bottom
#         self.button_frame = tk.Frame(self.root, bg='black')
#         self.button_frame.grid(row=2, column=0, columnspan=2, pady=10)
#
#         self.camera_button = tk.Button(self.button_frame, text="Change Camera Input", command=self.toggle_camera, width=20, height=2)
#         self.camera_button.grid(row=0, column=0, padx=10, pady=5)
#
#         self.start_button = tk.Button(self.button_frame, text="Start", command=self.start_detection, width=20, height=2)
#         self.start_button.grid(row=0, column=1, padx=10, pady=5)
#
#         self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_detection, state=tk.DISABLED, width=20, height=2)
#         self.stop_button.grid(row=0, column=2, padx=10, pady=5)
#
#         self.canvas = tk.Canvas(self.video_frame, width=640, height=480, bg='black')
#         self.canvas.pack()
#
#         self.detected_canvas = tk.Canvas(self.detected_frame, width=640, height=480, bg='black')
#         self.detected_canvas.pack()
#
#     def toggle_camera(self):
#         if self.is_running:
#             return
#
#         if self.cap is not None:
#             self.cap.release()
#             self.cap = None
#
#         # Toggle to the next camera input
#         self.current_camera = (self.current_camera + 1) % 2  # Assuming two cameras (0 and 1)
#         self.camera_label.config(text=f"Camera: {self.current_camera}")
#
#     def start_detection(self):
#         if self.cap is None:
#             self.cap = cv2.VideoCapture(self.current_camera)
#
#         self.is_running = True
#         self.start_button.config(state=tk.DISABLED)
#         self.stop_button.config(state=tk.NORMAL)
#         self.camera_button.config(state=tk.DISABLED)
#
#         while self.is_running:
#             ret, frame = self.cap.read()
#
#             if not ret:
#                 break
#
#             results = self.model.predict(source=frame, show=False)
#             result = results[0]
#             box = result.boxes.data
#             box_data = [0, 0, 0, 0, 0, 0]
#
#             if len(box) >= 1:
#                 for i in range(len(box)):
#                     if box[i].tolist()[5] in self.valid_vehicles:
#                         box_data = box[i].tolist()
#                         break
#
#             xmin, ymin, xmax, ymax, score, vehicle = box_data
#
#             nframe = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
#
#             if vehicle in self.valid_vehicles:
#                 text = number_plate_to_text(nframe)
#                 if text is not None and text[0:2].isalpha() and text[2:4].isdigit() and text[4:6].isalpha() and text[6:10].isdigit():
#                     self.show_detected_image(nframe, text)
#
#             self.show_camera_feed(frame)
#             self.root.update()
#
#         self.stop_button.config(state=tk.DISABLED)
#         self.start_button.config(state=tk.NORMAL)
#         self.camera_button.config(state=tk.NORMAL)
#         self.cap.release()
#
#     def stop_detection(self):
#         self.is_running = False
#
#     def show_camera_feed(self, frame):
#         if frame is not None:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = Image.fromarray(frame)
#             frame = ImageTk.PhotoImage(frame)
#             self.canvas.create_image(0, 0, anchor=tk.NW, image=frame)
#             self.canvas.image = frame
#
#     def show_detected_image(self, image, text):
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = Image.fromarray(image)
#         image = ImageTk.PhotoImage(image)
#         self.detected_canvas.create_image(0, 0, anchor=tk.NW, image=image)
#         self.detected_canvas.image = image
#
#         text_label = tk.Label(self.detected_canvas, text=f"Detected Plate: {text}", bg='black', fg='white', font=("Helvetica", 16))
#         text_label.place(x=10, y=10)
#
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = CarDetectionApp(root)
#     root.mainloop()

import tkinter as tk
import cv2
# import time
from ultralytics import YOLO
from deeplearning import number_plate_to_text
from PIL import Image, ImageTk


class CarDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Detection App")

        self.cap = None
        self.is_running = False
        self.model = YOLO('yolov8n.pt')
        self.valid_vehicles = [2.0, 3.0, 5.0, 7.0]
        self.current_camera = 0  # Default to main camera (0)

        self.create_widgets()
        self.configure_responsive_layout()

    def create_widgets(self):
        # Dark mode background
        self.root.configure(bg='black')

        # Frames to hold the video feed and detected image
        self.video_frame = tk.Frame(self.root, width=640, height=480, bg='black')
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.detected_frame = tk.Frame(self.root, width=640, height=480, bg='black')
        self.detected_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Camera label in the middle
        self.camera_label = tk.Label(self.root, text="Camera: 0", bg='black', fg='white')
        self.camera_label.grid(row=1, column=0, columnspan=2, pady=10, sticky="nsew")

        # Buttons frame with scrollbar
        self.button_frame = tk.Frame(self.root, bg='black')
        self.button_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="nsew")

        self.button_frame.grid_rowconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.video_frame, width=640, height=480, bg='black')
        self.canvas.pack(fill="both", expand=True)

        self.detected_canvas = tk.Canvas(self.detected_frame, width=640, height=480, bg='black')
        self.detected_canvas.pack(fill="both", expand=True)

        self.button_canvas = tk.Canvas(self.button_frame, bg='black')
        self.button_canvas.pack(side="left", fill="y")

        self.button_scrollbar = tk.Scrollbar(self.button_frame, orient="vertical", command=self.button_canvas.yview)
        self.button_scrollbar.pack(side="right", fill="y")

        self.button_canvas.configure(yscrollcommand=self.button_scrollbar.set)
        self.button_canvas.bind('<Configure>',
                                lambda e: self.button_canvas.configure(scrollregion=self.button_canvas.bbox("all")))

        self.button_frame_inner = tk.Frame(self.button_canvas, bg='black')
        self.button_canvas.create_window((0, 0), window=self.button_frame_inner, anchor="nw")
        self.button_frame_inner.bind("<Configure>", self.on_frame_configure)

        self.camera_button = tk.Button(self.button_frame_inner, text="Change Camera Input", command=self.toggle_camera,
                                       width=20, height=2)
        self.camera_button.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        self.start_button = tk.Button(self.button_frame_inner, text="Start", command=self.start_detection, width=20,
                                      height=2)
        self.start_button.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        self.stop_button = tk.Button(self.button_frame_inner, text="Stop", command=self.stop_detection,
                                     state=tk.DISABLED, width=20, height=2)
        self.stop_button.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")

    def configure_responsive_layout(self):
        # Configure rows and columns to expand and fill available space
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.detected_frame.grid_rowconfigure(0, weight=1)
        self.detected_frame.grid_columnconfigure(0, weight=1)

    def on_frame_configure(self, event):
        self.button_canvas.configure(scrollregion=self.button_canvas.bbox("all"))

    def toggle_camera(self):
        if self.is_running:
            return

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # Toggle to the next camera input
        self.current_camera = (self.current_camera + 1) % 2  # Assuming two cameras (0 and 1)
        self.camera_label.config(text=f"Camera: {self.current_camera}")

    def start_detection(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.current_camera)

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.camera_button.config(state=tk.DISABLED)

        while self.is_running:
            ret, frame = self.cap.read()

            if not ret:
                break

            results = self.model.predict(source=frame, show=False)
            result = results[0]
            box = result.boxes.data
            box_data = [0, 0, 0, 0, 0, 0]

            if len(box) >= 1:
                for i in range(len(box)):
                    if box[i].tolist()[5] in self.valid_vehicles:
                        box_data = box[i].tolist()
                        break

            xmin, ymin, xmax, ymax, score, vehicle = box_data

            nframe = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

            if vehicle in self.valid_vehicles:
                text = number_plate_to_text(nframe)
                if text is not None and text[0:2].isalpha() and text[2:4].isdigit() and text[4:6].isalpha() and text[
                                                                                                                6:10].isdigit():
                    self.show_detected_image(nframe, text)

            self.show_camera_feed(frame)
            self.root.update()

        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        self.camera_button.config(state=tk.NORMAL)
        self.cap.release()

    def stop_detection(self):
        self.is_running = False

    def show_camera_feed(self, frame):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=frame)
            self.canvas.image = frame

    def show_detected_image(self, image, text):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.detected_canvas.create_image(0, 0, anchor=tk.NW, image=image)
        self.detected_canvas.image = image

        text_label = tk.Label(self.detected_canvas, text=f"Detected Plate: {text}", bg='black', fg='white',
                              font=("Helvetica", 16))
        text_label.place(x=10, y=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = CarDetectionApp(root)
    root.mainloop()
