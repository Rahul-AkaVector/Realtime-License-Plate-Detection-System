import tkinter as tk
import cv2
from ultralytics import YOLO
from deeplearning import number_plate_to_text
from PIL import Image, ImageTk
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class CarDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Detection App")

        self.cap = None
        self.is_running = False
        self.model = YOLO('yolov8n.pt')
        self.valid_vehicles = [2.0, 3.0, 5.0, 7.0]
        self.current_camera = 0  # Default to main camera (0)

        # Initialize Firebase
        cred = credentials.Certificate(r"private-keys.json")
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()

        self.create_widgets()
        self.configure_responsive_layout()

    def create_widgets(self):
        self.root.configure(bg='black')

        self.video_frame = tk.Frame(self.root, width=640, height=480, bg='black')
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.canvas = tk.Canvas(self.video_frame, width=640, height=480, bg='black')
        self.canvas.pack(fill="both", expand=True)

        self.start_button = tk.Button(self.root, text="Start", command=self.start_detection, width=20, height=2)
        self.start_button.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

    def configure_responsive_layout(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

    def start_detection(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.current_camera)

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)

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
                if text is not None and text[0:2].isalpha() and text[2:4].isdigit() and text[4:6].isalpha() and text[6:10].isdigit():
                    self.show_detected_image(nframe, text)
                    self.check_and_notify(text)

            self.show_camera_feed(frame)
            self.root.update()

        self.cap.release()
        self.start_button.config(state=tk.NORMAL)

    def show_camera_feed(self, frame):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=frame)
            self.canvas.image = frame

    def show_detected_image(self, image, text):
        detected_window = tk.Toplevel(self.root)
        detected_window.title("Detected Plate")
        detected_window.configure(bg='black')

        detected_canvas = tk.Canvas(detected_window, width=320, height=240, bg='black')
        detected_canvas.pack(fill="both", expand=True)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        detected_canvas.create_image(0, 0, anchor=tk.NW, image=image)
        detected_canvas.image = image

        text_label = tk.Label(detected_canvas, text=f"Detected Plate: {text}", bg='black', fg='white', font=("Helvetica", 16))
        text_label.place(x=10, y=10)

    def check_and_notify(self, number_plate):
        reports_ref = self.db.collection('reports').where('number', '==', number_plate).stream()

        email = "rakshanetra@gmail.com"  # Your email address
        subject = "URGENT: Lost Car Spotted"

        # Static message
        message_text = f"Attention!\n\nThis is to inform you that your lost car with number plate: {number_plate} has been spotted at Khandeshwar CCTV Camera. Please contact the nearest police station at your earliest convenience.\n\nContact details:\nPolice Station: Khandeshwar\nPhone: 123-456-7890"

        msg = MIMEMultipart()
        msg['From'] = email
        msg['Subject'] = subject

        for report in reports_ref:
            data = report.to_dict()
            receiver_email = data['email']
            msg['To'] = receiver_email
            msg.attach(MIMEText(message_text, 'plain'))

            formatted_message = msg.as_string()

            # Send the email
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(email, "xomk hocv palx xaqr")  # Your email password
            server.sendmail(email, receiver_email, formatted_message)
            server.quit()

            print("Email has been sent to " + receiver_email)


if __name__ == "__main__":
    root = tk.Tk()
    app = CarDetectionApp(root)
    root.mainloop()
