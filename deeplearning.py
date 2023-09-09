from ultralytics import YOLO
import easyocr

model = YOLO("plate.pt")

reader = easyocr.Reader(['en'], gpu=False)

def license_plate_detection(path):
    result = model(path)
    result = result[0]
    box = result.boxes.data

    if len(box) == 0:
        return None, None, None, None, None

    box_data = box[0].tolist()
    xmin, ymin, xmax, ymax = box_data[:4]
    image = result.orig_img
    return image, xmin, ymin, xmax, ymax




char_to_int = {
    'O': '0',
    'I': '1',
    'J': '3',
    'A': '4',
    'G': '6',
    'S': '5',
}

int_to_char = {
    '0': 'O',
    '1': 'I',
    '3': 'J',
    '4': 'A',
    '6': 'G',
    '5': 'S',
}


def letter_to_number(text):
    converted = ""
    for char in text:
        if char in char_to_int:
            converted += char_to_int[char]
        else:
            converted += char
    return converted


def number_to_letter(text):
    converted = ""
    for char in text:
        if char in int_to_char:
            converted += int_to_char[char]
        else:
            converted += char
    return converted


def change_plate_number(text):
    if not text[-4:].isdigit():
        x4 = letter_to_number(text[-4:])
    else:
        x4 = text[-4:]

    if not text[-6:-4].isalpha():
        x3 = number_to_letter(text[-6:-4])
    else:
        x3 = text[-6:-4]

    if not text[-8:-6].isdigit():
        x2 = letter_to_number(text[-8:-6])
    else:
        x2 = text[-8:-6]

    if not text[-10:-8].isalpha():
        x1 = number_to_letter(text[-10:-8])
    else:
        x1 = text[-10:-8]
    return x1 + x2 + x3 + x4


def number_plate_to_text(path):
    img, xmin, ymin, xmax, ymax = license_plate_detection(path)

    if img is None or xmin is None or ymin is None or xmax is None or ymax is None:
        return None

    roi = img[int(ymin):int(ymax), int(xmin):int(xmax)]

    detections = reader.readtext(roi)

    for detection in detections:
        bbox, text, score = detection
        # print(text)
        if 10 <= len(text) <= 16:
            text = text.upper().replace(' ', '')
            text = change_plate_number(text)
            return text