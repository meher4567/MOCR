import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import cv2
from paddleocr import PaddleOCR
from googletrans import Translator


def preprocess_image(image_path):
    """
    Preprocess the image for better OCR accuracy.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {image_path}")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    binary_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return binary_image


def detect_text_boxes(image_path, rotate=False):
    """
    Detect text regions and extract text using PaddleOCR.
    Rotate the image if specified, then revert after processing.
    """
    ocr = PaddleOCR(use_angle_cls=True, lang='japan', use_gpu=True)

    if rotate:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image could not be read: {image_path}")
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_path = image_path.replace(".jpg", "_rotated.jpg").replace(".png", "_rotated.png")
        cv2.imwrite(rotated_path, rotated_image)
        image_path = rotated_path

    results = ocr.ocr(image_path, cls=True)

    detected_boxes = []
    for line in results[0]:
        bbox, text, confidence = line[0], line[1][0], line[1][1]
        if confidence > 0.7:
            detected_boxes.append({
                'bbox': [int(bbox[0][0]), int(bbox[0][1]), int(bbox[2][0]), int(bbox[2][1])],
                'text': text,
                'confidence': confidence,
            })

    return detected_boxes, image_path


def translate_text(text, source_lang="ja", target_lang="en"):
    """
    Translate text using Google Translate API via googletrans library.
    """
    translator = Translator()
    return translator.translate(text, src=source_lang, dest=target_lang).text


def remove_text_from_image(image_path, detected_boxes):
    """
    Remove text from the image by filling detected text areas with a solid white background.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {image_path}")

    for box in detected_boxes:
        x_min, y_min, x_max, y_max = box['bbox']

        # Expand box slightly to cover edges
        pad = 3
        x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
        x_max, y_max = min(image.shape[1], x_max + pad), min(image.shape[0], y_max + pad)

        # Fill the detected area with white
        image[y_min:y_max, x_min:x_max] = (255, 255, 255)

    cleaned_image_path = image_path.replace(".jpg", "_cleaned.jpg").replace(".png", "_cleaned.png")
    cv2.imwrite(cleaned_image_path, image)

    return cleaned_image_path


def map_rotated_to_vertical(height, width, rotated_boxes):
    """
    Map rotated bounding box coordinates to their vertical equivalents.
    """
    vertical_boxes = []
    for box in rotated_boxes:
        x_min, y_min = box['bbox'][0], box['bbox'][1]
        x_max, y_max = box['bbox'][2], box['bbox'][3]

        vertical_bbox = [
            y_min,
            width - x_max,
            y_max,
            width - x_min
        ]

        vertical_boxes.append({
            'bbox': vertical_bbox,
            'text': box['text'],
            'confidence': box['confidence']
        })

    return vertical_boxes


def calculate_middle_box(rotated_box, vertical_box):
    """
    Calculate a bounding box that represents the middle position between the rotated
    and vertical bounding boxes.
    """
    x_min_rot, y_min_rot, x_max_rot, y_max_rot = rotated_box
    x_min_vert, y_min_vert, x_max_vert, y_max_vert = vertical_box

    x_min = (x_min_rot + x_min_vert) // 2
    y_min = (y_min_rot + y_min_vert) // 2
    x_max = (x_max_rot + x_max_vert) // 2
    y_max = (y_max_rot + y_max_vert) // 2

    return [x_min, y_min, x_max, y_max]


def fill_translations_on_vertical_image(vertical_cleaned_image_path, rotated_boxes, vertical_boxes):
    """
    Fill translated text in the middle position between rotated and vertical boxes
    on the cleaned vertical image.
    """
    image = cv2.imread(vertical_cleaned_image_path)
    if image is None:
        raise FileNotFoundError(f"Image could not be read: {vertical_cleaned_image_path}")

    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    for rotated_box, vertical_box in zip(rotated_boxes, vertical_boxes):
        middle_box = calculate_middle_box(rotated_box['bbox'], vertical_box['bbox'])
        x_min, y_min, x_max, y_max = middle_box
        translated_text = translate_text(rotated_box['text'])

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (0, 0, 0)

        box_width = x_max - x_min
        box_height = y_max - y_min

        words = translated_text.split()
        wrapped_text = []
        current_line = ""
        for word in words:
            line_width, _ = cv2.getTextSize(current_line + " " + word, font, font_scale, font_thickness)[0]
            if line_width < box_width:
                current_line += " " + word
            else:
                wrapped_text.append(current_line.strip())
                current_line = word
        wrapped_text.append(current_line.strip())

        line_height = int(cv2.getTextSize("Test", font, font_scale, font_thickness)[0][1] * 1.5)
        y_offset = y_min + line_height
        for line in wrapped_text:
            if y_offset + line_height > y_max:
                break
            cv2.putText(image, line, (x_min + 5, y_offset), font, font_scale, text_color, font_thickness)
            y_offset += line_height + 5

    translated_image_path = vertical_cleaned_image_path.replace(".jpg", "_translated.jpg").replace(".png", "_translated.png")
    cv2.imwrite(translated_image_path, image)

    return translated_image_path

@csrf_exempt
def upload_images(request):
    """
    Django view to process uploaded images for OCR, translation, and proper orientation.
    """
    if request.method == 'POST' and request.FILES:
        uploaded_files = request.FILES.getlist('images')
        extraction_results = []

        for image_file in uploaded_files:
            file_path = default_storage.save(f'uploads/{image_file.name}', image_file)
            absolute_path = default_storage.path(file_path)

            try:
                rotated_boxes, rotated_path = detect_text_boxes(absolute_path, rotate=True)

                height, width = cv2.imread(absolute_path).shape[:2]
                vertical_boxes = map_rotated_to_vertical(height, width, rotated_boxes)

                cleaned_image_path = remove_text_from_image(rotated_path, rotated_boxes)

                translated_image_path = fill_translations_on_vertical_image(
                    cleaned_image_path, rotated_boxes, vertical_boxes
                )

                extraction_results.append({
                    'original_image_path': file_path,
                    'translated_image_path': translated_image_path.replace('\\', '/'),
                    'detected_boxes': rotated_boxes,
                })

            except Exception as e:
                extraction_results.append({
                    'original_image_path': file_path,
                    'error': str(e),
                })

        return JsonResponse({'message': 'Processing completed', 'results': extraction_results}, status=200)

    return JsonResponse({'error': 'Invalid request'}, status=400)
