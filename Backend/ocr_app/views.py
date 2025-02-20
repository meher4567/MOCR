import os
import torch

torch_folder = os.path.dirname(torch.__file__)
os.add_dll_directory(os.path.join(torch_folder, 'lib'))
print("Using PyTorch DLLs from:", os.path.join(torch_folder, 'lib'))

import cv2
import numpy as np
from paddleocr import PaddleOCR
import os
from PIL import Image
from manga_ocr import MangaOcr
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ocr_app//gt.json"

# Initialize PaddleOCR and MangaOCR
ocr = PaddleOCR(use_angle_cls=True, lang="japan")  # Set language to Japanese
mocr = MangaOcr()

from google.cloud import translate_v2 as translate


translate_client = translate.Client()

def translate_text(text, source_lang="ja", target_lang="en"):
    if not text.strip():
        return "Error: Input text is empty."

    try:
        result = translate_client.translate(text, source_language=source_lang, target_language=target_lang)
        return result["translatedText"]
    
    except Exception as e:
        return f"Translation error: {str(e)}"

def detect_text_boxes(image_path):
    """Detect text boxes using PaddleOCR and return bounding box coordinates."""
    results = ocr.ocr(image_path, cls=True)
    text_boxes = []

    for result in results:
        for line in result:
            if line:
                points = np.array(line[0], dtype=np.int32)
                x, y, w, h = cv2.boundingRect(points)
                text_boxes.append((x, y, w, h))

    return text_boxes

def boxes_intersect(box1, box2, padding=5):
    """Check if two boxes intersect or are close to each other."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1 -= padding
    y1 -= padding
    x2 -= padding
    y2 -= padding
    w1 += 2 * padding
    h1 += 2 * padding
    w2 += 2 * padding
    h2 += 2 * padding

    return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

def merge_close_boxes(boxes):
    """Merge only small intersecting boxes while keeping distinct groups separate."""
    if not boxes:
        return []

    merged_boxes = []
    used = set()

    for i, box1 in enumerate(boxes):
        if i in used:
            continue
        x_min, y_min, w_max, h_max = box1

        for j, box2 in enumerate(boxes):
            if i != j and j not in used and boxes_intersect(box1, box2):
                x2, y2, w2, h2 = box2
                x_min = min(x_min, x2)
                y_min = min(y_min, y2)
                w_max = max(x_min + w_max, x2 + w2) - x_min
                h_max = max(y_min + h_max, y2 + h2) - y_min
                used.add(j)

        used.add(i)
        merged_boxes.append((x_min, y_min, w_max, h_max))

    return merged_boxes

def draw_detected_boxes(image, text_boxes):
    """Draw detected and merged bounding boxes on the image."""
    for (x, y, w, h) in text_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image



def estimate_avg_character_height(text_boxes):
    """Estimate the average character height from bounding boxes."""
    if not text_boxes:
        return 10  # Default small text size

    total_height = sum(h for _, _, _, h in text_boxes)
    avg_height = total_height / len(text_boxes)

    return max(10, avg_height)  # Ensure a minimum size

def expand_box(x, y, w, h, image, text_boxes):
    """ 
    Expands the bounding box dynamically based on estimated text size.
    """
    height, width, _ = image.shape

    # Estimate average character height
    avg_char_height = estimate_avg_character_height(text_boxes)

    # Dynamically adjust expansion step & max expansion
    step = max(5, int(avg_char_height * 0.5))
    max_expansion = max(30, int(avg_char_height * 2))

    for expansion in range(0, max_expansion, step):
        # Expand the box
        new_x = max(0, x - expansion)
        new_y = max(0, y - expansion)
        new_w = min(width - new_x, w + 2 * expansion)
        new_h = min(height - new_y, h + 2 * expansion)

        # Convert inner lists to tuples
        text_boxes = [tuple(box) for box in text_boxes]  

        # Ensure we're not comparing the box to itself
        for other_box in text_boxes:
            if other_box == (x, y, w, h):  # Ignore the original box
                continue  
            
            if boxes_intersect((new_x, new_y, new_w, new_h), other_box):
                return x, y, w, h  # Stop expanding if intersection occurs

        # Extract the expanded region
        bubble_region = image[new_y:new_y + new_h, new_x:new_x + new_w]
        if bubble_region is None or bubble_region.size == 0:
            break  # Stop if the region is invalid

        # Convert OpenCV image (BGR) to PIL (RGB)
        bubble_pil = Image.fromarray(cv2.cvtColor(bubble_region, cv2.COLOR_BGR2RGB))

        # Try OCR
        text = mocr(bubble_pil)

        # If text is found, keep expanding
        if text.strip():
            x, y, w, h = new_x, new_y, new_w, new_h
        else:
            break  # Stop expanding if no text is found

    return x, y, w, h


def extract_text_with_manga_ocr(image, text_boxes):
    """Extract text from detected speech bubbles using MangaOCR and visualize final bounding boxes."""
    extracted_texts = []
    new_text_boxes = []  # Store updated (expanded) text boxes

    for (x, y, w, h) in text_boxes:
        # Expand the bounding box before passing to MangaOCR
        x, y, w, h = expand_box(x, y, w, h, image, text_boxes)
        new_text_boxes.append((x, y, w, h))  # Store the new expanded box

        # Extract the final expanded region
        bubble_region = image[y:y+h, x:x+w]
        if bubble_region is None or bubble_region.size == 0:
            continue  # Skip invalid regions

        # Convert OpenCV image (BGR) to RGB for PIL
        bubble_pil = Image.fromarray(cv2.cvtColor(bubble_region, cv2.COLOR_BGR2RGB))

        # Run OCR
        text = mocr(bubble_pil)
        extracted_texts.append({"box": (x, y, w, h), "text": text})

        # Draw the final expanded bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for final boxes

    return image, new_text_boxes  # Return OCR results, modified image, and updated boxes


def post_process_and_visualize_and_extract(image, text_boxes, merge_threshold=20):
    """
    Merge all nearby boxes for final visualization and extract text from merged areas.
    Return both a visualized image and a text-cleaned image with no text.

    - `merge_threshold`: Maximum allowed distance between boxes for merging.
    """
    if not text_boxes:
        return image, {}, image, []

    # Create a mask to merge all nearby boxes
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for (x, y, w, h) in text_boxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # Dilate the mask to merge nearby boxes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_threshold, merge_threshold))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours for the merged regions
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copy images for visualization and cleaning
    final_image = image.copy()
    text_cleaned_image = image.copy()
    extracted_texts = {}
    final_merged_text_boxes = []

    for i, contour in enumerate(contours):
        # Compute bounding box around contour
        x, y, w, h = cv2.boundingRect(contour)

        # Compute the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2

        # Append the merged box (with center and size) for use in overlays
        final_merged_text_boxes.append(((x, y, w, h), (center_x, center_y)))

        # Crop the merged region
        roi = image[y:y + h, x:x + w]

        # Convert the OpenCV image (roi) to PIL Image for MangaOCR
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # Extract text using MangaOCR
        text = mocr(roi_pil)
        extracted_texts[f"Box {i + 1}"] = text

        # Draw boundary for visualization
        cv2.polylines(final_image, [contour], isClosed=True, color=(0, 255, 0), thickness=2)

        # Remove text using inpainting (text-cleaning)
        cv2.drawContours(text_cleaned_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        text_cleaned_image = cv2.inpaint(text_cleaned_image, mask, 7, cv2.INPAINT_TELEA)

    return final_image, extracted_texts, text_cleaned_image, final_merged_text_boxes


import textwrap

def overlay_translated_text(image, text_boxes, translated_texts):
    """
    Overlay translated text in the respective boxes on the image.
    Prioritize horizontal space and stay within the bounding box.
    """
    final_image_with_text = image.copy()

    for (box, center), text in zip(text_boxes, translated_texts.values()):
        x, y, w, h = box
        center_x, center_y = center

        # Adjust font size to fit within the box
        max_font_scale = max(0.4, min(w / 250, h / 60))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 1 if max_font_scale < 0.6 else 2

        # Estimate max characters per line based on box width
        max_chars_per_line = max(5, int(w / 12))
        
        # Word wrapping without overflow
        wrapped_text = textwrap.fill(text, width=max_chars_per_line)
        text_lines = wrapped_text.split("\n")

        # Compute text height for multi-line placement
        text_height = cv2.getTextSize("Test", font, max_font_scale, font_thickness)[0][1]

        # Calculate starting position to vertically center the text block
        total_text_height = len(text_lines) * text_height + (len(text_lines) - 1) * 5
        start_y = max(y + h // 2 - total_text_height // 2, y + text_height)

        # Draw each line inside the bounding box
        for i, line in enumerate(text_lines):
            text_size = cv2.getTextSize(line, font, max_font_scale, font_thickness)[0]
            text_x = max(x + (w - text_size[0]) // 2, x)  # Horizontally centered
            text_y = start_y + i * (text_height + 5)

            # Ensure text doesn't go outside the box
            if text_y + text_size[1] > y + h:
                break

            # Draw background rectangle for text visibility
            cv2.rectangle(final_image_with_text, 
                           (text_x - 3, text_y - text_size[1] - 3), 
                           (text_x + text_size[0] + 3, text_y + 3), 
                           (255, 255, 255), 
                           -1)

            # Overlay text on the image
            cv2.putText(final_image_with_text, line, (text_x, text_y),
                         font, max_font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    return final_image_with_text


def translate_all_texts(extracted_texts):
    translated_texts = {}
    
    # Iterate over the dictionary and translate each value (text)
    for key, text in extracted_texts.items():
        translated_texts[key] = translate_text(text)  # Store the translated text with the same key
    
    return translated_texts


@csrf_exempt
def upload_images(request):
    """Django view to process images, detect speech bubbles, and return results."""
    if request.method == 'POST' and request.FILES:
        uploaded_files = request.FILES.getlist('images')
        results = []

        for image_file in uploaded_files:
            file_path = default_storage.save(os.path.join('uploads', image_file.name), image_file)
            absolute_path = default_storage.path(file_path)

            try:
                # Load image
                image = cv2.imread(absolute_path)
                if image is None:
                    raise FileNotFoundError(f"Could not read image: {absolute_path}")

                # Detect text boxes
                text_boxes = detect_text_boxes(absolute_path)

                # Merge only small overlapping boxes
                merged_boxes = merge_close_boxes(text_boxes)

                # Extract text using MangaOCR and get final image
                output_image, expanded_boxes = extract_text_with_manga_ocr(image, merged_boxes)
                                # Step 4: **ONLY NOW merge and visualize final bounding boxes**
                final_visualized_image, final_extracted_texts, final_text_cleaned_image, final_merged_text_boxes = post_process_and_visualize_and_extract(output_image, expanded_boxes)

                translated_texts = translate_all_texts(final_extracted_texts)
                # Assuming we have the final cleaned image and the translated texts
                final_overlay_image = overlay_translated_text(final_text_cleaned_image, final_merged_text_boxes, translated_texts)

                # Save the final image with overlaid text
                final_overlay_image_name = f"{os.path.splitext(image_file.name)[0]}_translated_overlay.png"
                final_overlay_image_path = default_storage.path(os.path.join('uploads', final_overlay_image_name))
                cv2.imwrite(final_overlay_image_path, final_overlay_image)

                
                # Step 5: Save the final image
                final_image_name = f"{os.path.splitext(image_file.name)[0]}_merged.png"
                final_image_path = default_storage.path(os.path.join('uploads', final_image_name))
                cv2.imwrite(final_image_path, final_visualized_image)

                cleaned_image_name = f"{os.path.splitext(image_file.name)[0]}_text_cleaned.png"
                cleaned_image_path = default_storage.path(os.path.join('uploads', cleaned_image_name))
                cv2.imwrite(cleaned_image_path, final_text_cleaned_image)
                # Save the processed image with final expanded boxes
                detected_image_name = f"{os.path.splitext(image_file.name)[0]}_final.png"
                detected_image_path = default_storage.path(os.path.join('uploads', detected_image_name))
                cv2.imwrite(detected_image_path, output_image)

                results.append({
                    'original_image_path': file_path.replace('\\', '/'),
                    'detected_bubbles_image_path': os.path.join('uploads', detected_image_name).replace('\\', '/'),
                    'final_merged_image_path': os.path.join('uploads', final_image_name).replace('\\', '/'),
                    'final_extracted_texts': final_extracted_texts,
                    'translated_texts': translated_texts,  # Translated English texts
                })

            except Exception as e:
                results.append({
                    'original_image_path': file_path.replace('\\', '/'),
                    'error': str(e),
                })

        return JsonResponse({"status": "success", "results": results})

    return JsonResponse({'error': 'Invalid request'}, status=400)
