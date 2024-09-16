import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import os
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load pre-trained text detection model (EAST)
EAST_MODEL_PATH = '../frozen_east_text_detection.pb'

def load_east_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"EAST model file not found: {path}")
    net = cv2.dnn.readNet(path)
    return net

def preprocess_image(image, target_size=(320, 320)):
    (H, W) = image.shape[:2]
    rW = W / float(target_size[0])
    rH = H / float(target_size[1])
    image = cv2.resize(image, target_size)
    return image, rW, rH

def detect_text(image, net):
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    
    rectangles = []
    confidences = []
    
    for y in range(0, scores.shape[2]):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, scores.shape[3]):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rectangles.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return rectangles, confidences

def apply_non_max_suppression(rectangles, confidences, overlap_threshold=0.4):
    if not rectangles:
        return []

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rectangles])
    
    try:
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences, 0.5, overlap_threshold)
    except Exception as e:
        logger.error(f"Error in NMSBoxes: {str(e)}")
        return []

    if len(indices) == 0:
        return []
    
    # Handle different return types of cv2.dnn.NMSBoxes
    if isinstance(indices, tuple):
        indices = indices[0]  # In some OpenCV versions, it returns a tuple
    
    return [rectangles[i] for i in indices]

def extract_text_from_region(image, rectangle):
    (x, y, w, h) = rectangle
    roi = image[y:y+h, x:x+w]
    text = pytesseract.image_to_string(roi)
    return text.strip()

def extract_product_info(text):
    patterns = {
        'weight': r'\b(\d+(?:\.\d+)?)\s*(mg|g|kg)\b',
        'volume': r'\b(\d+(?:\.\d+)?)\s*(ml|l|liter|litre)\b',
        'quantity': r'\b(\d+)\s*(pack|piece|pcs|count)\b',
        'price': r'\$\s*(\d+(?:\.\d{2})?)',
    }
    
    info = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            info[key] = [(float(value), unit.lower()) for value, unit in matches]
    
    return info

def process_image(image_path, east_model):
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return None

    orig = image.copy()
    (origH, origW) = image.shape[:2]

    (image, rW, rH) = preprocess_image(image, target_size=(320, 320))
    (H, W) = image.shape[:2]

    rectangles, confidences = detect_text(image, east_model)
    boxes = apply_non_max_suppression(rectangles, confidences)

    if not boxes:
        logger.warning(f"No text regions detected in {image_path}")
        return None

    results = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        padding = 0.05
        startX = max(0, int(startX - padding * (endX - startX)))
        startY = max(0, int(startY - padding * (endY - startY)))
        endX = min(origW, int(endX + padding * (endX - startX)))
        endY = min(origH, int(endY + padding * (endY - startY)))

        roi = orig[startY:endY, startX:endX]
        text = extract_text_from_region(roi, (0, 0, endX-startX, endY-startY))
        results.append(text)

    all_text = " ".join(results)
    product_info = extract_product_info(all_text)

    return product_info

def process_dataset(dataset_path, east_model):
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset directory not found: {dataset_path}")
        return {}

    results = {}
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dataset_path, filename)
            try:
                product_info = process_image(image_path, east_model)
                if product_info:
                    results[filename] = product_info
                else:
                    logger.warning(f"No product info extracted from {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
    return results

if __name__ == "__main__":
    try:
        east_model = load_east_model(EAST_MODEL_PATH)
        
        # Process entire dataset
        dataset_path = '../dataset/images'
        results = process_dataset(dataset_path, east_model)
        logger.info("\nResults for entire dataset:")
        for filename, info in results.items():
            logger.info(f"{filename}: {info}")
        
        logger.info(f"Successfully processed {len(results)} images out of {len(os.listdir(dataset_path))} total images.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
