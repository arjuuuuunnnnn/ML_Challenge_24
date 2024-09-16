import os
import cv2
import numpy as np
import pytesseract
import re
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Apply dilation to connect text components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gray = cv2.dilate(gray, kernel, iterations=1)

    return gray

def extract_text(image):
    # Perform OCR on the preprocessed image
    text = pytesseract.image_to_string(image)
    return text

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

def process_image(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is None:
        return None

    # Extract text from the preprocessed image
    text = extract_text(preprocessed_image)

    # Extract product information from the text
    product_info = extract_product_info(text)

    return product_info

def process_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset directory not found: {dataset_path}")
        return {}

    results = {}
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dataset_path, filename)
            try:
                product_info = process_image(image_path)
                if product_info:
                    results[filename] = product_info
                else:
                    logger.warning(f"No product info extracted from {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
    return results

if __name__ == "__main__":
    try:
        # Process entire dataset
        dataset_path = '../dataset/images'
        results = process_dataset(dataset_path)
        logger.info("\nResults for entire dataset:")
        for filename, info in results.items():
            logger.info(f"{filename}: {info}")
        
        logger.info(f"Successfully processed {len(results)} images out of {len(os.listdir(dataset_path))} total images.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
