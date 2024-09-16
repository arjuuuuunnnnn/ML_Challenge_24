import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import easyocr

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def preprocess_image(image_path):
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

        # Apply dilation and erosion to remove noise
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(denoised, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

        # Apply median blur
        img = cv2.medianBlur(img, 3)

        return img
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def extract_text_tesseract(image):
    try:
        text = pytesseract.image_to_string(image, config='--psm 6')
        logger.debug(f"Tesseract extracted text: {text[:100]}...")
        return text
    except Exception as e:
        logger.error(f"Error extracting text with Tesseract: {str(e)}")
        return ""

def extract_text_easyocr(image):
    try:
        result = reader.readtext(image)
        text = ' '.join([item[1] for item in result])
        logger.debug(f"EasyOCR extracted text: {text[:100]}...")
        return text
    except Exception as e:
        logger.error(f"Error extracting text with EasyOCR: {str(e)}")
        return ""

def extract_product_info(text):
    patterns = {
        'weight': r'\b(\d+(?:\.\d+)?)\s*(mg|g|kg)\b',
        'volume': r'\b(\d+(?:\.\d+)?)\s*(ml|l|liter|litre)\b',
        'quantity': r'\b(\d+)\s*(pack|piece|pcs|count)\b',
        'price': r'(?:[\$£€]|USD|EUR|GBP)\s*(\d+(?:\.\d{2})?)',
        'product_name': r'(?i)\b((?:\w+\s){1,4}(?:cereal|snack|drink|juice|milk|yogurt|cheese|food|product))\b',
        'expiry_date': r'\b(?:exp|best before|use by):\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        'brand': r'\b(?:by|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b',
        'size': r'\b(small|medium|large|x-large|xxl)\b',
        'color': r'\b(red|blue|green|yellow|black|white|purple|orange|pink)\b',
    }
    
    info = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            if key in ['weight', 'volume', 'quantity', 'price']:
                info[key] = [(float(value), unit.lower()) for value, unit in matches]
            else:
                info[key] = list(set(matches))  # Remove duplicates
    
    logger.debug(f"Extracted info: {info}")
    return info

def process_image(image_path):
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)
        if preprocessed_image is None:
            return None

        # Extract text using both Tesseract and EasyOCR
        text_tesseract = extract_text_tesseract(preprocessed_image)
        text_easyocr = extract_text_easyocr(preprocessed_image)

        # Combine texts
        combined_text = f"{text_tesseract} {text_easyocr}"

        # Extract product information from the combined text
        product_info = extract_product_info(combined_text)

        return product_info
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def process_dataset(dataset_path, max_workers=4):
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset directory not found: {dataset_path}")
        return {}

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_filename = {executor.submit(process_image, os.path.join(dataset_path, filename)): filename 
                              for filename in os.listdir(dataset_path) 
                              if filename.lower().endswith(('.png', '.jpg', '.jpeg'))}
        
        for future in as_completed(future_to_filename):
            filename = future_to_filename[future]
            try:
                product_info = future.result()
                if product_info:
                    results[filename] = product_info
                else:
                    logger.warning(f"No product info extracted from {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")

    return results

def main():
    try:
        # Check Tesseract version
        tesseract_version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract version: {tesseract_version}")

        # Process entire dataset
        dataset_path = '../dataset/images'
        results = process_dataset(dataset_path)
        
        logger.info("\nResults for entire dataset:")
        for filename, info in results.items():
            logger.info(f"{filename}: {info}")
        
        total_images = len([f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        logger.info(f"Successfully processed {len(results)} images out of {total_images} total images.")
        
        # Additional analysis
        if results:
            most_common_product = max(results.values(), key=lambda x: len(x.get('product_name', [])))
            logger.info(f"Most common product: {most_common_product.get('product_name', ['Unknown'])[0]}")
            
            prices = [price for info in results.values() for price, _ in info.get('price', [])]
            if prices:
                avg_price = np.mean(prices)
                logger.info(f"Average price: ${avg_price:.2f}")
            else:
                logger.info("No price information found.")
        else:
            logger.warning("No results to analyze.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
