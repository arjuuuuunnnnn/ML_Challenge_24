import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import easyocr
from skimage import io, color, filters, restoration
from scipy import ndimage

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def advanced_preprocess_image(image_path):
    try:
        # Read the image
        image = io.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Convert to grayscale if it's not already
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image

        # Apply contrast stretching
        p2, p98 = np.percentile(gray, (2, 98))
        stretched = filters.rescale_intensity(gray, in_range=(p2, p98))

        # Denoise using Non-local Means
        denoised = restoration.denoise_nl_means(stretched, h=0.1, fast_mode=True)

        # Apply adaptive thresholding
        thresh = filters.threshold_local(denoised, block_size=35, offset=10)
        binary = denoised > thresh

        # Perform morphological operations
        selem = np.ones((3, 3))
        eroded = ndimage.binary_erosion(binary, structure=selem)
        dilated = ndimage.binary_dilation(eroded, structure=selem)

        # Convert back to uint8
        processed = (dilated * 255).astype(np.uint8)

        return processed
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def extract_text_tesseract(image):
    try:
        # Apply image preprocessing specific to Tesseract
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(image, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

        # Use multiple PSM modes and combine results
        configs = ['--psm 6', '--psm 11', '--psm 3']
        texts = []
        for config in configs:
            text = pytesseract.image_to_string(img, config=config)
            texts.append(text)
        
        combined_text = ' '.join(texts)
        logger.debug(f"Tesseract extracted text: {combined_text[:100]}...")
        return combined_text
    except Exception as e:
        logger.error(f"Error extracting text with Tesseract: {str(e)}")
        return ""

def extract_text_easyocr(image):
    try:
        # Apply image preprocessing specific to EasyOCR
        img = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        result = reader.readtext(img, detail=0, paragraph=True)
        text = ' '.join(result)
        logger.debug(f"EasyOCR extracted text: {text[:100]}...")
        return text
    except Exception as e:
        logger.error(f"Error extracting text with EasyOCR: {str(e)}")
        return ""

def extract_product_info(text):
    patterns = {
        'weight': r'\b(\d+(?:\.\d+)?)\s*(mg|g|kg|oz|lbs?)\b',
        'volume': r'\b(\d+(?:\.\d+)?)\s*(ml|l|liter|litre|fl\s*oz)\b',
        'quantity': r'\b(\d+)\s*(pack|piece|pcs|count|ct)\b',
        'price': r'(?:[\$£€]|USD|EUR|GBP)\s*(\d+(?:\.\d{2})?)',
        'product_name': r'(?i)\b((?:\w+\s){1,5}(?:cereal|snack|drink|juice|milk|yogurt|cheese|food|product))\b',
        'expiry_date': r'\b(?:exp|best before|use by):\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
        'brand': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b',
        'size': r'\b(small|medium|large|x-large|xxl)\b',
        'color': r'\b(red|blue|green|yellow|black|white|purple|orange|pink)\b',
        'ingredients': r'ingredients:?\s*(.+?)(?:\.|$)',
        'nutrition_facts': r'(?:nutrition facts|nutritional information)(.+?)(?=\n\n|\Z)',
    }
    
    info = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        if matches:
            if key in ['weight', 'volume', 'quantity', 'price']:
                info[key] = [(float(value), unit.lower()) for value, unit in matches]
            elif key in ['ingredients', 'nutrition_facts']:
                info[key] = [m.strip() for m in matches]
            else:
                info[key] = list(set(matches))  # Remove duplicates
    
    logger.debug(f"Extracted info: {info}")
    return info

def process_image(image_path):
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Preprocess the image
        preprocessed_image = advanced_preprocess_image(image_path)
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
                              if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))}
        
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
        
        total_images = len([f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])
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

            # Additional statistics
            brands = [brand for info in results.values() for brand in info.get('brand', [])]
            if brands:
                most_common_brand = max(set(brands), key=brands.count)
                logger.info(f"Most common brand: {most_common_brand}")

            weights = [weight for info in results.values() for weight, unit in info.get('weight', [])]
            if weights:
                avg_weight = np.mean(weights)
                logger.info(f"Average weight: {avg_weight:.2f}")

        else:
            logger.warning("No results to analyze.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
