import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.transform import resize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the entity_unit_map
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

def advanced_preprocess(image):
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply adaptive thresholding
    thresh = threshold_local(gray, block_size=35, offset=10, method="gaussian")
    binary = gray > thresh

    # Remove border artifacts
    cleared = clear_border(binary)

    # Apply morphological closing
    closed = closing(cleared, square(3))

    return closed

def detect_text_regions(image):
    # Label connected regions
    labeled = label(image)

    # Get region properties
    regions = regionprops(labeled)

    # Filter regions based on area and aspect ratio
    text_regions = []
    for region in regions:
        if region.area > 100 and 0.1 < region.eccentricity < 0.995:
            text_regions.append(region.bbox)

    return text_regions

def extract_features(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Preprocess the image
    preprocessed = advanced_preprocess(image)
    
    # Detect text regions
    text_regions = detect_text_regions(preprocessed)
    
    features = {}
    for i, region in enumerate(text_regions):
        # Extract the region
        y1, x1, y2, x2 = region
        roi = image[y1:y2, x1:x2]
        
        # Perform OCR on the region
        ocr_result = pytesseract.image_to_string(roi, config='--psm 6')
        
        # Search for entity-value pairs
        for entity, units in entity_unit_map.items():
            pattern = r'(\d+(?:\.\d+)?)\s*(' + '|'.join(units) + ')'
            match = re.search(pattern, ocr_result.lower())
            if match:
                value, unit = match.groups()
                features[entity] = f"{float(value):.2f} {unit}"
    
    return features

def process_images(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(directory, filename)
            try:
                features = extract_features(image_path)
                for entity in entity_unit_map.keys():
                    results.append({
                        'index': f"{filename}_{entity}",
                        'prediction': features.get(entity, "")
                    })
                logging.info(f"Processed {filename} successfully")
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
    return results

def visualize_results(image_path, features):
    image = cv2.imread(image_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Features")
    for i, (entity, value) in enumerate(features.items()):
        plt.text(10, 30 + i*30, f"{entity}: {value}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.axis('off')
    plt.show()

# Directory containing your images
image_directory = "../dataset/images"

# Process all images
all_results = process_images(image_directory)

# Create DataFrame and save to CSV
df = pd.DataFrame(all_results)
df.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'")

# Visualize results for a sample image
sample_image = os.path.join(image_directory, os.listdir(image_directory)[0])
sample_features = extract_features(sample_image)
visualize_results(sample_image, sample_features)

# Print sample predictions
print("\nSample predictions:")
print(df.head(10))

# Calculate confidence score based on number of features extracted
df['confidence'] = df.groupby('index')['prediction'].transform(lambda x: (x != "").sum() / len(entity_unit_map))
average_confidence = df['confidence'].mean()
print(f"\nAverage confidence score: {average_confidence:.2f}")
