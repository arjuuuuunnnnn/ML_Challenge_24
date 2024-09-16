import os
import re
import pandas as pd
import torch
import torch.cuda.amp as amp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from paddleocr import PaddleOCR
from tqdm import tqdm

# Initialize GPU-accelerated OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

# Use mixed precision
scaler = amp.GradScaler()

# Entity-unit map
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

# Generate regex patterns
patterns = []
for entity, units in entity_unit_map.items():
    for unit in units:
        pattern = rf'(\d+\.?\d*)\s*({unit}s?)'
        patterns.append((pattern, entity, unit))

class ImageDataset(Dataset):
    def _init_(self, image_folder):
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _len_(self):
        return len(self.image_paths)

    def _getitem_(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, image_path

@torch.jit.script
def extract_measurements(text: str, patterns: list) -> list:
    measurements = []
    for pattern, entity, unit in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            value = float(match[0])
            measurements.append((entity, f"{value:.2f} {unit}"))
    return measurements

def process_batch(batch_images, batch_paths):
    # Perform OCR on the batch
    ocr_results = ocr.ocr(batch_images.cpu().numpy().transpose(0, 2, 3, 1))
    
    batch_results = []
    
    for image_result, image_path in zip(ocr_results, batch_paths):
        text = ' '.join([line[1][0] for line in image_result])
        measurements = extract_measurements(text, patterns)
        
        batch_results.append({
            'filename': os.path.basename(image_path),
            'measurements': measurements
        })
    
    return batch_results

def main():
    image_folder = '../dataset/train_images'
    output_csv = './output_predictions.csv'
    
    dataset = ImageDataset(image_folder)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=4, pin_memory=True)
    
    all_results = []
    
    for batch_images, batch_paths in tqdm(dataloader):
        batch_images = batch_images.cuda(non_blocking=True)
        batch_results = process_batch(batch_images, batch_paths)
        all_results.extend(batch_results)
    
    # Format results
    formatted_results = []
    for result in all_results:
        row = {'filename': result['filename']}
        for entity, measurement in result['measurements']:
            if entity not in row:
                row[entity] = measurement
        formatted_results.append(row)
    
    # Create DataFrame and save to CSV
    output_df = pd.DataFrame(formatted_results)
    output_df.to_csv(output_csv, index=False)
    
    print(f"Processing complete. Results saved to: {output_csv}")

if _name_ == "_main_":
    torch.backends.cudnn.benchmark = True
    main()
