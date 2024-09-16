import pytesseract
from PIL import Image
import re
import spacy
import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Load English tokenizer and entity recognizer from SpaCy
nlp = spacy.load('en_core_web_sm')

# Initialize Chroma DB Client
chroma_client = chromadb.Client()

# Create or get a Chroma DB collection
collection = chroma_client.create_collection(name="entity_vectors")

# Load Sentence-BERT for vectorization
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the entity-unit mapping
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                    'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

# Function to perform OCR on image
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Function to extract entities from text (both measurements and named entities)
def extract_entities(text):
    extracted_entities = {}
    
    # Extract measurement values and units using regex
    pattern = re.compile(r"(\d+\.?\d*)\s*([a-zA-Z]+)")
    
    for match in pattern.finditer(text):
        value, unit = match.groups()
        unit = unit.lower()

        # Map the extracted units to predefined categories
        for entity, units in entity_unit_map.items():
            if unit in units:
                extracted_entities[entity] = f"{float(value):.2f} {unit}"  # Format as float

    # Use SpaCy to extract other entities (if necessary)
    doc = nlp(text)
    for ent in doc.ents:
        extracted_entities[ent.label_] = ent.text  # Add SpaCy's recognized entities
    
    return extracted_entities


# Function to store entities in Chroma DB
def store_entities_in_chroma(entities, image_filename):
    for entity, value in entities.items():
        entity_str = f"{entity}: {value}"  # Combine entity name and value
        vector = model.encode(entity_str).tolist()  # Convert the entity into a vector and then to a list
        
        # Insert the entity and its vector into Chroma DB
        collection.add(documents=[entity_str], embeddings=[vector], metadatas=[{'image': image_filename}], ids=[image_filename + "_" + entity])


# Function to search for a similar entity in Chroma DB
def search_similar_entity_in_chroma(query):
    query_vector = model.encode(query).tolist()  # Convert the query vector to a list
    results = collection.query(query_embeddings=[query_vector], n_results=1)
    
    # Check if we have results and their distance
    if results['distances'][0][0] < 0.2:  # If distance is less than threshold (e.g., 0.2)
        return results['documents'][0][0], results['distances'][0][0]
    return None, None


# Function to create a CSV output from a folder of images
def create_output_csv_from_images(image_folder, output_csv):
    results = []

    # Iterate over each image in the folder
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(('.jpg', '.png', '.jpeg')):  # Only process image files
            image_path = os.path.join(image_folder, image_filename)
            
            # Extract text from the image
            extracted_text = extract_text_from_image(image_path)
            
            # Extract entities from the text
            entities = extract_entities(extracted_text)
            
            # Store extracted entities in Chroma DB
            store_entities_in_chroma(entities, image_filename)
            
            # Prioritize a specific entity for querying, e.g., 'width'
            if 'width' in entities:
                query_entity = f"width: {entities['width']}"
                matched_entity, distance = search_similar_entity_in_chroma(query_entity)
                
                if matched_entity:
                    results.append({
                        'index': image_filename,
                        'prediction': matched_entity,
                        'distance': distance
                    })
                else:
                    results.append({
                        'index': image_filename,
                        'prediction': entities.get('width', ''),
                        'distance': 'N/A'
                    })

    # Convert results into a DataFrame
    output_df = pd.DataFrame(results)
    
    # Save results to CSV
    output_df.to_csv(output_csv, index=False)

# Example usage
if __name__ == "__main__":
    image_folder_path = '../dataset/images'  # Replace with the actual path to the folder containing images
    output_csv_path = './output_predictions.csv'  # Path to save the output CSV
    
    create_output_csv_from_images(image_folder_path, output_csv_path)
