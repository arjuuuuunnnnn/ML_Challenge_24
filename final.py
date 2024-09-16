import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
import spacy
import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from transformers import pipeline

# Load English tokenizer and entity recognizer from SpaCy
nlp = spacy.load('en_core_web_sm')

# Initialize Chroma DB Client
chroma_client = chromadb.Client()

# Create or get a Chroma DB collection
collection = chroma_client.create_collection(name="entity_vectors")

# Load Sentence-BERT for vectorization
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize BERT-based NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

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

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return Image.fromarray(thresh)

def extract_text_from_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(preprocessed_image)
    return text

def extract_entities(text):
    extracted_entities = {}
    
    # Extract measurement values and units using regex
    pattern = re.compile(r"(\d+\.?\d*)\s*([a-zA-Z]+)")
    
    for match in pattern.finditer(text):
        value, unit = match.groups()
        unit = unit.lower()

        for entity, units in entity_unit_map.items():
            if unit in units:
                extracted_entities[entity] = f"{float(value):.2f} {unit}"

    # Use SpaCy for additional entity recognition
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ not in extracted_entities:
            extracted_entities[ent.label_] = ent.text

    # Use BERT-based NER for additional entity recognition
    ner_results = ner_pipeline(text)
    for result in ner_results:
        if result['entity'] not in extracted_entities:
            extracted_entities[result['entity']] = result['word']

    return extracted_entities

def store_entities_in_chroma(entities, image_filename):
    for entity, value in entities.items():
        entity_str = f"{entity}: {value}"
        vector = model.encode(entity_str).tolist()
        collection.add(documents=[entity_str], embeddings=[vector], metadatas=[{'image': image_filename}], ids=[image_filename + "_" + entity])

def search_similar_entity_in_chroma(query):
    query_vector = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=5)
    
    best_match = None
    best_score = 0

    for doc, distance in zip(results['documents'][0], results['distances'][0]):
        similarity_score = fuzz.ratio(query, doc)
        if similarity_score > best_score:
            best_score = similarity_score
            best_match = doc

    if best_score > 80:  # Threshold for considering a match
        return best_match, best_score
    return None, None

def create_output_csv_from_images(image_folder, output_csv):
    results = []
    index = 0

    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_folder, image_filename)
            extracted_text = extract_text_from_image(image_path)
            entities = extract_entities(extracted_text)
            store_entities_in_chroma(entities, image_filename)
            
            for entity, value in entities.items():
                query_entity = f"{entity}: {value}"
                matched_entity, similarity = search_similar_entity_in_chroma(query_entity)
                
                if matched_entity:
                    matched_value = matched_entity.split(': ')[1] if ': ' in matched_entity else matched_entity
                    results.append({
                        'index': str(index),
                        'prediction': matched_value
                    })
                else:
                    results.append({
                        'index': str(index),
                        'prediction': value
                    })
                index += 1

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False, header=False)

if __name__ == "__main__":
    image_folder_path = '../dataset/images'
    output_csv_path = './output_predictions.csv'
    create_output_csv_from_images(image_folder_path, output_csv_path)
