import pandas as pd
import numpy as np
import re
import string
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class DataPreprocessor:
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def load_data(self, file_path):
        """Load data from Excel file"""
        try:
            df = pd.read_excel(file_path)
            print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_text(self, text):
        """Clean text data by removing special characters, stopwords, and lemmatizing"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        tokens = word_tokenize(text)
        
        cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Join tokens back into a string
        cleaned_text = ' '.join(cleaned_tokens)
        
        return cleaned_text
    
    def preprocess_data(self, df):
        """Preprocess structured and text data"""
        # Create a copy to avoid modifying the original dataframe
        processed_df = df.copy()
        
        # Clean text data in Description column
        processed_df['CleanedDescription'] = processed_df['Description'].apply(self.clean_text)
        
        # Convert FilingDate to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(processed_df['FilingDate']):
            processed_df['FilingDate'] = pd.to_datetime(processed_df['FilingDate'])
        
        # Drop rows with missing values in key columns
        essential_columns = ['CaseID', 'FilingDate', 'CrimeType', 'Description', 'PriorityLabel']
        processed_df = processed_df.dropna(subset=essential_columns)
        
        # Check for and remove duplicate CaseIDs
        if processed_df['CaseID'].duplicated().any():
            processed_df = processed_df.drop_duplicates(subset=['CaseID'])
        
        # Ensure PriorityLabel is binary (0 or 1)
        processed_df['PriorityLabel'] = processed_df['PriorityLabel'].astype(int)
        if not set(processed_df['PriorityLabel'].unique()).issubset({0, 1}):
            raise ValueError("PriorityLabel contains values other than 0 and 1")
            
        return processed_df

if __name__ == "__main__":
    # Test the data preprocessor
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('crime_data_synthetic.xlsx')
    if df is not None:
        processed_df = preprocessor.preprocess_data(df)
        print(f"Data preprocessing completed. Processed dataframe shape: {processed_df.shape}")
