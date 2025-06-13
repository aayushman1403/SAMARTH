import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeatureEngineer:
    """Class to engineer features from the crime dataset"""
    
    def __init__(self):
        self.encoder = None
        self.scaler = None
        self.categorical_features = ['CrimeType', 'Location']
    
    def calculate_days_since_filing(self, df):
        """
        Calculate the number of days since the case was filed
        
        Args:
            df (pandas.DataFrame): The dataframe with a FilingDate column
            
        Returns:
            pandas.DataFrame: Dataframe with added days_since_filing column
        """
        result_df = df.copy()
        
        # Ensure FilingDate is datetime
        if not pd.api.types.is_datetime64_any_dtype(result_df['FilingDate']):
            result_df['FilingDate'] = pd.to_datetime(result_df['FilingDate'])
        
        # Calculate days since filing
        current_date = datetime.now().date()
        result_df['days_since_filing'] = (current_date - result_df['FilingDate'].dt.date).dt.days
        
        return result_df
    
    def get_date_features(self, df):
        """
        Extract date-related features from FilingDate
        
        Args:
            df (pandas.DataFrame): The dataframe with a FilingDate column
            
        Returns:
            pandas.DataFrame: Dataframe with added date features
        """
        result_df = df.copy()
        
        # Ensure FilingDate is datetime
        if not pd.api.types.is_datetime64_any_dtype(result_df['FilingDate']):
            result_df['FilingDate'] = pd.to_datetime(result_df['FilingDate'])
        
        # Extract date features
        result_df['filing_month'] = result_df['FilingDate'].dt.month
        result_df['filing_year'] = result_df['FilingDate'].dt.year
        result_df['filing_day_of_week'] = result_df['FilingDate'].dt.dayofweek
        result_df['filing_is_weekend'] = result_df['filing_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        return result_df
    
    def extract_text_features(self, df):
        """
        Extract features from the description text
        
        Args:
            df (pandas.DataFrame): The dataframe with a Description column
            
        Returns:
            pandas.DataFrame: Dataframe with added text features
        """
        result_df = df.copy()
        
        # Check if Description column exists
        if 'Description' not in result_df.columns:
            raise ValueError("Description column not found in the dataframe")
        
        # Calculate text length features
        result_df['description_length'] = result_df['Description'].apply(lambda x: len(str(x)))
        result_df['description_word_count'] = result_df['Description'].apply(lambda x: len(str(x).split()))
        
        # Check for specific keywords that might indicate priority
        priority_keywords = [
            'weapon', 'gun', 'knife', 'armed', 'violent', 'injured', 'child', 
            'elderly', 'vulnerable', 'hospital', 'emergency', 'severe', 'death', 
            'threat', 'blood', 'trauma', 'victim', 'wounds', 'attack'
        ]
        
        # Count priority keywords in description
        result_df['priority_keyword_count'] = result_df['Description'].apply(
            lambda x: sum(1 for keyword in priority_keywords if keyword.lower() in str(x).lower())
        )
        
        return result_df
    
    def encode_categorical_features(self, df, train=True):
        """
        One-hot encode categorical features
        
        Args:
            df (pandas.DataFrame): The dataframe with categorical columns
            train (bool): Whether this is training data (to fit the encoder) or not
            
        Returns:
            pandas.DataFrame: Dataframe with encoded categorical features
        """
        result_df = df.copy()
        
        # Check if categorical features exist
        available_categorical = [col for col in self.categorical_features if col in result_df.columns]
        if not available_categorical:
            return result_df
        
        if train:
            # Initialize and fit encoder
            self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.encoder.fit(result_df[available_categorical])
        
        if self.encoder is None:
            raise ValueError("Encoder not initialized. Call with train=True first.")
        
        # Transform categorical features
        encoded_features = self.encoder.transform(result_df[available_categorical])
        
        # Create dataframe with encoded features
        feature_names = self.encoder.get_feature_names_out(available_categorical)
        encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=result_df.index)
        
        # Concatenate with original dataframe
        result_df = pd.concat([result_df.drop(available_categorical, axis=1), encoded_df], axis=1)
        
        return result_df
    
    def scale_numerical_features(self, df, numerical_features=None, train=True):
        """
        Scale numerical features
        
        Args:
            df (pandas.DataFrame): The dataframe with numerical columns
            numerical_features (list): List of numerical features to scale
            train (bool): Whether this is training data (to fit the scaler) or not
            
        Returns:
            pandas.DataFrame: Dataframe with scaled numerical features
        """
        result_df = df.copy()
        
        # Default numerical features if not specified
        if numerical_features is None:
            numerical_features = [
                'days_since_filing', 'description_length', 'description_word_count',
                'priority_keyword_count', 'sentiment_compound', 'sentiment_positive',
                'sentiment_negative', 'sentiment_neutral'
            ]
        
        # Filter to only include columns that exist in the dataframe
        available_numerical = [col for col in numerical_features if col in result_df.columns]
        
        if not available_numerical:
            return result_df
        
        if train:
            # Initialize and fit scaler
            self.scaler = StandardScaler()
            self.scaler.fit(result_df[available_numerical])
        
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Call with train=True first.")
        
        # Transform numerical features
        result_df[available_numerical] = self.scaler.transform(result_df[available_numerical])
        
        return result_df
    
    def engineer_features(self, df, train=True):
        """
        Apply all feature engineering steps
        
        Args:
            df (pandas.DataFrame): The input dataframe
            train (bool): Whether this is training data or not
            
        Returns:
            pandas.DataFrame: Dataframe with engineered features
        """
        # Apply individual feature engineering steps
        result_df = self.calculate_days_since_filing(df)
        result_df = self.get_date_features(result_df)
        result_df = self.extract_text_features(result_df)
        result_df = self.encode_categorical_features(result_df, train)
        
        # Identify numerical features to scale
        numerical_features = [
            'days_since_filing', 'description_length', 'description_word_count',
            'priority_keyword_count', 'sentiment_compound', 'sentiment_positive',
            'sentiment_negative', 'sentiment_neutral'
        ]
        
        result_df = self.scale_numerical_features(result_df, numerical_features, train)
        
        return result_df
    
    def prepare_model_input(self, df, drop_columns=None):
        """
        Prepare final feature set for model input
        
        Args:
            df (pandas.DataFrame): The dataframe with engineered features
            drop_columns (list): Columns to exclude from model input
            
        Returns:
            pandas.DataFrame: Dataframe ready for model input
        """
        result_df = df.copy()
        
        # Default columns to drop
        if drop_columns is None:
            drop_columns = ['CaseID', 'FilingDate', 'Description', 'CleanedDescription', 
                           'PriorityLabel', 'sentiment_category']
        
        # Keep only columns that exist in the dataframe
        columns_to_drop = [col for col in drop_columns if col in result_df.columns]
        
        # Drop unnecessary columns
        if columns_to_drop:
            X = result_df.drop(columns_to_drop, axis=1)
        else:
            X = result_df
            
        # Extract target if available
        if 'PriorityLabel' in result_df.columns:
            y = result_df['PriorityLabel']
            return X, y
        else:
            return X, None

if __name__ == "__main__":
    # Test the feature engineer
    from data_preprocessor import DataPreprocessor
    from sentiment_analyzer import SentimentAnalyzer
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('crime_data_synthetic.xlsx')
    
    if df is not None:
        processed_df = preprocessor.preprocess_data(df)
        
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_df = sentiment_analyzer.add_sentiment_features(processed_df)
        
        feature_engineer = FeatureEngineer()
        engineered_df = feature_engineer.engineer_features(sentiment_df)
        
        print(f"Feature engineering completed. Dataframe shape: {engineered_df.shape}")
        
        X, y = feature_engineer.prepare_model_input(engineered_df)
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        print(f"Feature columns: {list(X.columns)}")
