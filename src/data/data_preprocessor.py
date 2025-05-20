"""
Data preprocessing for software quality prediction models.
Handles feature engineering, normalization, and dataset creation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Preprocess data for software quality prediction."""
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.scaler = StandardScaler()
        self.preprocessor = None
        self.numeric_features = None
        self.categorical_features = None
    
    def process_features(self, features_df, drop_columns=None):
        """
        Process a dataframe of features.
        
        Args:
            features_df (pd.DataFrame): DataFrame containing features
            drop_columns (list): Columns to drop from the dataset
        
        Returns:
            pd.DataFrame: Processed features
        """
        # Copy the dataframe to avoid modifying the original
        df = features_df.copy()
        
        # Drop specified columns
        if drop_columns:
            df = df.drop(columns=[col for col in drop_columns if col in df.columns], errors='ignore')
        
        # Identify numeric and categorical features
        self.numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Convert list columns to string representation
        for col in df.columns:
            if isinstance(df[col].iloc[0], list):
                df[col] = df[col].apply(lambda x: ','.join(x) if x else '')
                if col not in self.categorical_features:
                    self.categorical_features.append(col)
        
        # Handle missing values
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        # Fit and transform the data
        processed_data = self.preprocessor.fit_transform(df)
        
        # Create a new DataFrame with transformed features
        # Get feature names from OneHotEncoder
        if self.categorical_features:
            onehot_feature_names = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
        else:
            onehot_feature_names = []
        
        # Combine feature names
        feature_names = self.numeric_features + list(onehot_feature_names)
        
        # Create new DataFrame
        processed_df = pd.DataFrame(processed_data, columns=feature_names)
        
        return processed_df
    
    def transform_features(self, features_df):
        """
        Transform new features using the fitted preprocessor.
        
        Args:
            features_df (pd.DataFrame): DataFrame containing features
        
        Returns:
            pd.DataFrame: Transformed features
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fit yet. Call process_features first.")
        
        # Copy the dataframe
        df = features_df.copy()
        
        # Convert list columns to string representation
        for col in df.columns:
            if isinstance(df[col].iloc[0], list):
                df[col] = df[col].apply(lambda x: ','.join(x) if x else '')
        
        # Transform the data
        transformed_data = self.preprocessor.transform(df)
        
        # Get feature names from OneHotEncoder
        if self.categorical_features:
            onehot_feature_names = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
        else:
            onehot_feature_names = []
        
        # Combine feature names
        feature_names = self.numeric_features + list(onehot_feature_names)
        
        # Create new DataFrame
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names)
        
        return transformed_df
    
    def create_dataset(self, repositories_path, quality_scores_path, test_size=0.2, random_state=42):
        """
        Create a dataset for training quality models.
        
        Args:
            repositories_path (str): Path to directory containing repository features
            quality_scores_path (str): Path to file containing quality scores
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) splits for each quality dimension
        """
        # Load quality scores
        quality_scores = pd.read_csv(quality_scores_path)
        
        # Load repository features
        features_dfs = []
        
        for repo_id in quality_scores['repository_id']:
            feature_file = os.path.join(repositories_path, f"{repo_id}_features.json")
            if os.path.exists(feature_file):
                repo_features = pd.read_json(feature_file, orient='records', lines=True)
                repo_features['repository_id'] = repo_id
                features_dfs.append(repo_features)
        
        if not features_dfs:
            raise ValueError("No feature files found for the repositories in quality scores")
        
        # Combine features
        features_df = pd.concat(features_dfs, ignore_index=True)
        
        # Merge with quality scores
        merged_df = pd.merge(features_df, quality_scores, on='repository_id')
        
        # Drop repository_id column
        merged_df = merged_df.drop(columns=['repository_id'])
        
        # Get quality dimensions
        quality_dimensions = ['maintainability', 'reliability', 'scalability', 'security', 'efficiency']
        
        # Create train-test splits for each dimension
        splits = {}
        
        # Process features once
        X = merged_df.drop(columns=quality_dimensions)
        X_processed = self.process_features(X)
        
        for dimension in quality_dimensions:
            y = merged_df[dimension]
            
            # Create train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=random_state
            )
            
            splits[dimension] = (X_train, X_test, y_train, y_test)
        
        return splits
    
    def prepare_synthetic_dataset(self, features_df, feature_weights=None):
        """
        Create a synthetic dataset for training when real labels are not available.
        
        Args:
            features_df (pd.DataFrame): DataFrame containing features
            feature_weights (dict): Dictionary of feature weights for each quality dimension
        
        Returns:
            tuple: DataFrame with features and synthetic quality scores
        """
        # Default feature weights
        if feature_weights is None:
            feature_weights = {
                'maintainability': {
                    'complexity': -0.3,
                    'comment_ratio': 0.2,
                    'maintainability_index': 0.3,
                    'circular_dependencies': -0.2,
                    'lint_errors': -0.2
                },
                'reliability': {
                    'test_to_code_ratio': 0.3,
                    'test_frameworks_count': 0.2,
                    'lint_errors': -0.2,
                    'complexity': -0.2
                },
                'scalability': {
                    'avg_fan_out': -0.25,
                    'circular_dependencies': -0.3,
                    'avg_dependencies': -0.2,
                    'complexity': -0.25
                },
                'security': {
                    'dependency_count': -0.15,
                    'has_dependency_management': 0.3,
                    'lint_errors': -0.2,
                    'test_to_code_ratio': 0.2
                },
                'efficiency': {
                    'complexity': -0.3,
                    'halstead_effort': -0.3,
                    'loc': -0.2
                }
            }
        
        # Process the features
        processed_df = self.process_features(features_df)
        
        # Create synthetic scores for each quality dimension
        quality_scores = {}
        
        for dimension, weights in feature_weights.items():
            # Initialize scores with random noise
            scores = np.random.normal(0.5, 0.1, len(processed_df))
            
            # Apply feature weights
            for feature, weight in weights.items():
                if feature in features_df.columns:
                    # Normalize the feature values between 0 and 1
                    feature_values = features_df[feature].values
                    if feature_values.min() != feature_values.max():
                        normalized_values = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
                    else:
                        normalized_values = np.zeros_like(feature_values)
                    
                    # Apply weight
                    scores += weight * normalized_values
            
            # Scale scores to be between 0 and 1
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            
            # Add some noise
            scores += np.random.normal(0, 0.05, len(scores))
            
            # Clip to 0-1 range
            scores = np.clip(scores, 0, 1)
            
            quality_scores[dimension] = scores
        
        # Create a DataFrame with quality scores
        quality_df = pd.DataFrame(quality_scores)
        
        return features_df, quality_df