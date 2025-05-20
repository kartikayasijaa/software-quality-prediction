"""
Quality prediction models for different software quality dimensions.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns

class QualityModel:
    """Base model for predicting software quality dimensions."""
    
    def __init__(self, model_type='random_forest', feature_selection=True):
        """
        Initialize a quality prediction model.
        
        Args:
            model_type (str): Type of model to use ('random_forest', 'gradient_boosting', 'xgboost', 'ridge')
            feature_selection (bool): Whether to perform feature selection
        """
        self.model_type = model_type
        self.feature_selection = feature_selection
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.feature_importances = {}
    
    def _create_model(self):
        """Create the ML model based on model_type."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X, y):
        """Train the model on the given features and target."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        if self.feature_selection:
            base_model = self._create_model()
            self.feature_selector = SelectFromModel(base_model, threshold='median')
            self.feature_selector.fit(X_scaled, y)
            X_selected = self.feature_selector.transform(X_scaled)
            
            # Get selected feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_features = [X.columns[i] for i in selected_indices]
            
            # Create and train the final model
            self.model = self._create_model()
            self.model.fit(X_selected, y)
            
            # Store feature importances
            if hasattr(self.model, 'feature_importances_'):
                for i, importance in enumerate(self.model.feature_importances_):
                    self.feature_importances[self.selected_features[i]] = importance
        else:
            # Train without feature selection
            self.model = self._create_model()
            self.model.fit(X_scaled, y)
            
            # Store feature importances
            if hasattr(self.model, 'feature_importances_'):
                for i, importance in enumerate(self.model.feature_importances_):
                    self.feature_importances[X.columns[i]] = importance
    
    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Apply feature selection if used
        if self.feature_selection and self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
            return self.model.predict(X_selected)
        else:
            return self.model.predict(X_scaled)
    
    def evaluate(self, X, y):
        """Evaluate the model on test data."""
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'predictions': predictions,
            'actual': y
        }
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation."""
        X_scaled = self.scaler.fit_transform(X)
        
        if self.feature_selection:
            base_model = self._create_model()
            self.feature_selector = SelectFromModel(base_model, threshold='median')
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            model = self._create_model()
            scores = cross_val_score(model, X_selected, y, cv=cv, scoring='neg_mean_squared_error')
        else:
            model = self._create_model()
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
        
        rmse_scores = np.sqrt(-scores)
        return {
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std(),
            'all_rmse': rmse_scores
        }
    
    def optimize_hyperparameters(self, X, y, param_grid=None, cv=5):
        """Optimize hyperparameters using grid search."""
        X_scaled = self.scaler.fit_transform(X)
        
        if self.feature_selection:
            base_model = self._create_model()
            self.feature_selector = SelectFromModel(base_model, threshold='median')
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            features = X_selected
        else:
            features = X_scaled
        
        # Default parameter grids if none provided
        if param_grid is None:
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            elif self.model_type == 'xgboost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            elif self.model_type == 'ridge':
                param_grid = {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
        
        # Create the model
        base_model = self._create_model()
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(features, y)
        
        # Get best parameters and score
        best_params = grid_search.best_params_
        best_score = np.sqrt(-grid_search.best_score_)
        
        # Create model with best parameters
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(**best_params, random_state=42)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**best_params, random_state=42)
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**best_params, random_state=42)
        elif self.model_type == 'ridge':
            self.model = Ridge(**best_params, random_state=42)
        
        # Train the model
        self.model.fit(features, y)
        
        # Store feature importances
        if hasattr(self.model, 'feature_importances_'):
            if self.feature_selection:
                selected_indices = self.feature_selector.get_support(indices=True)
                selected_features = [X.columns[i] for i in selected_indices]
                for i, importance in enumerate(self.model.feature_importances_):
                    self.feature_importances[selected_features[i]] = importance
            else:
                for i, importance in enumerate(self.model.feature_importances_):
                    self.feature_importances[X.columns[i]] = importance
        
        return {
            'best_params': best_params,
            'best_rmse': best_score
        }
    
    def save_model(self, path):
        """Save the model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_importances': self.feature_importances,
            'model_type': self.model_type,
            'feature_selection': self.feature_selection
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load_model(cls, path):
        """Load a model from disk."""
        model_data = joblib.load(path)
        
        instance = cls(
            model_type=model_data['model_type'],
            feature_selection=model_data['feature_selection']
        )
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_selector = model_data['feature_selector']
        instance.feature_importances = model_data['feature_importances']
        
        return instance
    
    def plot_feature_importances(self, top_n=10):
        """Plot the top N most important features."""
        if not self.feature_importances:
            raise ValueError("No feature importances available")
        
        # Sort features by importance
        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top N features
        top_features = sorted_features[:top_n]
        
        # Create DataFrame for plotting
        df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=df)
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        
        return plt.gcf()


class MaintainabilityModel(QualityModel):
    """Model for predicting maintainability."""
    
    def __init__(self, model_type='random_forest', feature_selection=True):
        super().__init__(model_type, feature_selection)
        
    def get_key_indicators(self):
        """Get key indicators for maintainability."""
        return [
            'complexity',
            'loc',
            'comment_ratio',
            'maintainability_index',
            'circular_dependencies',
            'avg_inheritance_depth',
            'lint_errors'
        ]


class ReliabilityModel(QualityModel):
    """Model for predicting reliability."""
    
    def __init__(self, model_type='gradient_boosting', feature_selection=True):
        super().__init__(model_type, feature_selection)
        
    def get_key_indicators(self):
        """Get key indicators for reliability."""
        return [
            'test_to_code_ratio',
            'test_frameworks_count',
            'lint_errors',
            'avg_time_between_commits',
            'complexity_variance',
            'avg_complexity'
        ]


class ScalabilityModel(QualityModel):
    """Model for predicting scalability."""
    
    def __init__(self, model_type='xgboost', feature_selection=True):
        super().__init__(model_type, feature_selection)
        
    def get_key_indicators(self):
        """Get key indicators for scalability."""
        return [
            'avg_dependencies',
            'max_fan_in',
            'avg_fan_out',
            'circular_dependencies',
            'complexity',
            'avg_inheritance_depth'
        ]


class SecurityModel(QualityModel):
    """Model for predicting security."""
    
    def __init__(self, model_type='random_forest', feature_selection=True):
        super().__init__(model_type, feature_selection)
        
    def get_key_indicators(self):
        """Get key indicators for security."""
        return [
            'dependency_count',
            'has_dependency_management',
            'lint_errors',
            'test_to_code_ratio',
            'avg_file_churn'
        ]


class EfficiencyModel(QualityModel):
    """Model for predicting efficiency."""
    
    def __init__(self, model_type='gradient_boosting', feature_selection=True):
        super().__init__(model_type, feature_selection)
        
    def get_key_indicators(self):
        """Get key indicators for efficiency."""
        return [
            'complexity',
            'halstead_volume',
            'halstead_effort',
            'loc',
            'function_count',
            'avg_function_complexity'
        ]