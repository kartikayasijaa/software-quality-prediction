from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import json
import shutil
import zipfile
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename

from src.data.feature_extractor import CodeFeatureExtractor

app = Flask(__name__, static_folder='frontend/build')
# Enable CORS for all origins, but with specific settings for API endpoints
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration for uploads
UPLOAD_FOLDER = 'data/repos'
ALLOWED_EXTENSIONS = {'zip'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_repository():
    """Upload and extract a repository zip file."""
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submits an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Clear previous repository if it exists
        if os.path.exists(UPLOAD_FOLDER):
            for item in os.listdir(UPLOAD_FOLDER):
                item_path = os.path.join(UPLOAD_FOLDER, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)
            temp_path = temp.name
        
        # Extract the zip file
        repo_name = secure_filename(file.filename.rsplit('.', 1)[0])
        repo_path = os.path.join(UPLOAD_FOLDER, repo_name)
        
        try:
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                zip_ref.extractall(repo_path)
            
            # Remove the temporary file
            os.unlink(temp_path)
            
            return jsonify({
                'success': True,
                'repo_path': repo_path,
                'repo_name': repo_name
            })
            
        except Exception as e:
            return jsonify({'error': f'Error extracting zip file: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed. Please upload a .zip file.'}), 400

@app.route('/api/analyze', methods=['POST'])
def analyze_repository():
    """Analyze a repository and return quality predictions."""
    data = request.json
    repo_path = data.get('repo_path')
    
    if not repo_path or not os.path.exists(repo_path):
        return jsonify({'error': 'Repository path does not exist'}), 400
    
    try:
        # Extract features
        extractor = CodeFeatureExtractor(repo_path)
        features = extractor.extract_all_features()
        
        # Save features to file
        features_df = pd.DataFrame([features])
        os.makedirs('data/processed', exist_ok=True)
        features_file = os.path.join('data/processed', f"{os.path.basename(repo_path)}_features.json")
        features_df.to_json(features_file, orient='records', lines=True)
        
        # Generate synthetic scores based on features
        predictions = generate_synthetic_scores(features)
        
        # Calculate overall score
        overall_score = sum(predictions.values()) / len(predictions)
        
        # Generate key metrics based on feature values
        key_metrics = generate_key_metrics(features, predictions)
        
        # Generate recommendations
        recommendations = generate_recommendations(features, predictions)
        
        # Generate mock history data
        history = generate_history_data(os.path.basename(repo_path), predictions)
        
        # Generate feature importance data
        feature_importance = generate_feature_importance()
        
        # Construct response
        response = {
            'name': os.path.basename(repo_path),
            'overall_score': overall_score,
            'scores': predictions,
            'key_metrics': key_metrics,
            'recommendations': recommendations,
            'history': history,
            'feature_importance': feature_importance
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': f'Error analyzing repository: {str(e)}'}), 500

def generate_synthetic_scores(features):
    """Generate synthetic quality scores based on extracted features."""
    # Default weights for each dimension
    weights = {
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
            'avg_function_complexity': -0.3,
            'loc': -0.2
        }
    }
    
    # Initialize scores with base values
    scores = {
        'maintainability': 0.5,
        'reliability': 0.5,
        'scalability': 0.5,
        'security': 0.5,
        'efficiency': 0.5
    }
    
    # Apply weights to features for each dimension
    for dimension, feature_weights in weights.items():
        for feature, weight in feature_weights.items():
            if feature in features:
                # Normalize the feature value
                value = features[feature]
                if isinstance(value, (int, float)):
                    # Apply the weight
                    if feature == 'has_dependency_management':
                        # Boolean feature
                        scores[dimension] += weight * (1 if value else 0)
                    elif feature in ['complexity', 'lint_errors', 'circular_dependencies']:
                        # Features where lower is better
                        normalized = max(0, 1 - (value / 100))  # Assuming max value of 100
                        scores[dimension] += weight * normalized
                    else:
                        # Features where higher is better
                        normalized = min(1, value)
                        scores[dimension] += weight * normalized
    
    # Ensure scores are between 0 and 1
    for dimension in scores:
        scores[dimension] = max(0, min(1, scores[dimension]))
    
    return scores

def normalize_feature(features, feature_name, inverse=False):
    """Normalize a feature value to be between 0 and 1."""
    feature_ranges = {
        'complexity': (0, 100),
        'comment_ratio': (0, 0.5),
        'maintainability_index': (0, 100),
        'circular_dependencies': (0, 20),
        'test_to_code_ratio': (0, 1),
        'lint_errors': (0, 50)
    }
    
    value = features.get(feature_name, 0)
    min_val, max_val = feature_ranges.get(feature_name, (0, 1))
    
    # Normalize to 0-1 range
    if max_val > min_val:
        normalized = (value - min_val) / (max_val - min_val)
    else:
        normalized = 0
    
    # Clip to 0-1 range
    normalized = max(0, min(1, normalized))
    
    # Inverse if needed (for metrics where lower is better)
    if inverse:
        normalized = 1 - normalized
    
    return normalized

def generate_key_metrics(features, predictions):
    """Generate key metrics for each quality dimension."""
    # Define metrics for each dimension
    metrics_by_dimension = {
        'maintainability': [
            {'name': 'Code Complexity', 'value': normalize_feature(features, 'complexity', inverse=True)},
            {'name': 'Documentation', 'value': normalize_feature(features, 'comment_ratio')},
            {'name': 'Technical Debt', 'value': normalize_feature(features, 'maintainability_index')},
            {'name': 'Code Structure', 'value': 1 - normalize_feature(features, 'circular_dependencies', inverse=True)}
        ],
        'reliability': [
            {'name': 'Test Coverage', 'value': normalize_feature(features, 'test_to_code_ratio')},
            {'name': 'Exception Handling', 'value': 0.7},  # Placeholder
            {'name': 'Error Rate', 'value': 1 - normalize_feature(features, 'lint_errors', inverse=True)},
            {'name': 'Stability', 'value': 0.75}  # Placeholder
        ],
        'scalability': [
            {'name': 'Performance', 'value': 1 - normalize_feature(features, 'complexity', inverse=True)},
            {'name': 'Resource Usage', 'value': 0.65},  # Placeholder
            {'name': 'Architecture', 'value': 1 - normalize_feature(features, 'circular_dependencies', inverse=True)},
            {'name': 'Load Handling', 'value': 0.7}  # Placeholder
        ],
        'security': [
            {'name': 'Vulnerability Score', 'value': 0.85},  # Placeholder
            {'name': 'Secure Practices', 'value': 1 - normalize_feature(features, 'lint_errors', inverse=True)},
            {'name': 'Dependency Security', 'value': features.get('has_dependency_management', 0)},
            {'name': 'Input Validation', 'value': 0.8}  # Placeholder
        ],
        'efficiency': [
            {'name': 'Algorithm Complexity', 'value': 1 - normalize_feature(features, 'complexity', inverse=True)},
            {'name': 'Memory Usage', 'value': 0.7},  # Placeholder
            {'name': 'Execution Time', 'value': 0.75},  # Placeholder
            {'name': 'Resource Efficiency', 'value': 0.72}  # Placeholder
        ]
    }
    
    # Adjust metrics based on prediction scores to ensure consistency
    for dimension, metrics in metrics_by_dimension.items():
        pred_score = predictions[dimension]
        
        # Calculate current average
        avg = np.mean([m['value'] for m in metrics])
        
        # Adjust metrics to roughly match prediction
        if avg > 0:
            scale_factor = pred_score / avg
            for metric in metrics:
                # Scale and keep within 0-1 range
                metric['value'] = min(1.0, max(0.0, metric['value'] * scale_factor))
    
    return metrics_by_dimension

def generate_recommendations(features, predictions):
    """Generate recommendations based on features and predictions."""
    recommendations = []
    
    # Example logic for generating recommendations
    if features.get('circular_dependencies', 0) > 5:
        recommendations.append({
            'category': 'scalability',
            'description': 'Reduce circular dependencies to improve scalability',
            'severity': 'high',
            'impact': 0.15
        })
    
    if features.get('test_to_code_ratio', 0) < 0.5:
        recommendations.append({
            'category': 'reliability',
            'description': 'Increase test coverage for critical components',
            'severity': 'medium',
            'impact': 0.10
        })
    
    if features.get('complexity', 0) > 50:
        recommendations.append({
            'category': 'maintainability',
            'description': 'Refactor complex methods to improve maintainability',
            'severity': 'high',
            'impact': 0.12
        })
    
    if features.get('lint_errors', 0) > 20:
        recommendations.append({
            'category': 'security',
            'description': 'Fix code style and potential security issues detected by linters',
            'severity': 'medium',
            'impact': 0.08
        })
    
    if not features.get('has_dependency_management', False):
        recommendations.append({
            'category': 'security',
            'description': 'Implement dependency management to track and update dependencies',
            'severity': 'critical',
            'impact': 0.20
        })
    
    # Add default recommendations if none were generated
    if not recommendations:
        recommendations = [
            {
                'category': 'efficiency',
                'description': 'Optimize database queries in services',
                'severity': 'medium',
                'impact': 0.08
            },
            {
                'category': 'security',
                'description': 'Review input validation in public interfaces',
                'severity': 'high',
                'impact': 0.15
            }
        ]
    
    return recommendations

def generate_history_data(repo_name, predictions):
    """Generate mock history data for trends."""
    # Generate 5 historical data points
    current_month = datetime.now().replace(day=1)
    history = []
    
    # Fluctuation range for historical data
    fluctuation = 0.15
    
    for i in range(5):
        month = current_month - timedelta(days=30 * (4 - i))
        month_str = month.strftime('%Y-%m')
        
        # Create historical point with randomized fluctuation
        point = {'date': month_str}
        
        for dim, score in predictions.items():
            # Start with lower scores in the past and gradually improve
            base = max(0, score - fluctuation * (4 - i) / 4)
            point[dim] = base
        
        history.append(point)
    
    return history

def generate_feature_importance():
    """Generate feature importance data for visualization."""
    # Feature importance data (would be generated by ML models in production)
    feature_importance = {
        'maintainability': [
            {'name': 'Code Complexity', 'value': 28},
            {'name': 'Documentation', 'value': 22},
            {'name': 'Technical Debt', 'value': 20},
            {'name': 'Circular Dependencies', 'value': 15},
            {'name': 'Code Structure', 'value': 15}
        ],
        'reliability': [
            {'name': 'Test Coverage', 'value': 30},
            {'name': 'Exception Handling', 'value': 25},
            {'name': 'Error Handling', 'value': 20},
            {'name': 'Code Stability', 'value': 15},
            {'name': 'Test Quality', 'value': 10}
        ],
        'scalability': [
            {'name': 'Architecture', 'value': 35},
            {'name': 'Dependency Structure', 'value': 25},
            {'name': 'Resource Usage', 'value': 20},
            {'name': 'Performance', 'value': 20}
        ],
        'security': [
            {'name': 'Dependency Security', 'value': 30},
            {'name': 'Input Validation', 'value': 25},
            {'name': 'Authentication', 'value': 20},
            {'name': 'Data Protection', 'value': 15},
            {'name': 'Error Exposure', 'value': 10}
        ],
        'efficiency': [
            {'name': 'Algorithm Complexity', 'value': 35},
            {'name': 'Memory Usage', 'value': 25},
            {'name': 'Execution Time', 'value': 25},
            {'name': 'Resource Efficiency', 'value': 15}
        ]
    }
    
    return feature_importance

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve the frontend application."""
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')