from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables to store model and encoders
model = None
feature_names = None

def load_or_train_model():
    """Load existing model or train a new one"""
    global model, feature_names
    
    model_path = 'churn_model.pkl'
    
    if os.path.exists(model_path):
        # Load existing model
        print("Loading existing model...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            feature_names = model_data['feature_names']
        print("Model loaded successfully!")
    else:
        # Train new model if no existing model found
        print("Training new model...")
        train_model()
        print("Model trained successfully!")

def train_model():
    """Train the churn prediction model"""
    global model, feature_names
    
    # Load and prepare data (replace with your actual data path)
    try:
        # Update this path to your actual CSV file location
        df = pd.read_csv('Churn_Modelling.csv')
        
        # Data preprocessing (matching your notebook)
        df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        df['Gender'] = le_gender.fit_transform(df['Gender'])
        
        df['Geography'] = df['Geography'].str.strip().str.title()
        df['Geography'] = df['Geography'].map({'France': 0, 'Germany': 1, 'Spain': 2})
        
        # Prepare features and target
        X = df.drop('Exited', axis=1)
        y = df['Exited']
        feature_names = X.columns.tolist()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
        
        # Apply SMOTE for balanced training
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
        
        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_bal, y_train_bal)
        
        # Save the model
        model_data = {
            'model': model,
            'feature_names': feature_names
        }
        with open('churn_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Print model performance
        train_score = model.score(X_train_bal, y_train_bal)
        test_score = model.score(X_test, y_test)
        print(f"Model trained - Train Score: {train_score:.3f}, Test Score: {test_score:.3f}")
        
    except FileNotFoundError:
        print("Warning: Churn_Modelling.csv not found. Creating a mock model for demo purposes.")
        create_mock_model()

def create_mock_model():
    """Create a mock model for demonstration when data file is not available"""
    global model, feature_names
    
    # Create mock data similar to your dataset
    np.random.seed(42)
    n_samples = 1000
    
    feature_names = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    # Generate synthetic data
    X = pd.DataFrame({
        'CreditScore': np.random.randint(300, 850, n_samples),
        'Geography': np.random.randint(0, 3, n_samples),
        'Gender': np.random.randint(0, 2, n_samples),
        'Age': np.random.randint(18, 80, n_samples),
        'Tenure': np.random.randint(0, 10, n_samples),
        'Balance': np.random.uniform(0, 200000, n_samples),
        'NumOfProducts': np.random.randint(1, 5, n_samples),
        'HasCrCard': np.random.randint(0, 2, n_samples),
        'IsActiveMember': np.random.randint(0, 2, n_samples),
        'EstimatedSalary': np.random.uniform(20000, 150000, n_samples)
    })
    
    # Create synthetic target based on some logic
    y = ((X['Age'] > 50) | (X['Balance'] == 0) | (X['NumOfProducts'] > 2) | 
         (X['IsActiveMember'] == 0)).astype(int)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the mock model
    model_data = {
        'model': model,
        'feature_names': feature_names
    }
    with open('churn_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

def get_recommendations(prediction, probability, customer_data):
    """Generate personalized recommendations based on prediction"""
    recommendations = []
    
    if prediction == 1:  # High churn risk
        if customer_data.get('is_active_member', 0) == 0:
            recommendations.append("Engage customer with personalized offers to increase activity")
        
        if customer_data.get('num_products', 1) == 1:
            recommendations.append("Offer additional products or services to increase engagement")
        
        if customer_data.get('age', 30) > 50:
            recommendations.append("Provide senior-friendly services and support")
        
        if customer_data.get('balance', 0) == 0:
            recommendations.append("Encourage account usage with balance incentives")
        
        recommendations.append("Assign dedicated account manager for personalized service")
        recommendations.append("Offer loyalty rewards and retention programs")
        
    else:  # Low churn risk
        recommendations.append("Maintain current service quality")
        recommendations.append("Consider upselling additional products")
        
        if customer_data.get('is_active_member', 0) == 1:
            recommendations.append("Reward loyalty with exclusive benefits")
        
        recommendations.append("Monitor account activity for any changes")
    
    return recommendations

@app.route('/predict', methods=['POST'])
def predict_churn():
    """API endpoint for churn prediction"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['credit_score', 'geography', 'gender', 'age', 'tenure',
                          'balance', 'num_products', 'has_cr_card', 'is_active_member', 'estimated_salary']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'CreditScore': [data['credit_score']],
            'Geography': [data['geography']],
            'Gender': [data['gender']],
            'Age': [data['age']],
            'Tenure': [data['tenure']],
            'Balance': [data['balance']],
            'NumOfProducts': [data['num_products']],
            'HasCrCard': [data['has_cr_card']],
            'IsActiveMember': [data['is_active_member']],
            'EstimatedSalary': [data['estimated_salary']]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        churn_probability = probability[1]  # Probability of churn (class 1)
        
        # Get feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for feature, importance in zip(feature_names, model.feature_importances_):
                feature_importance[feature.lower()] = float(importance)
        
        # Generate recommendations
        recommendations = get_recommendations(prediction, churn_probability, data)
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'probability': float(churn_probability),
            'recommendations': recommendations,
            'feature_importance': feature_importance,
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'message': 'Churn Prediction API is running'
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'message': 'Customer Churn Prediction API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Make churn predictions',
            '/health': 'GET - Health check'
        }
    })

if __name__ == '__main__':
    print("Starting Customer Churn Prediction API...")
    print("Loading/Training model...")
    load_or_train_model()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)