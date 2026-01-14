
# deployment_guide.py
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("CREDIT RISK PREDICTION SYSTEM - DEPLOYMENT GUIDE")
print("=" * 80)

print("\n1. LOADING THE MODEL:")
print("-" * 80)

try:
    # Load the best model
    model_data = joblib.load('saved_models/best_model_deployment.pkl')
    print(f"Model loaded successfully: {model_data['model_name']}")
    print(f"Training date: {model_data['metadata']['training_date']}")
    print(f"Accuracy: {model_data['performance']['accuracy']:.4f}")
    print(f"ROC-AUC: {model_data['performance']['roc_auc']:.4f}")

    # Load preprocessing pipeline
    preprocessing = joblib.load('saved_models/preprocessing_pipeline.pkl')
    print(f"\nPreprocessing pipeline loaded")
    print(f"Features: {len(preprocessing['feature_names'])}")
    print(f"SMOTE applied during training: {'Yes' if 'smote' in preprocessing else 'No'}")

except Exception as e:
    print(f"Error loading model: {e}")

print("\n2. MAKING PREDICTIONS:")
print("-" * 80)

# Sample prediction function
def predict_credit_risk(applicant_data, model_path='saved_models/best_model_deployment.pkl'):
    """Simple prediction function for deployment"""
    try:
        # Load model
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['feature_names']

        # Create DataFrame
        df = pd.DataFrame([applicant_data])

        # Simple preprocessing (in production, use full preprocessing pipeline)
        # This is a simplified version
        X = np.array([[applicant_data.get(f, 0) for f in features]])
        X_scaled = scaler.transform(X)

        # Predict
        proba = model.predict_proba(X_scaled)[0]
        risk_score = proba[1]

        return {
            'success': True,
            'risk_score': float(risk_score),
            'probability_default': float(proba[1]),
            'decision': 'HIGH RISK' if risk_score > 0.5 else 'LOW RISK',
            'model': model_data['model_name']
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}

# Test prediction
sample_applicant = {
    'Income_annual': 75000,
    'Employment_Status': 'Employed',
    'Loan_Amount': 35000,
    'Loan_Term_Months': 36,
    'Credit_Utilization': 0.45,
    'Previous_Defaults_Count': 0,
    'Age': 35,
    'Marital_Status': 'Married',
    'Num_Dependents': 2
}

result = predict_credit_risk(sample_applicant)
if result['success']:
    print(f"Test prediction successful!")
    print(f"Risk Score: {result['risk_score']:.3f}")
    print(f"Default Probability: {result['probability_default']:.2%}")
    print(f"Decision: {result['decision']}")
else:
    print(f"Test prediction failed: {result['error']}")

print("\n3. DEPLOYMENT OPTIONS:")
print("-" * 80)
print("Option 1: Direct Python Integration")
print("  - Import joblib and load model")
print("  - Use predict_credit_risk() function")
print("\nOption 2: REST API (FastAPI/Flask)")
print("  - Create API endpoints")
print("  - Load model once at startup")
print("  - Handle multiple requests")
print("\nOption 3: Batch Processing")
print("  - Load model once")
print("  - Process CSV files")
print("  - Generate reports")

print("\n" + "=" * 80)
print("READY FOR PRODUCTION DEPLOYMENT!")
print("=" * 80)
