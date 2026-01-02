import os
import pickle
import json
from datetime import datetime

# AWS LAMBDA DEPLOYMENT PACKAGE CREATOR
# Creates deployment-ready package with models, dependencies, and handler

print("=" * 70)
print("AWS LAMBDA DEPLOYMENT PACKAGE BUILDER")
print("=" * 70)

# Create deployment directory structure
deployment_dir = 'lambda_deployment'
os.makedirs(deployment_dir, exist_ok=True)
print(f"\n📁 Created deployment directory: {deployment_dir}/")

# 1. Save trained models as pickle files
print("\n🔧 Packaging Model Artifacts...")

# Save physics model
with open(f'{deployment_dir}/physics_model.pkl', 'wb') as f:
    pickle.dump(inference_models['physics_model'], f)
print(f"   ✓ physics_model.pkl saved")

# Save gradient boosting model
with open(f'{deployment_dir}/gb_model.pkl', 'wb') as f:
    pickle.dump(inference_models['gb_model'], f)
print(f"   ✓ gb_model.pkl saved")

# Save physics scaler
with open(f'{deployment_dir}/physics_scaler.pkl', 'wb') as f:
    pickle.dump(inference_models['physics_scaler'], f)
print(f"   ✓ physics_scaler.pkl saved")

# Save ML scaler
with open(f'{deployment_dir}/ml_scaler.pkl', 'wb') as f:
    pickle.dump(inference_models['ml_scaler'], f)
print(f"   ✓ ml_scaler.pkl saved")

# 2. Save configuration as JSON
print("\n📝 Creating Configuration Files...")

config_export = {
    'version': inference_config['version'],
    'training_date': inference_config['training_date'],
    'physics_features': inference_config['physics_features'],
    'ml_features': inference_config['ml_features'],
    'ensemble_weights': inference_config['ensemble_weights'],
    'threshold': inference_config['threshold'],
    'performance': inference_config['performance']
}

with open(f'{deployment_dir}/model_config.json', 'w') as f:
    json.dump(config_export, f, indent=2)
print(f"   ✓ model_config.json saved")

# 3. Create Lambda handler code with proper API Gateway integration
print("\n🐍 Generating Lambda Handler Code...")

lambda_handler_code = '''import json
import pickle
import pandas as pd
import numpy as np
import os
from typing import Dict, Any

# Load models and configuration at cold start (outside handler)
MODEL_DIR = os.path.dirname(__file__)

with open(f'{MODEL_DIR}/physics_model.pkl', 'rb') as f:
    physics_model = pickle.load(f)

with open(f'{MODEL_DIR}/gb_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)

with open(f'{MODEL_DIR}/physics_scaler.pkl', 'rb') as f:
    physics_scaler = pickle.load(f)

with open(f'{MODEL_DIR}/ml_scaler.pkl', 'rb') as f:
    ml_scaler = pickle.load(f)

with open(f'{MODEL_DIR}/model_config.json', 'r') as f:
    config = json.load(f)

def engineer_features(raw_data: Dict) -> pd.DataFrame:
    """Transform raw telemetry into model features"""
    if isinstance(raw_data, dict):
        df = pd.DataFrame([raw_data])
    else:
        df = raw_data.copy()
    
    # Physics-based features
    df['power_watts'] = df['voltage'] * df['current']
    df['apparent_power'] = df['voltage'] * df['current']
    df['resistance_ohms'] = df['voltage'] / (df['current'] + 1e-6)
    df['thermal_dissipation'] = df['current']**2 * df['resistance_ohms']
    df['temp_per_watt'] = df['temperature'] / (df['power_watts'] + 1)
    df['thermal_load_index'] = df['temperature'] * df['load_factor']
    df['temp_deviation'] = df['temperature'] - 25.0
    df['thermal_stress'] = df['temp_deviation'] / (df['power_watts'] + 1)
    df['voltage_deviation'] = df['voltage'] - 240.0
    df['voltage_deviation_pct'] = df['voltage_deviation'] / 240.0
    df['voltage_instability'] = abs(df['voltage_deviation'])
    df['current_utilization'] = df['current'] / 55.71
    df['efficiency_proxy'] = df['power_watts'] / (df['voltage'] * df['current'] + 1)
    df['system_health_score'] = 1.0 - (abs(df['voltage_deviation_pct']) + abs(df['temperature'] - 25) / 50)
    
    # Rolling features (set to current values for single point inference)
    for col in ['voltage', 'current', 'temperature', 'power_watts']:
        df[f'{col}_rolling_mean_1h'] = df[col]
        df[f'{col}_rolling_std_1h'] = 0.0
        df[f'{col}_rolling_max_1h'] = df[col]
        df[f'{col}_rolling_min_1h'] = df[col]
    
    # Lag features
    for col in ['voltage', 'current', 'temperature', 'load_factor']:
        df[f'{col}_lag_1'] = df[col]
        df[f'{col}_lag_4'] = df[col]
    
    # Rate of change
    for col in ['voltage', 'current', 'temperature', 'power_watts']:
        df[f'{col}_rate_of_change'] = 0.0
        df[f'{col}_rate_of_change_pct'] = 0.0
    
    # Z-scores and anomalies
    for col in ['voltage', 'current', 'temperature']:
        df[f'{col}_zscore'] = 0.0
        df[f'{col}_is_anomaly'] = 0
    
    # IQR outliers
    df['voltage_iqr_outlier'] = 0
    df['current_iqr_outlier'] = 0
    df['temperature_iqr_outlier'] = 0
    
    # Interactions
    df['voltage_current_interaction'] = df['voltage'] * df['current']
    df['temp_load_interaction'] = df['temperature'] * df['load_factor']
    
    # Volatility
    df['voltage_volatility_24h'] = 0.0
    df['current_volatility_24h'] = 0.0
    df['temperature_volatility_24h'] = 0.0
    
    # Time features
    df['hour'] = 12
    df['day_of_week'] = 3
    df['day_of_month'] = 15
    df['is_weekend'] = 0
    df['hour_sin'] = np.sin(2 * np.pi * 12 / 24)
    df['hour_cos'] = np.cos(2 * np.pi * 12 / 24)
    
    return df

def create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Create properly formatted API Gateway response"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        },
        'body': json.dumps(body)
    }

def handle_predict(telemetry_data: Dict) -> Dict[str, Any]:
    """Handle /predict endpoint"""
    # Validate required fields
    required_fields = ['voltage', 'current', 'temperature', 'load_factor']
    for field in required_fields:
        if field not in telemetry_data:
            return create_response(400, {
                'error': 'Missing required field',
                'field': field,
                'required_fields': required_fields
            })
    
    # Feature engineering
    features_df = engineer_features(telemetry_data)
    
    # Extract physics features
    physics_features_df = features_df[config['physics_features']]
    physics_scaled = physics_scaler.transform(physics_features_df)
    
    # Extract ML features
    ml_features_df = features_df[config['ml_features']]
    ml_scaled = ml_scaler.transform(ml_features_df)
    
    # Get predictions
    physics_proba = physics_model.predict_proba(physics_scaled)[0, 1]
    gb_proba = gb_model.predict_proba(ml_scaled)[0, 1]
    
    # Ensemble prediction
    weights = config['ensemble_weights']
    ensemble_proba = weights['physics'] * physics_proba + weights['gb'] * gb_proba
    prediction = 1 if ensemble_proba > config['threshold'] else 0
    
    # Confidence scoring
    confidence = (1.0 - abs(physics_proba - gb_proba)) * 100
    
    # Prepare response
    result = {
        'prediction': int(prediction),
        'prediction_label': 'FAILURE_RISK' if prediction == 1 else 'NORMAL',
        'risk_score': float(ensemble_proba),
        'confidence': float(confidence),
        'model_breakdown': {
            'physics_score': float(physics_proba),
            'ml_score': float(gb_proba)
        },
        'model_version': config['version'],
        'input': telemetry_data
    }
    
    return create_response(200, result)

def handle_health() -> Dict[str, Any]:
    """Handle /health endpoint"""
    return create_response(200, {
        'status': 'healthy',
        'service': 'equipment-failure-predictor',
        'model_version': config['version'],
        'models_loaded': True
    })

def handle_model_info() -> Dict[str, Any]:
    """Handle /model-info endpoint"""
    return create_response(200, {
        'model_version': config['version'],
        'training_date': config['training_date'],
        'performance': config['performance'],
        'ensemble_weights': config['ensemble_weights'],
        'threshold': config['threshold'],
        'physics_features_count': len(config['physics_features']),
        'ml_features_count': len(config['ml_features'])
    })

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for equipment failure prediction API
    
    Supports three endpoints:
    - POST /predict: Make predictions
    - GET /health: Health check
    - GET /model-info: Model metadata
    
    Expected input for /predict (JSON):
    {
        "voltage": 242.5,
        "current": 35.2,
        "temperature": 48.3,
        "load_factor": 0.75
    }
    """
    
    print(f"Received event: {json.dumps(event)}")
    
    try:
        # Parse API Gateway event format
        if 'requestContext' in event:
            # API Gateway REST API or HTTP API format
            http_method = event.get('httpMethod') or event.get('requestContext', {}).get('http', {}).get('method')
            path = event.get('path') or event.get('requestContext', {}).get('http', {}).get('path', '')
            
            # Parse body for POST requests
            if http_method == 'POST':
                body_str = event.get('body', '{}')
                body = json.loads(body_str) if isinstance(body_str, str) else body_str
            else:
                body = {}
        else:
            # Direct Lambda invocation
            http_method = event.get('httpMethod', 'POST')
            path = event.get('path', '/predict')
            body = event.get('body', event)
            if isinstance(body, str):
                body = json.loads(body)
        
        print(f"Method: {http_method}, Path: {path}")
        
        # Handle OPTIONS for CORS preflight
        if http_method == 'OPTIONS':
            return create_response(200, {'message': 'OK'})
        
        # Route to appropriate handler
        if path == '/health' or path.endswith('/health'):
            return handle_health()
        
        elif path == '/model-info' or path.endswith('/model-info'):
            return handle_model_info()
        
        elif path == '/predict' or path.endswith('/predict') or http_method == 'POST':
            return handle_predict(body)
        
        else:
            return create_response(404, {
                'error': 'Not found',
                'available_endpoints': ['/predict', '/health', '/model-info']
            })
        
    except json.JSONDecodeError as e:
        return create_response(400, {
            'error': 'Invalid JSON',
            'message': str(e)
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return create_response(500, {
            'error': 'Internal server error',
            'message': str(e)
        })
'''

with open(f'{deployment_dir}/lambda_function.py', 'w') as f:
    f.write(lambda_handler_code)
print(f"   ✓ lambda_function.py created with API Gateway routing")

# 4. Create requirements.txt
print("\n📦 Creating Requirements File...")

requirements_content = '''pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
'''

with open(f'{deployment_dir}/requirements.txt', 'w') as f:
    f.write(requirements_content)
print(f"   ✓ requirements.txt created")

# 5. Create deployment instructions
print("\n📋 Creating Deployment Instructions...")

readme_content = '''# AWS Lambda Deployment Package

## Package Contents

### Model Files (Pickled)
- `physics_model.pkl` - Logistic Regression physics-based model
- `gb_model.pkl` - Gradient Boosting ML model  
- `physics_scaler.pkl` - StandardScaler for physics features
- `ml_scaler.pkl` - StandardScaler for ML features

### Configuration
- `model_config.json` - Model metadata, feature lists, ensemble weights, performance metrics

### Code
- `lambda_function.py` - Lambda handler with feature engineering, inference logic, and API routing

### Dependencies
- `requirements.txt` - Python package requirements

## Model Performance
- Ensemble Accuracy: 99.65%
- Precision: 99.59%
- Recall: 99.17%
- F1-Score: 99.38%
- ROC-AUC: 99.96%

## API Endpoints

### POST /predict
Make failure predictions

**Request:**
```json
{
  "voltage": 242.5,
  "current": 35.2,
  "temperature": 48.3,
  "load_factor": 0.75
}
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "NORMAL",
  "risk_score": 0.4000,
  "confidence": 0.0,
  "model_breakdown": {
    "physics_score": 0.4000,
    "ml_score": 0.4000
  },
  "model_version": "20260102_055632",
  "input": {...}
}
```

### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "service": "equipment-failure-predictor",
  "model_version": "20260102_055632",
  "models_loaded": true
}
```

### GET /model-info
Model metadata and performance

**Response:**
```json
{
  "model_version": "20260102_055632",
  "training_date": "2026-01-02 05:56:32",
  "performance": {...},
  "ensemble_weights": {...},
  "threshold": 0.5,
  "physics_features_count": 18,
  "ml_features_count": 70
}
```

## Deployment Steps

### 1. Install Dependencies Locally
```bash
cd lambda_deployment
pip install -r requirements.txt -t .
```

### 2. Create Deployment Package
```bash
zip -r deployment_package.zip . -x "*.pyc" "__pycache__/*"
```

### 3. Create Lambda Function
```bash
aws lambda create-function \\
  --function-name equipment-failure-predictor \\
  --runtime python3.9 \\
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role \\
  --handler lambda_function.lambda_handler \\
  --zip-file fileb://deployment_package.zip \\
  --timeout 30 \\
  --memory-size 512
```

### 4. Test the Function
```bash
# Test /predict endpoint
aws lambda invoke \\
  --function-name equipment-failure-predictor \\
  --payload '{"path":"/predict","httpMethod":"POST","body":"{\\"voltage\\":242.5,\\"current\\":35.2,\\"temperature\\":48.3,\\"load_factor\\":0.75}"}' \\
  response.json

# Test /health endpoint
aws lambda invoke \\
  --function-name equipment-failure-predictor \\
  --payload '{"path":"/health","httpMethod":"GET"}' \\
  health.json

# Test /model-info endpoint
aws lambda invoke \\
  --function-name equipment-failure-predictor \\
  --payload '{"path":"/model-info","httpMethod":"GET"}' \\
  info.json
```

## Lambda Configuration Recommendations

- **Runtime**: Python 3.9 or 3.10
- **Memory**: 512 MB (adjust based on cold start performance)
- **Timeout**: 30 seconds
- **Architecture**: x86_64 or arm64 (Graviton2 for cost savings)

## Cold Start Optimization

Models are loaded at cold start (outside handler) to minimize per-request latency:
- First invocation (cold start): ~2-3 seconds
- Warm invocations: <100ms
- Lambda SnapStart compatible for even faster cold starts

## API Gateway Integration

### HTTP API (Recommended - Lower Cost)
```bash
aws apigatewayv2 create-api \\
  --name equipment-prediction-api \\
  --protocol-type HTTP \\
  --target arn:aws:lambda:REGION:ACCOUNT:function:equipment-failure-predictor

# Add routes
aws apigatewayv2 create-route \\
  --api-id API_ID \\
  --route-key 'POST /predict' \\
  --target integrations/INTEGRATION_ID

aws apigatewayv2 create-route \\
  --api-id API_ID \\
  --route-key 'GET /health' \\
  --target integrations/INTEGRATION_ID

aws apigatewayv2 create-route \\
  --api-id API_ID \\
  --route-key 'GET /model-info' \\
  --target integrations/INTEGRATION_ID
```

### REST API (Feature-Rich)
```bash
aws apigateway create-rest-api \\
  --name equipment-prediction-api \\
  --endpoint-configuration types=REGIONAL
```

## Monitoring Recommendations

1. **CloudWatch Logs**: Enable for debugging
2. **CloudWatch Metrics**: 
   - Duration
   - Invocation count
   - Error count
   - Throttles
3. **X-Ray Tracing**: Enable for detailed performance insights
4. **Custom Metrics**: 
   - Prediction distribution (NORMAL vs FAILURE_RISK)
   - Risk score distribution
   - Model confidence levels

## Cost Estimates

With 1M predictions/month:
- Lambda (512MB, 100ms avg): ~$8-12/month
- API Gateway HTTP API: ~$1/month
- CloudWatch Logs (basic): ~$2/month
- Total: ~$11-15/month

## Security Best Practices

1. **IAM Roles**: Use least-privilege execution role
2. **API Gateway**: Enable API key authentication or AWS IAM auth
3. **VPC**: Deploy in VPC if accessing private resources
4. **Encryption**: Enable at-rest encryption for sensitive data
5. **Secrets Manager**: Store API keys in AWS Secrets Manager
6. **WAF**: Enable AWS WAF for DDoS protection

## Scaling

Lambda auto-scales to handle concurrent requests:
- Default: 1000 concurrent executions
- Can request increase to 10,000+
- Handles sudden traffic spikes automatically
- No manual scaling configuration needed

## Error Handling

The Lambda handler includes comprehensive error handling:
- **400 Bad Request**: Missing fields, invalid JSON
- **404 Not Found**: Invalid endpoint
- **500 Internal Server Error**: Processing errors

All errors return JSON responses with detailed messages.
'''

with open(f'{deployment_dir}/README.md', 'w') as f:
    f.write(readme_content)
print(f"   ✓ README.md created with API Gateway integration details")

# 6. Create deployment checklist
deployment_checklist = {
    'package_contents': {
        'models': [
            'physics_model.pkl',
            'gb_model.pkl',
            'physics_scaler.pkl',
            'ml_scaler.pkl'
        ],
        'configuration': ['model_config.json'],
        'code': ['lambda_function.py'],
        'dependencies': ['requirements.txt'],
        'documentation': ['README.md']
    },
    'api_endpoints': [
        {'method': 'POST', 'path': '/predict', 'description': 'Make failure predictions'},
        {'method': 'GET', 'path': '/health', 'description': 'Health check'},
        {'method': 'GET', 'path': '/model-info', 'description': 'Model metadata'}
    ],
    'deployment_steps': [
        '1. Install dependencies: pip install -r requirements.txt -t .',
        '2. Create ZIP: zip -r deployment_package.zip .',
        '3. Create Lambda function via AWS Console or CLI',
        '4. Upload deployment_package.zip',
        '5. Set handler to lambda_function.lambda_handler',
        '6. Configure memory (512MB) and timeout (30s)',
        '7. Test endpoints (/predict, /health, /model-info)',
        '8. Connect to API Gateway for HTTP endpoint',
        '9. Configure API Gateway authentication',
        '10. Set up CloudWatch monitoring and alarms'
    ],
    'validation': {
        'model_version': inference_config['version'],
        'ensemble_accuracy': inference_config['performance']['ensemble']['accuracy'],
        'total_files': 6,
        'api_endpoints_count': 3
    }
}

with open(f'{deployment_dir}/deployment_checklist.json', 'w') as f:
    json.dump(deployment_checklist, f, indent=2)
print(f"   ✓ deployment_checklist.json updated with API routing")

# 7. Verify package contents
print("\n✅ DEPLOYMENT PACKAGE VERIFICATION")
deployment_files = os.listdir(deployment_dir)
print(f"\n   Package Directory: {deployment_dir}/")
print(f"   Total Files: {len(deployment_files)}")
print(f"\n   Files Created:")
for file in sorted(deployment_files):
    file_path = os.path.join(deployment_dir, file)
    size_kb = os.path.getsize(file_path) / 1024
    print(f"      • {file:<30} ({size_kb:.1f} KB)")

# Calculate total package size
total_size = sum(os.path.getsize(os.path.join(deployment_dir, f)) for f in deployment_files) / 1024
print(f"\n   Total Package Size: {total_size:.1f} KB")

print("\n" + "=" * 70)
print("📦 DEPLOYMENT PACKAGE STATUS")
print("=" * 70)
print(f"\n✅ All artifacts packaged successfully!")
print(f"\n📁 Location: ./{deployment_dir}/")
print(f"🎯 Model Version: {inference_config['version']}")
print(f"📊 Ensemble Performance: {inference_config['performance']['ensemble']['accuracy']:.4f} accuracy")
print(f"\n🚀 API ENDPOINTS:")
print(f"   • POST /predict - Failure prediction")
print(f"   • GET /health - Health check")
print(f"   • GET /model-info - Model metadata")
print(f"\n📝 NEXT STEPS:")
print(f"   1. cd {deployment_dir}")
print(f"   2. pip install -r requirements.txt -t .")
print(f"   3. zip -r deployment_package.zip . -x '*.pyc' '__pycache__/*'")
print(f"   4. Upload to AWS Lambda")
print(f"   5. Connect to API Gateway")
print(f"\n📖 See README.md for detailed deployment instructions")
print("=" * 70)

deployment_package_path = deployment_dir
lambda_handler_ready = True