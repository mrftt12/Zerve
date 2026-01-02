import boto3
import json
import os
import zipfile
from pathlib import Path
import time

# AWS deployment automation for Equipment Failure Prediction API
# Creates Lambda function, API Gateway, IAM roles, and CloudWatch logging

# Configuration
FUNCTION_NAME = 'equipment-failure-prediction-api'
API_NAME = 'EquipmentPredictionAPI'
REGION = 'us-east-1'
LAMBDA_RUNTIME = 'python3.9'
LAMBDA_HANDLER = 'lambda_function.lambda_handler'
LAMBDA_MEMORY = 512
LAMBDA_TIMEOUT = 30

# Paths
DEPLOYMENT_DIR = 'lambda_deployment'
ZIP_FILE = 'lambda_deployment_package.zip'

# ============================================================================
# 1. CREATE LAMBDA DEPLOYMENT PACKAGE
# ============================================================================
print("=" * 80)
print("STEP 1: Creating Lambda deployment package")
print("=" * 80)

deployment_path = Path(DEPLOYMENT_DIR)
if not deployment_path.exists():
    print(f"❌ ERROR: {DEPLOYMENT_DIR} not found. Run the deployment package builder first.")
    raise FileNotFoundError(f"{DEPLOYMENT_DIR} directory not found")

# Create zip file
zip_path = Path(ZIP_FILE)
if zip_path.exists():
    os.remove(ZIP_FILE)
    print(f"✓ Removed existing {ZIP_FILE}")

with zipfile.ZipFile(ZIP_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path in deployment_path.glob('*'):
        if file_path.is_file():
            zipf.write(file_path, file_path.name)
            print(f"  Added: {file_path.name}")

zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
print(f"\n✓ Deployment package created: {ZIP_FILE} ({zip_size_mb:.2f} MB)\n")

# ============================================================================
# 2. GENERATE IAM ROLE POLICY
# ============================================================================
print("=" * 80)
print("STEP 2: IAM Role Configuration")
print("=" * 80)

iam_role_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}

lambda_execution_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}

print("IAM Role Trust Policy:")
print(json.dumps(iam_role_policy, indent=2))
print("\nLambda Execution Policy:")
print(json.dumps(lambda_execution_policy, indent=2))
print()

# ============================================================================
# 3. GENERATE AWS CLI COMMANDS
# ============================================================================
print("=" * 80)
print("STEP 3: AWS CLI Deployment Commands")
print("=" * 80)

cli_commands = f"""
# ============================================================================
# AWS CLI DEPLOYMENT SCRIPT - Equipment Failure Prediction API
# ============================================================================

# Prerequisites:
# 1. Install AWS CLI: https://aws.amazon.com/cli/
# 2. Configure credentials: aws configure
# 3. Set your AWS account ID below

export AWS_ACCOUNT_ID="YOUR_AWS_ACCOUNT_ID"
export AWS_REGION="{REGION}"
export FUNCTION_NAME="{FUNCTION_NAME}"
export API_NAME="{API_NAME}"
export ROLE_NAME="lambda-equipment-prediction-role"

# ============================================================================
# STEP 1: Create IAM Role for Lambda
# ============================================================================
echo "Creating IAM role..."

# Create trust policy file
cat > trust-policy.json << 'EOF'
{json.dumps(iam_role_policy, indent=2)}
EOF

# Create execution policy file
cat > execution-policy.json << 'EOF'
{json.dumps(lambda_execution_policy, indent=2)}
EOF

# Create IAM role
aws iam create-role \\
  --role-name $ROLE_NAME \\
  --assume-role-policy-document file://trust-policy.json \\
  --region $AWS_REGION

# Attach execution policy
aws iam put-role-policy \\
  --role-name $ROLE_NAME \\
  --policy-name LambdaExecutionPolicy \\
  --policy-document file://execution-policy.json

# Wait for role to propagate
echo "Waiting for IAM role to propagate..."
sleep 10

export ROLE_ARN="arn:aws:iam::$AWS_ACCOUNT_ID:role/$ROLE_NAME"
echo "Role ARN: $ROLE_ARN"

# ============================================================================
# STEP 2: Create Lambda Function
# ============================================================================
echo "Creating Lambda function..."

aws lambda create-function \\
  --function-name $FUNCTION_NAME \\
  --runtime {LAMBDA_RUNTIME} \\
  --role $ROLE_ARN \\
  --handler {LAMBDA_HANDLER} \\
  --zip-file fileb://{ZIP_FILE} \\
  --timeout {LAMBDA_TIMEOUT} \\
  --memory-size {LAMBDA_MEMORY} \\
  --region $AWS_REGION \\
  --environment Variables='{{MODEL_VERSION=v1.0,ENABLE_MONITORING=true}}'

echo "Lambda function created successfully!"

# ============================================================================
# STEP 3: Create API Gateway REST API
# ============================================================================
echo "Creating API Gateway..."

# Create REST API
export API_ID=$(aws apigateway create-rest-api \\
  --name $API_NAME \\
  --description "Equipment Failure Prediction REST API" \\
  --region $AWS_REGION \\
  --query 'id' \\
  --output text)

echo "API ID: $API_ID"

# Get root resource ID
export ROOT_RESOURCE_ID=$(aws apigateway get-resources \\
  --rest-api-id $API_ID \\
  --region $AWS_REGION \\
  --query 'items[0].id' \\
  --output text)

# Create /predict resource
export PREDICT_RESOURCE_ID=$(aws apigateway create-resource \\
  --rest-api-id $API_ID \\
  --parent-id $ROOT_RESOURCE_ID \\
  --path-part predict \\
  --region $AWS_REGION \\
  --query 'id' \\
  --output text)

echo "Predict Resource ID: $PREDICT_RESOURCE_ID"

# Create POST method
aws apigateway put-method \\
  --rest-api-id $API_ID \\
  --resource-id $PREDICT_RESOURCE_ID \\
  --http-method POST \\
  --authorization-type NONE \\
  --region $AWS_REGION

# Configure Lambda integration
aws apigateway put-integration \\
  --rest-api-id $API_ID \\
  --resource-id $PREDICT_RESOURCE_ID \\
  --http-method POST \\
  --type AWS_PROXY \\
  --integration-http-method POST \\
  --uri arn:aws:apigateway:$AWS_REGION:lambda:path/2015-03-31/functions/arn:aws:lambda:$AWS_REGION:$AWS_ACCOUNT_ID:function:$FUNCTION_NAME/invocations \\
  --region $AWS_REGION

# Grant API Gateway permission to invoke Lambda
aws lambda add-permission \\
  --function-name $FUNCTION_NAME \\
  --statement-id apigateway-access \\
  --action lambda:InvokeFunction \\
  --principal apigateway.amazonaws.com \\
  --source-arn "arn:aws:execute-api:$AWS_REGION:$AWS_ACCOUNT_ID:$API_ID/*/*" \\
  --region $AWS_REGION

# ============================================================================
# STEP 4: Deploy API
# ============================================================================
echo "Deploying API..."

aws apigateway create-deployment \\
  --rest-api-id $API_ID \\
  --stage-name prod \\
  --stage-description "Production deployment" \\
  --description "Initial deployment" \\
  --region $AWS_REGION

# ============================================================================
# STEP 5: Get Endpoint URL
# ============================================================================
export API_ENDPOINT="https://$API_ID.execute-api.$AWS_REGION.amazonaws.com/prod/predict"

echo ""
echo "=" * 80
echo "✓ DEPLOYMENT COMPLETE!"
echo "=" * 80
echo ""
echo "API Endpoint URL: $API_ENDPOINT"
echo ""
echo "Save this URL for testing and client integration."
echo ""

# ============================================================================
# STEP 6: Configure CloudWatch Logging
# ============================================================================
echo "CloudWatch logs are automatically configured."
echo "View logs at: https://console.aws.amazon.com/cloudwatch/home?region=$AWS_REGION#logsV2:log-groups/log-group//aws/lambda/$FUNCTION_NAME"

# Cleanup temporary files
rm -f trust-policy.json execution-policy.json

echo ""
echo "Deployment script completed!"
"""

print(cli_commands)

# Save CLI commands to file
cli_script_path = 'deploy_to_aws.sh'
with open(cli_script_path, 'w') as f:
    f.write(cli_commands)
os.chmod(cli_script_path, 0o755)
print(f"\n✓ CLI deployment script saved to: {cli_script_path}")

# ============================================================================
# 4. GENERATE BOTO3 DEPLOYMENT SCRIPT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Boto3 Python Deployment Script")
print("=" * 80)

boto3_script = """
import boto3
import json
import time
from pathlib import Path

# Configuration
AWS_REGION = 'us-east-1'
FUNCTION_NAME = 'equipment-failure-prediction-api'
API_NAME = 'EquipmentPredictionAPI'
ROLE_NAME = 'lambda-equipment-prediction-role'
LAMBDA_RUNTIME = 'python3.9'
LAMBDA_HANDLER = 'lambda_function.lambda_handler'
LAMBDA_MEMORY = 512
LAMBDA_TIMEOUT = 30
ZIP_FILE = 'lambda_deployment_package.zip'

# AWS Account ID (must be set)
AWS_ACCOUNT_ID = input("Enter your AWS Account ID: ").strip()

# Initialize clients
iam_client = boto3.client('iam', region_name=AWS_REGION)
lambda_client = boto3.client('lambda', region_name=AWS_REGION)
apigateway_client = boto3.client('apigateway', region_name=AWS_REGION)

print("\\n" + "=" * 80)
print("AWS Deployment - Equipment Failure Prediction API")
print("=" * 80)

try:
    # ========================================================================
    # STEP 1: Create IAM Role
    # ========================================================================
    print("\\n[1/6] Creating IAM role...")
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    
    try:
        role_response = iam_client.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Execution role for equipment prediction Lambda'
        )
        role_arn = role_response['Role']['Arn']
        print(f"✓ IAM role created: {role_arn}")
        
        # Attach execution policy
        execution_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "arn:aws:logs:*:*:*"
            }]
        }
        
        iam_client.put_role_policy(
            RoleName=ROLE_NAME,
            PolicyName='LambdaExecutionPolicy',
            PolicyDocument=json.dumps(execution_policy)
        )
        print("✓ Execution policy attached")
        
        print("  Waiting 10 seconds for IAM propagation...")
        time.sleep(10)
        
    except iam_client.exceptions.EntityAlreadyExistsException:
        role_arn = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{ROLE_NAME}"
        print(f"✓ Using existing IAM role: {role_arn}")
    
    # ========================================================================
    # STEP 2: Create Lambda Function
    # ========================================================================
    print("\\n[2/6] Creating Lambda function...")
    
    with open(ZIP_FILE, 'rb') as f:
        zip_content = f.read()
    
    try:
        lambda_response = lambda_client.create_function(
            FunctionName=FUNCTION_NAME,
            Runtime=LAMBDA_RUNTIME,
            Role=role_arn,
            Handler=LAMBDA_HANDLER,
            Code={'ZipFile': zip_content},
            Timeout=LAMBDA_TIMEOUT,
            MemorySize=LAMBDA_MEMORY,
            Environment={
                'Variables': {
                    'MODEL_VERSION': 'v1.0',
                    'ENABLE_MONITORING': 'true'
                }
            }
        )
        lambda_arn = lambda_response['FunctionArn']
        print(f"✓ Lambda function created: {lambda_arn}")
        
    except lambda_client.exceptions.ResourceConflictException:
        print("✓ Lambda function already exists, updating code...")
        lambda_client.update_function_code(
            FunctionName=FUNCTION_NAME,
            ZipFile=zip_content
        )
        lambda_arn = f"arn:aws:lambda:{AWS_REGION}:{AWS_ACCOUNT_ID}:function:{FUNCTION_NAME}"
        print(f"✓ Lambda function updated: {lambda_arn}")
    
    # ========================================================================
    # STEP 3: Create API Gateway
    # ========================================================================
    print("\\n[3/6] Creating API Gateway...")
    
    api_response = apigateway_client.create_rest_api(
        name=API_NAME,
        description='Equipment Failure Prediction REST API'
    )
    api_id = api_response['id']
    print(f"✓ API created: {api_id}")
    
    # ========================================================================
    # STEP 4: Configure API Resources
    # ========================================================================
    print("\\n[4/6] Configuring API resources...")
    
    # Get root resource
    resources = apigateway_client.get_resources(restApiId=api_id)
    root_id = resources['items'][0]['id']
    
    # Create /predict resource
    predict_resource = apigateway_client.create_resource(
        restApiId=api_id,
        parentId=root_id,
        pathPart='predict'
    )
    predict_resource_id = predict_resource['id']
    print(f"✓ Created /predict resource: {predict_resource_id}")
    
    # Create POST method
    apigateway_client.put_method(
        restApiId=api_id,
        resourceId=predict_resource_id,
        httpMethod='POST',
        authorizationType='NONE'
    )
    print("✓ POST method configured")
    
    # Configure Lambda integration
    lambda_uri = f"arn:aws:apigateway:{AWS_REGION}:lambda:path/2015-03-31/functions/{lambda_arn}/invocations"
    
    apigateway_client.put_integration(
        restApiId=api_id,
        resourceId=predict_resource_id,
        httpMethod='POST',
        type='AWS_PROXY',
        integrationHttpMethod='POST',
        uri=lambda_uri
    )
    print("✓ Lambda integration configured")
    
    # ========================================================================
    # STEP 5: Grant API Gateway Permission
    # ========================================================================
    print("\\n[5/6] Granting API Gateway permissions...")
    
    source_arn = f"arn:aws:execute-api:{AWS_REGION}:{AWS_ACCOUNT_ID}:{api_id}/*/*"
    
    try:
        lambda_client.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId='apigateway-access',
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com',
            SourceArn=source_arn
        )
        print("✓ Permission granted")
    except lambda_client.exceptions.ResourceConflictException:
        print("✓ Permission already exists")
    
    # ========================================================================
    # STEP 6: Deploy API
    # ========================================================================
    print("\\n[6/6] Deploying API to production stage...")
    
    apigateway_client.create_deployment(
        restApiId=api_id,
        stageName='prod',
        stageDescription='Production deployment',
        description='Initial deployment'
    )
    
    endpoint_url = f"https://{api_id}.execute-api.{AWS_REGION}.amazonaws.com/prod/predict"
    
    print("\\n" + "=" * 80)
    print("✓ DEPLOYMENT SUCCESSFUL!")
    print("=" * 80)
    print(f"\\nAPI Endpoint URL:\\n{endpoint_url}")
    print(f"\\nCloudWatch Logs:\\nhttps://console.aws.amazon.com/cloudwatch/home?region={AWS_REGION}#logsV2:log-groups/log-group//aws/lambda/{FUNCTION_NAME}")
    print("\\nSave the endpoint URL for testing and integration.\\n")
    
    # Return endpoint for testing
    with open('api_endpoint.txt', 'w') as f:
        f.write(endpoint_url)
    print("✓ Endpoint URL saved to: api_endpoint.txt")

except Exception as e:
    print(f"\\n❌ Deployment failed: {str(e)}")
    raise
"""

boto3_script_path = 'deploy_boto3.py'
with open(boto3_script_path, 'w') as f:
    f.write(boto3_script)
print(f"\n✓ Boto3 deployment script saved to: {boto3_script_path}")

# ============================================================================
# 5. GENERATE TESTING INSTRUCTIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: API Testing Instructions")
print("=" * 80)

test_instructions = """
# ============================================================================
# API ENDPOINT TESTING GUIDE
# ============================================================================

After deployment, test your API endpoint using the following methods:

# ----------------------------------------------------------------------------
# 1. CURL TEST
# ----------------------------------------------------------------------------

curl -X POST https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/prod/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "voltage": 245.5,
    "current": 42.3,
    "temperature": 55.2,
    "load_factor": 0.85
  }'

# Expected Response:
# {
#   "prediction": 0,
#   "risk_score": 15.3,
#   "confidence": 0.92,
#   "alert_level": "NORMAL",
#   "model_version": "v1.0",
#   "timestamp": "2026-01-02T12:00:00Z"
# }

# ----------------------------------------------------------------------------
# 2. PYTHON REQUESTS TEST
# ----------------------------------------------------------------------------

import requests
import json

endpoint_url = "https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/prod/predict"

test_data = {
    "voltage": 245.5,
    "current": 42.3,
    "temperature": 55.2,
    "load_factor": 0.85
}

response = requests.post(endpoint_url, json=test_data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# ----------------------------------------------------------------------------
# 3. BATCH TESTING
# ----------------------------------------------------------------------------

test_cases = [
    {"voltage": 245.5, "current": 42.3, "temperature": 55.2, "load_factor": 0.85},  # Normal
    {"voltage": 220.0, "current": 48.0, "temperature": 70.0, "load_factor": 0.95},  # Warning
    {"voltage": 210.0, "current": 52.0, "temperature": 80.0, "load_factor": 1.05},  # Critical
]

for i, test_data in enumerate(test_cases, 1):
    response = requests.post(endpoint_url, json=test_data)
    result = response.json()
    print(f"Test {i}: {result['alert_level']} (Risk: {result['risk_score']:.1f}%)")

# ----------------------------------------------------------------------------
# 4. MONITOR CLOUDWATCH LOGS
# ----------------------------------------------------------------------------

# View logs via AWS CLI:
aws logs tail /aws/lambda/equipment-failure-prediction-api --follow

# View logs in AWS Console:
# https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups
"""

print(test_instructions)

test_guide_path = 'API_TESTING_GUIDE.md'
with open(test_guide_path, 'w') as f:
    f.write(test_instructions)
print(f"\n✓ Testing guide saved to: {test_guide_path}")

# ============================================================================
# 6. GENERATE AUTHENTICATION SETUP GUIDE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: AWS Authentication Setup Guide")
print("=" * 80)

auth_guide = """
# ============================================================================
# AWS AUTHENTICATION SETUP GUIDE
# ============================================================================

## Prerequisites

1. **AWS Account**
   - Sign up at: https://aws.amazon.com/
   - Note your AWS Account ID (12-digit number)

2. **IAM User with Appropriate Permissions**
   - Create IAM user: https://console.aws.amazon.com/iam/
   - Required permissions:
     * IAM: CreateRole, PutRolePolicy, PassRole
     * Lambda: CreateFunction, UpdateFunctionCode, AddPermission
     * API Gateway: CreateRestApi, CreateResource, PutMethod, CreateDeployment
     * CloudWatch Logs: CreateLogGroup (automatic)

## Method 1: AWS CLI Setup (Recommended)

### Step 1: Install AWS CLI

# macOS
brew install awscli

# Windows
# Download from: https://aws.amazon.com/cli/

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

### Step 2: Configure Credentials

aws configure

# You will be prompted for:
# - AWS Access Key ID: [Your access key]
# - AWS Secret Access Key: [Your secret key]
# - Default region name: us-east-1
# - Default output format: json

### Step 3: Verify Configuration

aws sts get-caller-identity

# Should output your account ID and user ARN

## Method 2: Boto3 (Python SDK) Setup

### Step 1: Install Boto3

pip install boto3

### Step 2: Configure Credentials

Create ~/.aws/credentials file:

[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY

Create ~/.aws/config file:

[default]
region = us-east-1
output = json

### Step 3: Test Connection

python -c "import boto3; print(boto3.client('sts').get_caller_identity())"

## Getting AWS Access Keys

1. Sign in to AWS Console: https://console.aws.amazon.com/
2. Navigate to IAM → Users → [Your User]
3. Click "Security credentials" tab
4. Click "Create access key"
5. Download and save the key pair securely

⚠️ **IMPORTANT**: Never commit access keys to version control!

## Environment Variables (Alternative)

export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"

## IAM Role for EC2/ECS (Production)

For production deployments, use IAM roles instead of access keys:
- EC2: Attach IAM role to instance
- ECS: Configure task execution role
- Lambda: Automatically uses execution role

## Troubleshooting

### "Unable to locate credentials"
- Run: aws configure
- Verify ~/.aws/credentials exists

### "Access Denied"
- Verify IAM permissions
- Check policy attached to user/role

### "Invalid security token"
- Regenerate access keys
- Update credentials in ~/.aws/credentials

## Security Best Practices

1. **Use IAM roles** instead of access keys when possible
2. **Rotate access keys** regularly (every 90 days)
3. **Enable MFA** for AWS Console access
4. **Use least privilege** - grant only required permissions
5. **Never share credentials** or commit to Git
6. **Monitor usage** in CloudTrail
"""

auth_guide_path = 'AWS_AUTH_SETUP.md'
with open(auth_guide_path, 'w') as f:
    f.write(auth_guide)
print(f"\n✓ Authentication guide saved to: {auth_guide_path}")

# ============================================================================
# 7. DEPLOYMENT SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DEPLOYMENT AUTOMATION COMPLETE")
print("=" * 80)

deployment_summary = {
    'deployment_package': ZIP_FILE,
    'deployment_package_size_mb': round(zip_size_mb, 2),
    'cli_script': cli_script_path,
    'boto3_script': boto3_script_path,
    'testing_guide': test_guide_path,
    'auth_guide': auth_guide_path,
    'lambda_config': {
        'function_name': FUNCTION_NAME,
        'runtime': LAMBDA_RUNTIME,
        'handler': LAMBDA_HANDLER,
        'memory_mb': LAMBDA_MEMORY,
        'timeout_seconds': LAMBDA_TIMEOUT
    },
    'api_config': {
        'api_name': API_NAME,
        'region': REGION,
        'stage': 'prod',
        'endpoint_pattern': f'https://{{API_ID}}.execute-api.{REGION}.amazonaws.com/prod/predict'
    }
}

print("\n📦 Generated Files:")
print(f"  • {ZIP_FILE} ({zip_size_mb:.2f} MB) - Lambda deployment package")
print(f"  • {cli_script_path} - AWS CLI deployment script")
print(f"  • {boto3_script_path} - Boto3 Python deployment script")
print(f"  • {test_guide_path} - API testing instructions")
print(f"  • {auth_guide_path} - AWS authentication setup guide")

print("\n🚀 Deployment Options:")
print("\n  Option 1: AWS CLI (Bash)")
print(f"    1. Configure AWS: aws configure")
print(f"    2. Edit script: Set AWS_ACCOUNT_ID in {cli_script_path}")
print(f"    3. Run: ./{cli_script_path}")

print("\n  Option 2: Boto3 (Python)")
print(f"    1. Configure AWS: aws configure")
print(f"    2. Run: python {boto3_script_path}")
print(f"    3. Enter AWS Account ID when prompted")

print("\n✅ Next Steps:")
print("  1. Set up AWS authentication (see AWS_AUTH_SETUP.md)")
print("  2. Run deployment script (CLI or Boto3)")
print("  3. Save the generated API endpoint URL")
print("  4. Test the endpoint (see API_TESTING_GUIDE.md)")
print("  5. Monitor CloudWatch logs for performance")

print("\n📊 Architecture:")
print("  Lambda Function → API Gateway → HTTPS Endpoint")
print("  CloudWatch Logs → Monitoring & Debugging")
print("  IAM Role → Secure execution permissions")

print("\n" + "=" * 80)
print()

deployment_summary
