import zipfile
import os
from pathlib import Path

# CREATE AWS LAMBDA DEPLOYMENT ZIP PACKAGE
print("=" * 70)
print("AWS LAMBDA DEPLOYMENT ZIP PACKAGE CREATOR")
print("=" * 70)

# Define paths
lambda_deployment_folder = 'lambda_deployment'
output_zip_path = 'lambda_deployment_package.zip'

print(f"\n📦 Creating deployment package...")
print(f"   Source: {lambda_deployment_folder}/")
print(f"   Target: {output_zip_path}")

# Verify deployment folder exists
if not os.path.exists(lambda_deployment_folder):
    print(f"\n❌ ERROR: Deployment folder '{lambda_deployment_folder}' not found!")
    raise FileNotFoundError(f"Deployment folder '{lambda_deployment_folder}' does not exist")

# Get list of files to include
files_to_zip = []
for root, dirs, files in os.walk(lambda_deployment_folder):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        # Calculate relative path for zip archive
        arcname = os.path.relpath(file_path, lambda_deployment_folder)
        files_to_zip.append((file_path, arcname))

print(f"\n📋 Files to package: {len(files_to_zip)}")
print("\n   Files included:")

# Create ZIP file
with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path, arcname in files_to_zip:
        zipf.write(file_path, arcname)
        file_size_kb = os.path.getsize(file_path) / 1024
        print(f"      ✓ {arcname:<35} ({file_size_kb:>8.1f} KB)")

# Verify ZIP was created and get size
if os.path.exists(output_zip_path):
    zip_size_bytes = os.path.getsize(output_zip_path)
    zip_size_kb = zip_size_bytes / 1024
    zip_size_mb = zip_size_kb / 1024
    
    print("\n" + "=" * 70)
    print("✅ DEPLOYMENT PACKAGE CREATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📦 Package Details:")
    print(f"   Filename: {output_zip_path}")
    print(f"   Size: {zip_size_kb:.2f} KB ({zip_size_mb:.2f} MB)")
    print(f"   Files: {len(files_to_zip)}")
    print(f"   Compression: ZIP_DEFLATED")
    
    print(f"\n📂 Package Contents:")
    print(f"   • ML Models (4 files): physics_model.pkl, gb_model.pkl, + scalers")
    print(f"   • Lambda Handler: lambda_function.py")
    print(f"   • Configuration: model_config.json")
    print(f"   • Dependencies: requirements.txt")
    print(f"   • Documentation: README.md, deployment_checklist.json")
    
    print(f"\n🚀 Ready for AWS Lambda Deployment!")
    print(f"\n📝 Next Steps:")
    print(f"   1. Upload {output_zip_path} to AWS Lambda Console")
    print(f"   2. Or use AWS CLI:")
    print(f"      aws lambda create-function \\")
    print(f"        --function-name equipment-failure-predictor \\")
    print(f"        --runtime python3.9 \\")
    print(f"        --handler lambda_function.lambda_handler \\")
    print(f"        --zip-file fileb://{output_zip_path} \\")
    print(f"        --role arn:aws:iam::ACCOUNT_ID:role/lambda-execution-role \\")
    print(f"        --memory-size 512 \\")
    print(f"        --timeout 30")
    print(f"\n   3. Connect to API Gateway for HTTP endpoints")
    print(f"   4. Test endpoints: /predict, /health, /model-info")
    
    print("\n" + "=" * 70)
    
    # Verify ZIP contents
    print("\n🔍 Verifying ZIP package contents...")
    with zipfile.ZipFile(output_zip_path, 'r') as zipf:
        zip_info_list = zipf.infolist()
        print(f"   ✓ ZIP contains {len(zip_info_list)} files")
        
        # Check for essential files
        essential_files = [
            'lambda_function.py',
            'model_config.json',
            'requirements.txt',
            'physics_model.pkl',
            'gb_model.pkl',
            'physics_scaler.pkl',
            'ml_scaler.pkl'
        ]
        
        files_in_zip = [info.filename for info in zip_info_list]
        missing_files = [f for f in essential_files if f not in files_in_zip]
        
        if missing_files:
            print(f"\n   ⚠️ WARNING: Missing essential files: {missing_files}")
        else:
            print(f"   ✓ All essential files present")
    
    print("\n✅ Deployment package validation complete!")
    print("=" * 70)
    
    zip_package_created = True
    zip_package_path_final = output_zip_path
    zip_package_size_mb_final = zip_size_mb
    
else:
    print(f"\n❌ ERROR: Failed to create ZIP file at {output_zip_path}")
    zip_package_created = False
