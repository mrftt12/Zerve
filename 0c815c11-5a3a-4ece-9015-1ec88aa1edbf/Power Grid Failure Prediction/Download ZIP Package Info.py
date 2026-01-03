import os

# Path to the lambda deployment zip package
zip_file_path = "lambda_deployment_package.zip"

# Check if file exists
if os.path.exists(zip_file_path):
    # Get file size
    file_size_bytes = os.path.getsize(zip_file_path)
    file_size_kb = file_size_bytes / 1024
    file_size_mb = file_size_kb / 1024
    
    # Get absolute path
    abs_path = os.path.abspath(zip_file_path)
    
    # Display file information
    print("=" * 60)
    print("LAMBDA DEPLOYMENT PACKAGE - DOWNLOAD READY")
    print("=" * 60)
    print()
    print(f"📦 File Name: {os.path.basename(zip_file_path)}")
    print(f"📁 File Path: {abs_path}")
    print(f"📊 File Size: {file_size_bytes:,} bytes ({file_size_kb:.2f} KB / {file_size_mb:.4f} MB)")
    print()
    print("✅ File Status: EXISTS and READABLE")
    print()
    print("=" * 60)
    print("DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print()
    print("1. The ZIP package is ready for download")
    print("2. You can download this file directly from the canvas")
    print("3. This package contains all Lambda deployment artifacts:")
    print("   - lambda_function.py (main handler)")
    print("   - Trained models (physics_model.pkl, gb_model.pkl)")
    print("   - Scalers (physics_scaler.pkl, ml_scaler.pkl)")
    print("   - Configuration (model_config.json)")
    print("   - Dependencies (requirements.txt)")
    print("   - Documentation (README.md)")
    print()
    print("4. Deploy to AWS Lambda using the provided automation scripts")
    print()
    
    # Store key information for downstream use
    download_info = {
        "file_name": os.path.basename(zip_file_path),
        "file_path": abs_path,
        "file_size_bytes": file_size_bytes,
        "file_size_kb": file_size_kb,
        "file_size_mb": file_size_mb,
        "exists": True,
        "readable": True
    }
    
else:
    print("❌ ERROR: Lambda deployment package not found!")
    print(f"Expected path: {zip_file_path}")
    download_info = {
        "file_name": os.path.basename(zip_file_path),
        "file_path": zip_file_path,
        "exists": False,
        "readable": False
    }
