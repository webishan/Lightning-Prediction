"""
Test Google Earth Engine Access
Run this after setting up GEE to verify everything works
"""

import sys

def test_gee_setup():
    """Test if Google Earth Engine is properly set up"""
    
    print("=" * 70)
    print("🌍 TESTING GOOGLE EARTH ENGINE ACCESS")
    print("=" * 70)
    print()
    
    # Test 1: Check if earthengine-api is installed
    print("Test 1: Checking if earthengine-api is installed...")
    try:
        import ee
        print("✅ earthengine-api is installed")
    except ImportError:
        print("❌ earthengine-api is NOT installed")
        print("\n💡 Fix: Run this command:")
        print("   pip install earthengine-api")
        return False
    
    print()
    
    # Test 2: Check if authenticated
    print("Test 2: Checking authentication...")
    try:
        # Initialize with the user's Google Cloud project
        ee.Initialize(project='bangladesh-lightning-detection')
        print("✅ Successfully authenticated with Google Earth Engine")
        print("   Project: bangladesh-lightning-detection")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("\n💡 Fix: Run this command:")
        print("   python -c \"import ee; ee.Authenticate()\"")
        print("\nThis will open a browser for you to login with your Google account.")
        return False
    
    print()
    
    # Test 3: Try to access a dataset
    print("Test 3: Trying to access Landsat 8 imagery...")
    try:
        # Get Landsat 8 collection
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        
        # Count images over Bangladesh in August 2024
        bangladesh_region = ee.Geometry.Rectangle([88.0, 20.5, 92.7, 26.6])
        
        images = collection.filterBounds(bangladesh_region) \
                          .filterDate('2024-08-01', '2024-08-31') \
                          .filter(ee.Filter.lt('CLOUD_COVER', 50))
        
        count = images.size().getInfo()
        
        print(f"✅ Found {count} Landsat 8 images over Bangladesh in August 2024")
        print("✅ You can access satellite imagery!")
        
    except Exception as e:
        print(f"❌ Error accessing data: {e}")
        print("\n💡 Possible issues:")
        print("   1. Your GEE account may not be approved yet")
        print("   2. Network connection issue")
        print("   3. Need to re-authenticate")
        return False
    
    print()
    
    # Test 4: Try to get image info
    print("Test 4: Getting sample image information...")
    try:
        # Get one image
        image = images.first()
        
        # Get some properties
        date = image.date().format('YYYY-MM-dd').getInfo()
        cloud_cover = image.get('CLOUD_COVER').getInfo()
        path_row = f"{image.get('WRS_PATH').getInfo()}/{image.get('WRS_ROW').getInfo()}"
        
        print(f"✅ Sample image details:")
        print(f"   Date: {date}")
        print(f"   Cloud Cover: {cloud_cover:.1f}%")
        print(f"   Path/Row: {path_row}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not get image details: {e}")
        print("   (This is OK if there are no images for this date)")
    
    print()
    print("=" * 70)
    print("🎉 SUCCESS! Google Earth Engine is working perfectly!")
    print("=" * 70)
    print()
    print("✅ You are ready to extract satellite imagery!")
    print()
    print("Next steps:")
    print("   1. Run: python extract_hybrid_gee_nasa.py")
    print("   2. This will extract images from GEE + weather from NASA POWER")
    print()
    
    return True


def check_prerequisites():
    """Check if all prerequisites are met"""
    
    print("Checking prerequisites...")
    print()
    
    # Check Python version
    import sys
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("⚠️  Warning: Python 3.7+ recommended for Earth Engine")
    else:
        print("✅ Python version OK")
    
    print()
    
    # Check required packages
    required_packages = ['ee', 'requests', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print()
        print("💡 Install missing packages:")
        if 'ee' in missing_packages:
            print("   pip install earthengine-api")
        if 'requests' in missing_packages:
            print("   pip install requests")
        if 'pandas' in missing_packages:
            print("   pip install pandas")
        return False
    
    print()
    return True


if __name__ == "__main__":
    print()
    
    # Check prerequisites first
    if not check_prerequisites():
        print()
        print("❌ Please install missing packages first")
        sys.exit(1)
    
    print()
    
    # Run main test
    success = test_gee_setup()
    
    if success:
        print("🚀 Ready to extract data!")
        sys.exit(0)
    else:
        print("❌ Setup incomplete. Follow the instructions above.")
        sys.exit(1)
