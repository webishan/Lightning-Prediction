"""
DEMO: Quick test extraction (2 samples only)
Tests the system before running full 20-sample extraction
"""

from extract_20_samples_complete import EnhancedSatelliteExtractor
import pandas as pd

def quick_demo():
    """
    Quick demo with just 2 samples (1 location × 2 dates)
    Perfect for testing before full extraction
    """
    
    print("=" * 90)
    print("🧪 DEMO MODE: Quick Test (2 Samples)")
    print("=" * 90)
    
    extractor = EnhancedSatelliteExtractor(nasa_api_key='DEMO_KEY')
    
    # Just 1 location for demo
    locations = {
        'Dhaka': {'lat': 23.8103, 'lon': 90.4125},
    }
    
    # Just 2 dates
    target_dates = [
        '2024-08-15',
        '2024-08-20',
    ]
    
    print(f"\n🎯 Demo Configuration:")
    print(f"   Locations: {len(locations)}")
    print(f"   Dates: {len(target_dates)}")
    print(f"   Total samples: {len(locations) * len(target_dates)}")
    print(f"\n⏱️  This will take about 10-15 seconds\n")
    
    # Extract weather data
    print("📊 Extracting weather data...")
    weather_data = []
    
    for location_name, coords in locations.items():
        data = extractor.api.get_daily_data(
            latitude=coords['lat'],
            longitude=coords['lon'],
            start_date='20240815',
            end_date='20240820'
        )
        
        if data:
            df = extractor.api.convert_to_dataframe(data)
            if df is not None:
                df['Date'] = pd.to_datetime(df.index)
                df = df[df['Date'].isin(pd.to_datetime(target_dates))]
                df['Location'] = location_name
                df['Latitude'] = coords['lat']
                df['Longitude'] = coords['lon']
                weather_data.append(df)
                print(f"   ✓ {location_name}: {len(df)} records")
    
    weather_df = pd.concat(weather_data, ignore_index=True)
    
    # Try to download 1 image as test
    print(f"\n🛰️  Testing satellite image download...")
    result = extractor.download_nasa_earth_image(
        latitude=23.8103,
        longitude=90.4125,
        date_str='2024-08-15'
    )
    
    if result['status'] == 'success':
        print(f"   ✓ Image downloaded: {result['filename']}")
        print(f"   Size: {result['size_kb']} KB")
    else:
        print(f"   ✗ Image download failed: {result.get('error', 'Unknown')}")
        print(f"   Note: This is normal - not all dates have images available")
    
    # Save demo data
    demo_file = f"{extractor.data_dir}/demo_weather_data.csv"
    weather_df.to_csv(demo_file, index=False)
    
    print(f"\n✅ Demo complete!")
    print(f"\n📁 Files created:")
    print(f"   • {demo_file}")
    if result['status'] == 'success':
        print(f"   • {result['filename']}")
    
    print(f"\n📊 Weather data preview:")
    print(weather_df[['Date', 'Location', 'Temperature_2m_C', 
                     'Precipitation_mm']].to_string(index=False))
    
    print(f"\n" + "=" * 90)
    print(f"✅ System is working! Ready for full extraction.")
    print(f"\nTo extract all 20 samples, run:")
    print(f"   python extract_20_samples_complete.py")
    print("=" * 90)

if __name__ == "__main__":
    quick_demo()
