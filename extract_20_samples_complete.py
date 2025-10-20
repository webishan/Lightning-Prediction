"""
Enhanced Satellite + Weather Data Extractor
Uses multiple satellite data sources for better coverage
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from nasa_power_api import NASAPowerAPI
import time
import os
from pathlib import Path
import json

class EnhancedSatelliteExtractor:
    """
    Extract satellite images from multiple sources + weather data
    """
    
    def __init__(self, nasa_api_key='DEMO_KEY'):
        self.api = NASAPowerAPI()
        self.nasa_api_key = nasa_api_key
        self.satellite_dir = "satellite_images"
        self.data_dir = "weather_data"
        
        # Create directories
        Path(self.satellite_dir).mkdir(exist_ok=True)
        Path(self.data_dir).mkdir(exist_ok=True)
    
    def get_sentinel_metadata(self, latitude, longitude, date_str):
        """
        Get Sentinel-2 satellite metadata (ESA Copernicus)
        Note: This gets metadata only. Actual images require separate download.
        """
        # Sentinel Hub API would go here
        # For now, we'll create a placeholder
        return {
            'source': 'Sentinel-2',
            'location': f"{latitude},{longitude}",
            'date': date_str,
            'status': 'metadata_only',
            'note': 'Use Sentinel Hub or Google Earth Engine for actual images'
        }
    
    def get_nasa_earth_asset(self, latitude, longitude, date_str):
        """
        Get NASA Earth asset information
        """
        url = "https://api.nasa.gov/planetary/earth/assets"
        
        params = {
            'lon': longitude,
            'lat': latitude,
            'date': date_str,
            'dim': 0.15,
            'api_key': self.nasa_api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return {
                    'status': 'success',
                    'source': 'NASA Earth',
                    'date': data.get('date', date_str),
                    'url': data.get('url', ''),
                    'cloud_score': data.get('cloud_score', 'N/A')
                }
            else:
                return {'status': 'failed', 'error': f"HTTP {response.status_code}"}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def download_nasa_earth_image(self, latitude, longitude, date_str):
        """
        Download actual satellite image from NASA Earth API
        """
        url = "https://api.nasa.gov/planetary/earth/imagery"
        
        params = {
            'lon': longitude,
            'lat': latitude,
            'date': date_str,
            'dim': 0.15,
            'api_key': self.nasa_api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                filename = f"{self.satellite_dir}/nasa_earth_{latitude}_{longitude}_{date_str.replace('-', '')}.png"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                # Get file size
                file_size = os.path.getsize(filename) / 1024  # KB
                
                return {
                    'status': 'success',
                    'source': 'NASA Earth',
                    'filename': filename,
                    'size_kb': round(file_size, 2),
                    'date': date_str,
                    'lat': latitude,
                    'lon': longitude
                }
            else:
                return {
                    'status': 'failed',
                    'error': f"HTTP {response.status_code}",
                    'date': date_str
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'date': date_str
            }
    
    def extract_20_samples(self):
        """
        Extract exactly 20 samples: 4 locations × 5 dates
        """
        
        print("=" * 100)
        print("🎯 EXTRACTING 20 SAMPLES: SATELLITE IMAGES + WEATHER DATA")
        print("=" * 100)
        
        # 4 Strategic locations across Bangladesh
        locations = {
            'Dhaka': {'lat': 23.8103, 'lon': 90.4125},
            'Chittagong': {'lat': 22.3569, 'lon': 91.7832},
            'Sylhet': {'lat': 24.8949, 'lon': 91.8687},
            'Rangpur': {'lat': 25.7439, 'lon': 89.2752},
        }
        
        # 5 specific dates (early August 2024 - monsoon season)
        target_dates = [
            '2024-08-05',
            '2024-08-10',
            '2024-08-15',
            '2024-08-20',
            '2024-08-25',
        ]
        
        print(f"\n📍 Locations: {len(locations)}")
        for name, coords in locations.items():
            print(f"   • {name}: {coords['lat']:.4f}°N, {coords['lon']:.4f}°E")
        
        print(f"\n📅 Dates: {len(target_dates)}")
        for date in target_dates:
            print(f"   • {date}")
        
        print(f"\n🎯 Total Samples: {len(locations)} × {len(target_dates)} = {len(locations) * len(target_dates)}")
        
        # STEP 1: Extract weather data for all dates
        print("\n" + "=" * 100)
        print("📊 STEP 1: EXTRACTING WEATHER DATA (NASA POWER API)")
        print("=" * 100)
        
        weather_data = []
        
        for idx, (location_name, coords) in enumerate(locations.items(), 1):
            print(f"\n[{idx}/{len(locations)}] 📍 {location_name}")
            
            try:
                # Get full date range
                data = self.api.get_daily_data(
                    latitude=coords['lat'],
                    longitude=coords['lon'],
                    start_date='20240805',
                    end_date='20240825'
                )
                
                if data:
                    df = self.api.convert_to_dataframe(data)
                    if df is not None:
                        # Filter to only target dates
                        df['Date'] = pd.to_datetime(df.index)
                        df = df[df['Date'].isin(pd.to_datetime(target_dates))]
                        
                        df['Location'] = location_name
                        df['Latitude'] = coords['lat']
                        df['Longitude'] = coords['lon']
                        
                        weather_data.append(df)
                        print(f"    ✓ Retrieved {len(df)} records")
                    else:
                        print(f"    ✗ Failed to convert data")
                else:
                    print(f"    ✗ No data received")
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
            
            time.sleep(1)  # Be respectful to API
        
        # Combine weather data
        if weather_data:
            weather_df = pd.concat(weather_data, ignore_index=True)
            weather_df['Date'] = pd.to_datetime(weather_df['Date'])
            
            # Reorder columns
            cols = ['Date', 'Location', 'Latitude', 'Longitude'] + \
                   [col for col in weather_df.columns if col not in 
                    ['Date', 'Location', 'Latitude', 'Longitude']]
            weather_df = weather_df[cols]
            
            # Sort by date and location
            weather_df = weather_df.sort_values(['Date', 'Location']).reset_index(drop=True)
            
            print(f"\n✅ Weather data extracted: {len(weather_df)} records")
            print(f"   Columns: {weather_df.shape[1]}")
        else:
            weather_df = pd.DataFrame()
            print("\n❌ No weather data extracted")
        
        # STEP 2: Download satellite images
        print("\n" + "=" * 100)
        print("🛰️  STEP 2: DOWNLOADING SATELLITE IMAGES (NASA Earth API)")
        print("=" * 100)
        
        print(f"\n⚠️  API Key: {self.nasa_api_key}")
        if self.nasa_api_key == 'DEMO_KEY':
            print(f"    ⚠️  Using DEMO_KEY - Limited to 30 requests/hour")
            print(f"    💡 Get free API key at: https://api.nasa.gov/")
        
        print(f"\n🎯 Downloading {len(locations) * len(target_dates)} images...")
        
        image_info = []
        downloaded = 0
        failed = 0
        
        for date_str in target_dates:
            print(f"\n📅 {date_str}")
            
            for location_name, coords in locations.items():
                print(f"   📍 {location_name}...", end=' ')
                
                result = self.download_nasa_earth_image(
                    latitude=coords['lat'],
                    longitude=coords['lon'],
                    date_str=date_str
                )
                
                result['location'] = location_name
                image_info.append(result)
                
                if result['status'] == 'success':
                    downloaded += 1
                    print(f"✓ {result['size_kb']} KB")
                else:
                    failed += 1
                    print(f"✗ {result.get('error', 'Failed')}")
                
                # Delay to respect rate limits
                time.sleep(2.5)  # ~24 requests/minute for DEMO_KEY
        
        print(f"\n📊 Download Summary:")
        print(f"   ✓ Success: {downloaded}")
        print(f"   ✗ Failed: {failed}")
        print(f"   📁 Total: {len(image_info)}")
        
        # Create image info dataframe
        image_df = pd.DataFrame(image_info)
        
        # STEP 3: Save all data
        print("\n" + "=" * 100)
        print("💾 STEP 3: SAVING DATA")
        print("=" * 100)
        
        # Save weather data
        weather_file = f"{self.data_dir}/weather_data_20_samples.csv"
        weather_df.to_csv(weather_file, index=False)
        print(f"\n✅ Weather Data Saved:")
        print(f"   📁 File: {weather_file}")
        print(f"   📊 Records: {len(weather_df)}")
        print(f"   📈 Features: {weather_df.shape[1]}")
        
        # Save image metadata
        image_file = f"{self.data_dir}/satellite_images_metadata.csv"
        image_df.to_csv(image_file, index=False)
        print(f"\n✅ Satellite Image Metadata Saved:")
        print(f"   📁 File: {image_file}")
        print(f"   📊 Records: {len(image_df)}")
        print(f"   🛰️  Successful downloads: {downloaded}")
        
        # Create summary file
        summary = {
            'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(weather_df),
            'locations': list(locations.keys()),
            'dates': target_dates,
            'weather_records': len(weather_df),
            'satellite_images_downloaded': downloaded,
            'satellite_images_failed': failed,
            'weather_file': weather_file,
            'image_file': image_file,
            'image_directory': self.satellite_dir
        }
        
        summary_file = f"{self.data_dir}/extraction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✅ Summary Saved:")
        print(f"   📁 File: {summary_file}")
        
        # Display preview
        print("\n" + "=" * 100)
        print("👁️  DATA PREVIEW")
        print("=" * 100)
        
        print("\n📊 Weather Data (first 10 rows):")
        print(weather_df[['Date', 'Location', 'Temperature_2m_C', 
                         'Precipitation_mm', 'Relative_Humidity_%']].head(10).to_string(index=False))
        
        print("\n🛰️  Satellite Images (successful downloads):")
        successful_images = image_df[image_df['status'] == 'success']
        if len(successful_images) > 0:
            print(successful_images[['location', 'date', 'filename', 'size_kb']].to_string(index=False))
        else:
            print("   ⚠️  No images downloaded successfully")
        
        # Final summary
        print("\n" + "=" * 100)
        print("✅ EXTRACTION COMPLETE!")
        print("=" * 100)
        
        print(f"""
📁 OUTPUT FILES:
   1. {weather_file}
      • Weather parameters for all 20 samples
      • Ready for CSV import
   
   2. {image_file}
      • Satellite image metadata and download status
   
   3. {self.satellite_dir}/
      • Downloaded satellite images (PNG format)
      • {downloaded} images available
   
   4. {summary_file}
      • Extraction summary and metadata

📊 DATASET SUMMARY:
   • Total samples: {len(weather_df)}
   • Locations: {len(locations)}
   • Time points: {len(target_dates)}
   • Weather features: {weather_df.shape[1]}
   • Satellite images: {downloaded}/{len(image_info)}

🎯 STATUS: {'✅ Ready for analysis!' if len(weather_df) == 20 else '⚠️ Check for missing data'}

💡 NEXT STEPS:
   1. Review weather data CSV
   2. Check satellite images in {self.satellite_dir}/
   3. Use for lightning detection model training
        """)
        
        return weather_df, image_df, summary


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("🛰️  SATELLITE IMAGES + WEATHER DATA EXTRACTION TOOL")
    print("=" * 100)
    
    print("""
This script will extract:
   ✓ 20 weather data samples (4 locations × 5 dates)
   ✓ Corresponding satellite images
   ✓ Data saved in separate CSV files
   ✓ Images saved in dedicated folder

Configuration:
   • Locations: Dhaka, Chittagong, Sylhet, Rangpur
   • Dates: Aug 5, 10, 15, 20, 25 (2024)
   • Weather: NASA POWER API
   • Images: NASA Earth API
    """)
    
    # Check if user has API key
    print("=" * 100)
    api_key = input("\nEnter NASA API key (or press Enter to use DEMO_KEY): ").strip()
    
    if not api_key:
        api_key = 'DEMO_KEY'
        print("⚠️  Using DEMO_KEY - Limited to 30 requests/hour")
        print("💡 Get free key at: https://api.nasa.gov/")
    
    print("\n🚀 Starting extraction...\n")
    
    # Create extractor and run
    extractor = EnhancedSatelliteExtractor(nasa_api_key=api_key)
    weather_df, image_df, summary = extractor.extract_20_samples()
