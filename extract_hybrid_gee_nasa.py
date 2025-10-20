"""
Hybrid Data Extractor
- Satellite Images: Google Earth Engine (Landsat 8)
- Weather Data: NASA POWER API

This combines the reliability of GEE with the free weather data from NASA POWER.
"""

import ee
import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Import our existing NASA POWER API wrapper
try:
    from nasa_power_api import NASAPowerAPI
except ImportError:
    print("❌ Error: nasa_power_api.py not found in current directory")
    print("💡 Make sure nasa_power_api.py is in the same folder")
    sys.exit(1)


class HybridDataExtractor:
    """
    Extracts synchronized satellite imagery and weather data
    - Google Earth Engine: Landsat 8 imagery
    - NASA POWER API: Weather parameters
    """
    
    def __init__(self):
        """Initialize the hybrid extractor"""
        
        # Initialize Earth Engine
        try:
            # Initialize with the user's Google Cloud project
            ee.Initialize(project='bangladesh-lightning-detection')
            print("✅ Google Earth Engine initialized (project: bangladesh-lightning-detection)")
        except Exception as e:
            print(f"❌ Error initializing Earth Engine: {e}")
            print("\n💡 Make sure you ran: python -c \"import ee; ee.Authenticate()\"")
            sys.exit(1)
        
        # Initialize NASA POWER API
        self.nasa_api = NASAPowerAPI()
        print("✅ NASA POWER API initialized")
        
        # Create directories
        self.base_dir = Path(__file__).parent
        self.image_dir = self.base_dir / 'satellite_images_gee'
        self.data_dir = self.base_dir / 'weather_data'
        self.image_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        self.results = []
        self.weather_data = []
    
    def get_landsat8_image(self, lat, lon, date, location_name, cloud_cover_max=50):
        """
        Get Landsat 8 image from Google Earth Engine
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date string 'YYYY-MM-DD'
            location_name: Name of location
            cloud_cover_max: Maximum cloud cover percentage (default 50%)
        
        Returns:
            dict: Result with status and image info
        """
        
        try:
            # Parse date
            target_date = datetime.strptime(date, '%Y-%m-%d')
            
            # Create date range (±8 days for Landsat 8's 16-day cycle)
            start_date = (target_date - timedelta(days=8)).strftime('%Y-%m-%d')
            end_date = (target_date + timedelta(days=8)).strftime('%Y-%m-%d')
            
            # Define point of interest
            point = ee.Geometry.Point([lon, lat])
            
            # Get Landsat 8 Collection 2, Level 2 (surface reflectance)
            collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterBounds(point) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover_max)) \
                .sort('CLOUD_COVER')
            
            # Get image count
            count = collection.size().getInfo()
            
            if count == 0:
                return {
                    'status': 'no_data',
                    'error': f'No cloud-free images found within ±8 days of {date}',
                    'date': date,
                    'location': location_name,
                    'images_found': 0
                }
            
            # Get the best image (lowest cloud cover)
            image = collection.first()
            
            # Get image metadata
            image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            cloud_cover = image.get('CLOUD_COVER').getInfo()
            scene_id = image.get('LANDSAT_SCENE_ID').getInfo()
            
            # Scale factors for Landsat 8 Collection 2
            def apply_scale_factors(img):
                optical_bands = img.select('SR_B.').multiply(0.0000275).add(-0.2)
                return img.addBands(optical_bands, None, True)
            
            scaled_image = apply_scale_factors(image)
            
            # Create RGB composite (Red, Green, Blue)
            rgb = scaled_image.select(['SR_B4', 'SR_B3', 'SR_B2'])
            
            # Define region (15km x 15km around point, matching NASA Earth API)
            region = point.buffer(7500).bounds()
            
            # Get download URL
            url = rgb.getThumbURL({
                'region': region.getInfo(),
                'dimensions': 512,
                'format': 'png',
                'min': 0,
                'max': 0.3
            })
            
            # Download image
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                # Save image
                filename = f"{location_name}_{image_date}.png"
                filepath = self.image_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                file_size = filepath.stat().st_size / 1024  # KB
                
                print(f"   ✅ {location_name}: {image_date} ({cloud_cover:.1f}% clouds, {file_size:.1f} KB)")
                
                return {
                    'status': 'success',
                    'filename': filename,
                    'file_size_kb': file_size,
                    'date': date,
                    'actual_image_date': image_date,
                    'location': location_name,
                    'cloud_cover': cloud_cover,
                    'scene_id': scene_id,
                    'images_found': count
                }
            else:
                return {
                    'status': 'error',
                    'error': f'HTTP {response.status_code}',
                    'date': date,
                    'location': location_name
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'date': date,
                'location': location_name
            }
    
    def get_weather_data(self, lat, lon, date, location_name):
        """
        Get weather data from NASA POWER API
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date string 'YYYY-MM-DD'
            location_name: Name of location
        
        Returns:
            dict: Weather data
        """
        
        try:
            # Convert date format from YYYY-MM-DD to YYYYMMDD for NASA POWER API
            date_formatted = date.replace('-', '')
            
            # Get data for the specific date
            data = self.nasa_api.get_daily_data(
                latitude=lat,
                longitude=lon,
                start_date=date_formatted,
                end_date=date_formatted
            )
            
            if data and 'properties' in data and 'parameter' in data['properties']:
                params = data['properties']['parameter']
                
                # Extract all parameters using the formatted date (YYYYMMDD)
                weather_record = {
                    'Date': date,  # Keep original format for display
                    'Location': location_name,
                    'Latitude': lat,
                    'Longitude': lon,
                    'Temperature_2m_C': params.get('T2M', {}).get(date_formatted),
                    'Temperature_Max_C': params.get('T2M_MAX', {}).get(date_formatted),
                    'Temperature_Min_C': params.get('T2M_MIN', {}).get(date_formatted),
                    'Temperature_Dewpoint_C': params.get('T2MDEW', {}).get(date_formatted),
                    'Temperature_Range_C': params.get('T2M_RANGE', {}).get(date_formatted),
                    'Relative_Humidity_%': params.get('RH2M', {}).get(date_formatted),
                    'Specific_Humidity_g/kg': params.get('QV2M', {}).get(date_formatted),
                    'Precipitation_mm': params.get('PRECTOTCORR', {}).get(date_formatted),
                    'Wind_Speed_2m_m/s': params.get('WS2M', {}).get(date_formatted),
                    'Wind_Speed_10m_m/s': params.get('WS10M', {}).get(date_formatted),
                    'Wind_Direction_deg': params.get('WD2M', {}).get(date_formatted),
                    'Surface_Pressure_kPa': params.get('PS', {}).get(date_formatted),
                    'Solar_Radiation_Shortwave_W/m2': params.get('ALLSKY_SFC_SW_DWN', {}).get(date_formatted),
                    'Solar_Radiation_Longwave_W/m2': params.get('ALLSKY_SFC_LW_DWN', {}).get(date_formatted),
                }
                
                print(f"   ✅ {location_name}: Weather data retrieved")
                return weather_record
            else:
                print(f"   ❌ {location_name}: No weather data available")
                return None
                
        except Exception as e:
            print(f"   ❌ {location_name}: Weather error - {e}")
            return None
    
    def extract_hybrid_data(self, locations, dates, cloud_cover_max=50):
        """
        Extract both satellite imagery and weather data
        
        Args:
            locations: Dict of location names and coordinates {'Name': {'lat': x, 'lon': y}}
            dates: List of date strings ['YYYY-MM-DD', ...]
            cloud_cover_max: Maximum cloud cover for satellite images (default 50%)
        """
        
        total_samples = len(locations) * len(dates)
        
        print("=" * 70)
        print("🌍 HYBRID DATA EXTRACTION")
        print("   Satellite Images: Google Earth Engine (Landsat 8)")
        print("   Weather Data: NASA POWER API")
        print("=" * 70)
        print(f"📍 Locations: {len(locations)}")
        print(f"📅 Dates: {len(dates)}")
        print(f"📊 Total samples: {total_samples}")
        print(f"☁️  Max cloud cover: {cloud_cover_max}%")
        print("=" * 70)
        print()
        
        start_time = time.time()
        
        for idx, (location_name, coords) in enumerate(locations.items(), 1):
            print(f"[{idx}/{len(locations)}] 📍 {location_name} ({coords['lat']:.4f}°N, {coords['lon']:.4f}°E)")
            print("-" * 70)
            
            for date in dates:
                # Get satellite image
                print(f"🛰️  Satellite: {date}...", end=" ")
                image_result = self.get_landsat8_image(
                    lat=coords['lat'],
                    lon=coords['lon'],
                    date=date,
                    location_name=location_name,
                    cloud_cover_max=cloud_cover_max
                )
                self.results.append(image_result)
                
                # Get weather data
                print(f"☁️  Weather: {date}...", end=" ")
                weather_data = self.get_weather_data(
                    lat=coords['lat'],
                    lon=coords['lon'],
                    date=date,
                    location_name=location_name
                )
                if weather_data:
                    self.weather_data.append(weather_data)
                
                print()
                
                # Respectful delay
                time.sleep(2)
            
            print()
        
        # Save results
        self.save_results()
        
        # Print summary
        elapsed = time.time() - start_time
        self.print_summary(elapsed)
    
    def save_results(self):
        """Save extraction results to CSV files"""
        
        # Save satellite image metadata
        satellite_df = pd.DataFrame(self.results)
        satellite_file = self.data_dir / 'satellite_images_gee_metadata.csv'
        satellite_df.to_csv(satellite_file, index=False)
        print(f"\n💾 Satellite metadata saved: {satellite_file}")
        
        # Save weather data
        weather_df = pd.DataFrame(self.weather_data)
        weather_file = self.data_dir / 'weather_data_hybrid.csv'
        weather_df.to_csv(weather_file, index=False)
        print(f"💾 Weather data saved: {weather_file}")
    
    def print_summary(self, elapsed_time):
        """Print extraction summary"""
        
        satellite_df = pd.DataFrame(self.results)
        weather_df = pd.DataFrame(self.weather_data)
        
        # Satellite summary
        total_sat = len(satellite_df)
        successful_sat = len(satellite_df[satellite_df['status'] == 'success'])
        no_data_sat = len(satellite_df[satellite_df['status'] == 'no_data'])
        error_sat = len(satellite_df[satellite_df['status'] == 'error'])
        success_rate_sat = (successful_sat / total_sat * 100) if total_sat > 0 else 0
        
        # Weather summary
        total_weather = len(weather_df)
        success_rate_weather = (total_weather / total_sat * 100) if total_sat > 0 else 0
        
        print("\n" + "=" * 70)
        print("📊 EXTRACTION SUMMARY")
        print("=" * 70)
        
        print("\n🛰️  SATELLITE IMAGES (Google Earth Engine):")
        print(f"   ✅ Successful: {successful_sat}/{total_sat} ({success_rate_sat:.1f}%)")
        print(f"   ⚠️  No data: {no_data_sat}/{total_sat}")
        print(f"   ❌ Errors: {error_sat}/{total_sat}")
        
        if successful_sat > 0:
            avg_cloud = satellite_df[satellite_df['status'] == 'success']['cloud_cover'].mean()
            avg_size = satellite_df[satellite_df['status'] == 'success']['file_size_kb'].mean()
            total_size = satellite_df[satellite_df['status'] == 'success']['file_size_kb'].sum()
            print(f"   ☁️  Avg cloud cover: {avg_cloud:.1f}%")
            print(f"   💾 Total size: {total_size:.1f} KB ({total_size/1024:.2f} MB)")
            print(f"   📏 Avg image size: {avg_size:.1f} KB")
        
        print(f"\n☁️  WEATHER DATA (NASA POWER API):")
        print(f"   ✅ Successful: {total_weather}/{total_sat} ({success_rate_weather:.1f}%)")
        print(f"   📊 Parameters: 14 weather variables per sample")
        
        print(f"\n⏱️  PERFORMANCE:")
        print(f"   Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"   Avg per sample: {elapsed_time/total_sat:.1f} seconds")
        
        print("=" * 70)
        
        print("\n📂 OUTPUT FILES:")
        print(f"   🛰️  {self.image_dir} - Satellite images")
        print(f"   📊 {self.data_dir}/satellite_images_gee_metadata.csv")
        print(f"   📊 {self.data_dir}/weather_data_hybrid.csv")
        
        print("\n✅ Extraction complete!")


def main():
    """Main execution function"""
    
    print("\n" + "=" * 70)
    print("🌍 HYBRID DATA EXTRACTOR")
    print("   Google Earth Engine + NASA POWER API")
    print("=" * 70)
    
    # Configuration
    locations = {
        'Dhaka': {'lat': 23.8103, 'lon': 90.4125},
        'Chittagong': {'lat': 22.3569, 'lon': 91.7832},
        'Sylhet': {'lat': 24.8949, 'lon': 91.8687},
        'Rangpur': {'lat': 25.7439, 'lon': 89.2752},
    }
    
    # Dates - August 2023 (monsoon season) - Using historical dates with confirmed data availability
    dates = ['2023-08-15', '2023-08-18', '2023-08-21', '2023-08-24', '2023-08-27']
    
    print(f"\n📍 Locations: {', '.join(locations.keys())}")
    print(f"📅 Dates: {', '.join(dates)}")
    print(f"📊 Total samples: {len(locations) * len(dates)}")
    
    # Ask for cloud cover threshold
    print(f"\n☁️  Cloud cover threshold:")
    print("   • Lower = Clearer images, but fewer results")
    print("   • Higher = More results, but cloudier images")
    print("   • Recommended: 30-50%")
    
    cloud_input = input("\nMax cloud cover % [50]: ").strip()
    cloud_cover_max = int(cloud_input) if cloud_input else 50
    
    print(f"\n✅ Using max cloud cover: {cloud_cover_max}%")
    
    input("\nPress ENTER to start extraction...")
    
    # Create extractor and run
    extractor = HybridDataExtractor()
    extractor.extract_hybrid_data(
        locations=locations,
        dates=dates,
        cloud_cover_max=cloud_cover_max
    )
    
    print("\n🎉 All done! Your hybrid dataset is ready for analysis.")


if __name__ == "__main__":
    main()
