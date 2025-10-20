"""
30-Minute Interval Data Extractor
Extracts weather data every 30 minutes using NASA POWER hourly data
Combined with Google Earth Engine satellite imagery
"""

import ee
import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta
import sys

try:
    from nasa_power_api import NASAPowerAPI
except ImportError:
    print("❌ Error: nasa_power_api.py not found")
    sys.exit(1)


class ThirtyMinuteExtractor:
    """
    Extracts data at 30-minute intervals
    - Google Earth Engine: Landsat 8 imagery
    - NASA POWER API: Hourly weather data (interpolated to 30-min)
    """
    
    def __init__(self):
        """Initialize the extractor"""
        
        # Initialize Earth Engine
        try:
            ee.Initialize(project='bangladesh-lightning-detection')
            print("✅ Google Earth Engine initialized")
        except Exception as e:
            print(f"❌ Error initializing Earth Engine: {e}")
            sys.exit(1)
        
        # Initialize NASA POWER API
        self.nasa_api = NASAPowerAPI()
        print("✅ NASA POWER API initialized")
        
        # Create directories
        self.base_dir = Path(__file__).parent
        self.image_dir = self.base_dir / 'satellite_images_30min'
        self.data_dir = self.base_dir / 'weather_data'
        self.image_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        self.results = []
        self.weather_data = []
    
    def get_hourly_weather_data(self, lat, lon, date, location_name):
        """
        Get hourly weather data from NASA POWER API
        
        Args:
            lat: Latitude
            lon: Longitude
            date: Date string 'YYYY-MM-DD'
            location_name: Name of location
        
        Returns:
            list: List of hourly weather records
        """
        
        try:
            # Convert date format
            date_formatted = date.replace('-', '')
            
            # Get hourly data for the specific date
            data = self.nasa_api.get_hourly_data(
                latitude=lat,
                longitude=lon,
                start_date=date_formatted,
                end_date=date_formatted
            )
            
            if data and 'properties' in data and 'parameter' in data['properties']:
                params = data['properties']['parameter']
                
                # Extract hourly data and create 30-minute intervals
                weather_records = []
                
                # NASA POWER provides hourly data (24 hours per day)
                for hour in range(24):
                    # Hour key format: YYYYMMDDHH
                    hour_key = f"{date_formatted}{hour:02d}"
                    
                    # Get values for this hour
                    temp = params.get('T2M', {}).get(hour_key)
                    rh = params.get('RH2M', {}).get(hour_key)
                    ws = params.get('WS2M', {}).get(hour_key)
                    precip = params.get('PRECTOTCORR', {}).get(hour_key)
                    
                    if temp is not None:  # Only add if we have data
                        # Create record for :00 minutes
                        time_00 = f"{hour:02d}:00"
                        weather_records.append({
                            'Date': date,
                            'Time': time_00,
                            'DateTime': f"{date} {time_00}",
                            'Location': location_name,
                            'Latitude': lat,
                            'Longitude': lon,
                            'Temperature_2m_C': temp,
                            'Relative_Humidity_%': rh,
                            'Wind_Speed_2m_m/s': ws,
                            'Precipitation_mm': precip,
                        })
                        
                        # Create record for :30 minutes (interpolated/repeated)
                        # Note: NASA POWER provides hourly data, so :30 uses same values
                        time_30 = f"{hour:02d}:30"
                        weather_records.append({
                            'Date': date,
                            'Time': time_30,
                            'DateTime': f"{date} {time_30}",
                            'Location': location_name,
                            'Latitude': lat,
                            'Longitude': lon,
                            'Temperature_2m_C': temp,
                            'Relative_Humidity_%': rh,
                            'Wind_Speed_2m_m/s': ws,
                            'Precipitation_mm': precip,
                        })
                
                if weather_records:
                    print(f"   ✅ {location_name}: {len(weather_records)} records (30-min intervals)")
                    return weather_records
                else:
                    print(f"   ❌ {location_name}: No data available")
                    return []
                    
        except Exception as e:
            print(f"   ❌ {location_name}: Error - {e}")
            return []
    
    def get_landsat8_image(self, lat, lon, date, location_name, cloud_cover_max=50):
        """Get Landsat 8 image from Google Earth Engine"""
        
        try:
            target_date = datetime.strptime(date, '%Y-%m-%d')
            start_date = (target_date - timedelta(days=8)).strftime('%Y-%m-%d')
            end_date = (target_date + timedelta(days=8)).strftime('%Y-%m-%d')
            
            point = ee.Geometry.Point([lon, lat])
            
            collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterBounds(point) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover_max)) \
                .sort('CLOUD_COVER')
            
            count = collection.size().getInfo()
            
            if count == 0:
                return {
                    'status': 'no_data',
                    'error': f'No cloud-free images found',
                    'date': date,
                    'location': location_name,
                }
            
            image = collection.first()
            image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            cloud_cover = image.get('CLOUD_COVER').getInfo()
            
            # Scale and get RGB
            def apply_scale_factors(img):
                optical_bands = img.select('SR_B.').multiply(0.0000275).add(-0.2)
                return img.addBands(optical_bands, None, True)
            
            scaled_image = apply_scale_factors(image)
            rgb = scaled_image.select(['SR_B4', 'SR_B3', 'SR_B2'])
            region = point.buffer(7500).bounds()
            
            url = rgb.getThumbURL({
                'region': region.getInfo(),
                'dimensions': 512,
                'format': 'png',
                'min': 0,
                'max': 0.3
            })
            
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                filename = f"{location_name}_{image_date}.png"
                filepath = self.image_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                file_size = filepath.stat().st_size / 1024
                
                print(f"   ✅ {location_name}: Satellite image ({cloud_cover:.1f}% clouds, {file_size:.1f} KB)")
                
                return {
                    'status': 'success',
                    'filename': filename,
                    'file_size_kb': file_size,
                    'date': date,
                    'actual_image_date': image_date,
                    'location': location_name,
                    'cloud_cover': cloud_cover,
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
    
    def extract_30min_data(self, locations, dates, cloud_cover_max=70):
        """
        Extract both satellite imagery and 30-minute interval weather data
        
        Args:
            locations: Dict of location names and coordinates
            dates: List of date strings ['YYYY-MM-DD', ...]
            cloud_cover_max: Maximum cloud cover for satellite images
        """
        
        print("=" * 70)
        print("⏱️  30-MINUTE INTERVAL DATA EXTRACTION")
        print("   Satellite Images: Google Earth Engine (Landsat 8)")
        print("   Weather Data: NASA POWER API (Hourly → 30-min intervals)")
        print("=" * 70)
        print(f"📍 Locations: {len(locations)}")
        print(f"📅 Dates: {len(dates)}")
        print(f"⏱️  Time points per day: 48 (every 30 minutes)")
        print(f"📊 Total weather samples: {len(locations) * len(dates) * 48}")
        print(f"🛰️  Satellite images: {len(locations) * len(dates)}")
        print(f"☁️  Max cloud cover: {cloud_cover_max}%")
        print("=" * 70)
        print()
        
        start_time = time.time()
        
        for idx, (location_name, coords) in enumerate(locations.items(), 1):
            print(f"\n[{idx}/{len(locations)}] 📍 {location_name} ({coords['lat']:.4f}°N, {coords['lon']:.4f}°E)")
            print("-" * 70)
            
            for date in dates:
                # Get satellite image (one per day)
                print(f"🛰️  Satellite: {date}...")
                image_result = self.get_landsat8_image(
                    lat=coords['lat'],
                    lon=coords['lon'],
                    date=date,
                    location_name=location_name,
                    cloud_cover_max=cloud_cover_max
                )
                self.results.append(image_result)
                
                # Get 30-minute interval weather data (48 per day)
                print(f"☁️  Weather (30-min): {date}...")
                weather_records = self.get_hourly_weather_data(
                    lat=coords['lat'],
                    lon=coords['lon'],
                    date=date,
                    location_name=location_name
                )
                
                if weather_records:
                    self.weather_data.extend(weather_records)
                
                # Delay
                time.sleep(2)
            
            print()
        
        # Save results
        self.save_results()
        
        # Print summary
        elapsed = time.time() - start_time
        self.print_summary(elapsed)
    
    def save_results(self):
        """Save extraction results to CSV files"""
        
        # Save satellite metadata
        satellite_df = pd.DataFrame(self.results)
        satellite_file = self.data_dir / 'satellite_images_30min_metadata.csv'
        satellite_df.to_csv(satellite_file, index=False)
        print(f"\n💾 Satellite metadata: {satellite_file}")
        
        # Save weather data
        weather_df = pd.DataFrame(self.weather_data)
        weather_file = self.data_dir / 'weather_data_30min_intervals.csv'
        weather_df.to_csv(weather_file, index=False)
        print(f"💾 Weather data: {weather_file}")
    
    def print_summary(self, elapsed_time):
        """Print extraction summary"""
        
        satellite_df = pd.DataFrame(self.results)
        weather_df = pd.DataFrame(self.weather_data)
        
        # Satellite summary
        total_sat = len(satellite_df)
        successful_sat = len(satellite_df[satellite_df['status'] == 'success'])
        success_rate_sat = (successful_sat / total_sat * 100) if total_sat > 0 else 0
        
        # Weather summary
        total_weather = len(weather_df)
        
        print("\n" + "=" * 70)
        print("📊 EXTRACTION SUMMARY")
        print("=" * 70)
        
        print(f"\n🛰️  SATELLITE IMAGES:")
        print(f"   ✅ Successful: {successful_sat}/{total_sat} ({success_rate_sat:.1f}%)")
        
        if successful_sat > 0:
            avg_cloud = satellite_df[satellite_df['status'] == 'success']['cloud_cover'].mean()
            print(f"   ☁️  Avg cloud cover: {avg_cloud:.1f}%")
        
        print(f"\n☁️  WEATHER DATA (30-min intervals):")
        print(f"   ✅ Total records: {total_weather}")
        print(f"   ⏱️  Time resolution: 30 minutes")
        print(f"   📊 Parameters: Temperature, Humidity, Wind, Precipitation")
        
        if total_weather > 0:
            unique_dates = weather_df['Date'].nunique()
            unique_locations = weather_df['Location'].nunique()
            print(f"   📅 Dates covered: {unique_dates}")
            print(f"   📍 Locations: {unique_locations}")
            print(f"   ⏱️  Records per location-date: ~48 (every 30 min)")
        
        print(f"\n⏱️  PERFORMANCE:")
        print(f"   Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        
        print("=" * 70)
        
        print("\n📂 OUTPUT FILES:")
        print(f"   🛰️  {self.image_dir} - Satellite images")
        print(f"   📊 {self.data_dir}/satellite_images_30min_metadata.csv")
        print(f"   📊 {self.data_dir}/weather_data_30min_intervals.csv")
        
        print("\n✅ Extraction complete!")


def main():
    """Main execution function"""
    
    print("\n" + "=" * 70)
    print("⏱️  30-MINUTE INTERVAL DATA EXTRACTOR")
    print("   Google Earth Engine + NASA POWER API")
    print("=" * 70)
    
    # Configuration
    locations = {
        'Dhaka': {'lat': 23.8103, 'lon': 90.4125},
        'Chittagong': {'lat': 22.3569, 'lon': 91.7832},
    }
    
    # Use 2 dates from dry season for better satellite success
    dates = ['2023-01-15', '2023-01-20']
    
    print(f"\n📍 Locations: {', '.join(locations.keys())}")
    print(f"📅 Dates: {', '.join(dates)}")
    print(f"⏱️  30-minute intervals: 48 per day")
    print(f"📊 Total weather samples: {len(locations)} × {len(dates)} × 48 = {len(locations) * len(dates) * 48}")
    print(f"🛰️  Satellite images: {len(locations)} × {len(dates)} = {len(locations) * len(dates)}")
    
    print("\n🚀 Starting extraction...")
    
    # Create extractor and run
    extractor = ThirtyMinuteExtractor()
    extractor.extract_30min_data(
        locations=locations,
        dates=dates,
        cloud_cover_max=70
    )
    
    print("\n🎉 All done! Check the output files.")


if __name__ == "__main__":
    main()
