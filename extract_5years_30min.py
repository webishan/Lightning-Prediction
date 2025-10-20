"""
5-Year Historical Data Extraction (2019-2023)
Extracts weather data at 30-minute intervals for multiple years
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


class FiveYearExtractor:
    """
    Extracts 5 years of historical weather data (2019-2023)
    - NASA POWER API: Hourly weather data (converted to 30-min intervals)
    - Google Earth Engine: Landsat 8 imagery (selected dates only)
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
        self.image_dir = self.base_dir / 'satellite_images_5years'
        self.data_dir = self.base_dir / 'weather_data_5years'
        self.image_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        self.satellite_results = []
        self.extraction_log = []
    
    def get_hourly_weather_batch(self, lat, lon, year, location_name):
        """
        Get hourly weather data for entire year
        
        Args:
            lat: Latitude
            lon: Longitude
            year: Year (e.g., 2019)
            location_name: Name of location
        
        Returns:
            pd.DataFrame: Weather data with 30-min intervals
        """
        
        try:
            # Get data for entire year in chunks (monthly)
            all_records = []
            
            for month in range(1, 13):
                # Determine days in month
                if month in [1, 3, 5, 7, 8, 10, 12]:
                    days = 31
                elif month in [4, 6, 9, 11]:
                    days = 30
                else:  # February
                    days = 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
                
                start_date = f"{year}{month:02d}01"
                end_date = f"{year}{month:02d}{days}"
                
                print(f"      Month {month:02d}: {start_date} to {end_date}...", end=' ')
                
                try:
                    # Get hourly data
                    data = self.nasa_api.get_hourly_data(
                        latitude=lat,
                        longitude=lon,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if data and 'properties' in data and 'parameter' in data['properties']:
                        params = data['properties']['parameter']
                        
                        # Process each day in the month
                        for day in range(1, days + 1):
                            date_str = f"{year}-{month:02d}-{day:02d}"
                            date_formatted = f"{year}{month:02d}{day:02d}"
                            
                            # Process 24 hours
                            for hour in range(24):
                                hour_key = f"{date_formatted}{hour:02d}"
                                
                                # Get values
                                temp = params.get('T2M', {}).get(hour_key)
                                rh = params.get('RH2M', {}).get(hour_key)
                                ws = params.get('WS2M', {}).get(hour_key)
                                precip = params.get('PRECTOTCORR', {}).get(hour_key)
                                solar = params.get('ALLSKY_SFC_SW_DWN', {}).get(hour_key)
                                
                                if temp is not None:
                                    # Create records for :00 and :30
                                    for minute in [0, 30]:
                                        time_str = f"{hour:02d}:{minute:02d}"
                                        all_records.append({
                                            'Date': date_str,
                                            'Time': time_str,
                                            'DateTime': f"{date_str} {time_str}",
                                            'Location': location_name,
                                            'Latitude': lat,
                                            'Longitude': lon,
                                            'Temperature_2m_C': temp,
                                            'Relative_Humidity_%': rh,
                                            'Wind_Speed_2m_m/s': ws,
                                            'Precipitation_mm': precip,
                                            'Solar_Radiation_kWh/m2': solar,
                                        })
                        
                        print(f"✅ {len(all_records)} records")
                        
                    else:
                        print("❌ No data")
                    
                    # Small delay between months
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    self.extraction_log.append({
                        'Location': location_name,
                        'Year': year,
                        'Month': month,
                        'Status': 'Error',
                        'Error': str(e)
                    })
            
            if all_records:
                df = pd.DataFrame(all_records)
                print(f"   ✅ {location_name} {year}: {len(df)} total records (30-min intervals)")
                
                # Save yearly file
                yearly_file = self.data_dir / f"{location_name}_{year}_30min.csv"
                df.to_csv(yearly_file, index=False)
                print(f"   💾 Saved: {yearly_file.name}")
                
                self.extraction_log.append({
                    'Location': location_name,
                    'Year': year,
                    'Records': len(df),
                    'Status': 'Success',
                    'File': yearly_file.name
                })
                
                return df
            else:
                print(f"   ❌ {location_name} {year}: No data retrieved")
                self.extraction_log.append({
                    'Location': location_name,
                    'Year': year,
                    'Records': 0,
                    'Status': 'Failed',
                    'Error': 'No data'
                })
                return None
                
        except Exception as e:
            print(f"   ❌ {location_name} {year}: Error - {e}")
            self.extraction_log.append({
                'Location': location_name,
                'Year': year,
                'Records': 0,
                'Status': 'Error',
                'Error': str(e)
            })
            return None
    
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
                    'error': f'No images <{cloud_cover_max}% clouds',
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
                
                print(f"      ✅ {location_name} {date}: Image saved ({cloud_cover:.1f}% clouds)")
                
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
    
    def extract_5years_data(self, locations, years, sample_dates_per_year=4):
        """
        Extract 5 years of weather data + selected satellite images
        
        Args:
            locations: Dict of location names and coordinates
            years: List of years [2019, 2020, 2021, 2022, 2023]
            sample_dates_per_year: Number of satellite images per year per location
        """
        
        print("=" * 70)
        print("📅 5-YEAR HISTORICAL DATA EXTRACTION (2019-2023)")
        print("   Weather Data: NASA POWER API (30-min intervals)")
        print("   Satellite Images: Google Earth Engine (Selected dates)")
        print("=" * 70)
        print(f"📍 Locations: {len(locations)}")
        print(f"📅 Years: {len(years)}")
        print(f"⏱️  Expected weather records per location per year: ~17,520 (365 days × 48)")
        print(f"📊 Total expected weather records: {len(locations) * len(years) * 365 * 48:,}")
        print(f"🛰️  Satellite images: {len(locations) * len(years) * sample_dates_per_year} (sampled)")
        print("=" * 70)
        print()
        
        start_time = time.time()
        total_records = 0
        
        for idx, (location_name, coords) in enumerate(locations.items(), 1):
            print(f"\n{'='*70}")
            print(f"[{idx}/{len(locations)}] 📍 {location_name} ({coords['lat']:.4f}°N, {coords['lon']:.4f}°E)")
            print(f"{'='*70}\n")
            
            for year in years:
                print(f"   📅 Year {year}:")
                
                # Get weather data for entire year
                weather_df = self.get_hourly_weather_batch(
                    lat=coords['lat'],
                    lon=coords['lon'],
                    year=year,
                    location_name=location_name
                )
                
                if weather_df is not None:
                    total_records += len(weather_df)
                
                # Get sample satellite images (quarterly)
                # Dry season months for better cloud-free images
                sample_dates = [
                    f"{year}-01-15",  # January (dry)
                    f"{year}-04-15",  # April (transition)
                    f"{year}-07-15",  # July (monsoon)
                    f"{year}-10-15",  # October (transition)
                ][:sample_dates_per_year]
                
                print(f"\n   🛰️  Satellite Images (sampled):")
                for date in sample_dates:
                    result = self.get_landsat8_image(
                        lat=coords['lat'],
                        lon=coords['lon'],
                        date=date,
                        location_name=location_name,
                        cloud_cover_max=70
                    )
                    self.satellite_results.append(result)
                    time.sleep(2)
                
                print()
        
        # Save metadata
        self.save_metadata()
        
        # Print summary
        elapsed = time.time() - start_time
        self.print_summary(elapsed, total_records)
    
    def save_metadata(self):
        """Save extraction metadata"""
        
        # Save satellite metadata
        if self.satellite_results:
            satellite_df = pd.DataFrame(self.satellite_results)
            satellite_file = self.data_dir / 'satellite_images_5years_metadata.csv'
            satellite_df.to_csv(satellite_file, index=False)
            print(f"\n💾 Satellite metadata: {satellite_file}")
        
        # Save extraction log
        if self.extraction_log:
            log_df = pd.DataFrame(self.extraction_log)
            log_file = self.data_dir / 'extraction_log_5years.csv'
            log_df.to_csv(log_file, index=False)
            print(f"💾 Extraction log: {log_file}")
    
    def print_summary(self, elapsed_time, total_records):
        """Print extraction summary"""
        
        print("\n" + "=" * 70)
        print("📊 5-YEAR EXTRACTION SUMMARY")
        print("=" * 70)
        
        print(f"\n☁️  WEATHER DATA (30-min intervals):")
        print(f"   ✅ Total records extracted: {total_records:,}")
        print(f"   ⏱️  Time resolution: 30 minutes")
        print(f"   📊 Parameters: Temp, Humidity, Wind, Precip, Solar Radiation")
        
        # Calculate success rate from log
        success_count = sum(1 for log in self.extraction_log if log['Status'] == 'Success')
        total_attempts = len(self.extraction_log)
        
        if total_attempts > 0:
            success_rate = (success_count / total_attempts) * 100
            print(f"   ✅ Success rate: {success_count}/{total_attempts} ({success_rate:.1f}%)")
        
        print(f"\n🛰️  SATELLITE IMAGES:")
        if self.satellite_results:
            satellite_df = pd.DataFrame(self.satellite_results)
            total_sat = len(satellite_df)
            successful_sat = len(satellite_df[satellite_df['status'] == 'success'])
            success_rate_sat = (successful_sat / total_sat * 100) if total_sat > 0 else 0
            
            print(f"   ✅ Successful: {successful_sat}/{total_sat} ({success_rate_sat:.1f}%)")
            
            if successful_sat > 0:
                avg_cloud = satellite_df[satellite_df['status'] == 'success']['cloud_cover'].mean()
                print(f"   ☁️  Avg cloud cover: {avg_cloud:.1f}%")
        
        print(f"\n⏱️  PERFORMANCE:")
        print(f"   Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        if total_records > 0:
            print(f"   Records per second: {total_records/elapsed_time:.1f}")
        
        print("=" * 70)
        
        print("\n📂 OUTPUT FILES:")
        print(f"   📊 {self.data_dir}/ - Individual yearly CSV files")
        print(f"   🛰️  {self.image_dir}/ - Satellite images")
        print(f"   📋 extraction_log_5years.csv - Detailed extraction log")
        
        print("\n✅ Extraction complete!")
        print(f"\n💡 TIP: Each location has separate yearly files:")
        print(f"   Example: {self.data_dir}/Dhaka_2019_30min.csv")
        print(f"            {self.data_dir}/Dhaka_2020_30min.csv")
        print(f"   You can combine them or analyze separately")


def main():
    """Main execution function"""
    
    print("\n" + "=" * 70)
    print("📅 5-YEAR HISTORICAL DATA EXTRACTOR (2019-2023)")
    print("   NASA POWER API + Google Earth Engine")
    print("=" * 70)
    
    # Configuration
    locations = {
        'Dhaka': {'lat': 23.8103, 'lon': 90.4125},
        'Chittagong': {'lat': 22.3569, 'lon': 91.7832},
    }
    
    years = [2019, 2020, 2021, 2022, 2023]
    
    print(f"\n📍 Locations: {', '.join(locations.keys())}")
    print(f"📅 Years: {', '.join(map(str, years))}")
    print(f"⏱️  30-minute intervals: 48 per day")
    print(f"📊 Expected weather records: {len(locations)} × {len(years)} years × 365 days × 48 = {len(locations) * len(years) * 365 * 48:,}")
    print(f"🛰️  Satellite images: {len(locations) * len(years) * 4} (quarterly samples)")
    
    print("\n⚠️  NOTE: This will take some time! Large dataset.")
    print("   Estimated time: 20-30 minutes")
    print("   Files will be saved as: Location_Year_30min.csv")
    
    input("\nPress Enter to start extraction...")
    
    # Create extractor and run
    extractor = FiveYearExtractor()
    extractor.extract_5years_data(
        locations=locations,
        years=years,
        sample_dates_per_year=4  # 4 satellite images per year (quarterly)
    )
    
    print("\n🎉 All done! Check the weather_data_5years/ directory.")


if __name__ == "__main__":
    main()
