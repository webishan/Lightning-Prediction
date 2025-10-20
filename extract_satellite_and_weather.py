"""
Extract Satellite Images + Weather Data for Lightning Detection
Downloads satellite images and weather data for the same time/location
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from nasa_power_api import NASAPowerAPI
import time
import os
from pathlib import Path
import json

class SatelliteWeatherExtractor:
    """
    Extract both satellite images and weather data for the same time periods
    """
    
    def __init__(self):
        self.api = NASAPowerAPI()
        self.satellite_dir = "satellite_images"
        self.data_dir = "weather_data"
        
        # Create directories
        Path(self.satellite_dir).mkdir(exist_ok=True)
        Path(self.data_dir).mkdir(exist_ok=True)
    
    def get_nasa_earth_image(self, latitude, longitude, date_str, dim=0.15):
        """
        Download satellite image from NASA Earth API
        
        Parameters:
        -----------
        latitude : float
            Latitude of location
        longitude : float
            Longitude of location
        date_str : str
            Date in YYYY-MM-DD format
        dim : float
            Width/height of image in degrees (0.15 = ~15km)
        
        Returns:
        --------
        dict : Information about downloaded image
        """
        # NASA Earth API endpoint
        # Note: This API provides Landsat 8 imagery (free, no API key needed for limited use)
        url = "https://api.nasa.gov/planetary/earth/imagery"
        
        params = {
            'lon': longitude,
            'lat': latitude,
            'date': date_str,
            'dim': dim,
            'api_key': 'DEMO_KEY'  # Use DEMO_KEY for testing (limited to 30 requests/hour)
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                # Save image
                filename = f"{self.satellite_dir}/sat_{latitude}_{longitude}_{date_str.replace('-', '')}.png"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                return {
                    'status': 'success',
                    'filename': filename,
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
    
    def extract_combined_data(self, locations, start_date, end_date, 
                             download_images=True):
        """
        Extract both satellite images and weather data for multiple locations
        
        Parameters:
        -----------
        locations : dict
            Dictionary of {name: {'lat': lat, 'lon': lon}}
        start_date : str
            Start date in YYYYMMDD format
        end_date : str
            End date in YYYYMMDD format
        download_images : bool
            Whether to download satellite images (can be slow)
        
        Returns:
        --------
        tuple : (weather_df, image_info_df)
        """
        
        print("=" * 90)
        print("🛰️  SATELLITE IMAGES + WEATHER DATA EXTRACTION")
        print("=" * 90)
        
        # Convert dates
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        days = (end_dt - start_dt).days + 1
        
        print(f"\n📅 Date Range: {start_dt.date()} to {end_dt.date()} ({days} days)")
        print(f"📍 Locations: {len(locations)}")
        print(f"🎯 Total Samples: {len(locations) * days}")
        print(f"🛰️  Satellite Images: {'Yes' if download_images else 'No (metadata only)'}")
        
        # Extract weather data first (faster)
        print("\n" + "=" * 90)
        print("📊 STEP 1: EXTRACTING WEATHER DATA")
        print("=" * 90)
        
        all_weather_data = []
        
        for idx, (location_name, coords) in enumerate(locations.items(), 1):
            print(f"\n[{idx}/{len(locations)}] 📍 {location_name}")
            print(f"    Coordinates: {coords['lat']:.4f}°N, {coords['lon']:.4f}°E")
            
            try:
                # Get weather data
                data = self.api.get_daily_data(
                    latitude=coords['lat'],
                    longitude=coords['lon'],
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data:
                    df = self.api.convert_to_dataframe(data)
                    if df is not None:
                        df['Location'] = location_name
                        df['Latitude'] = coords['lat']
                        df['Longitude'] = coords['lon']
                        all_weather_data.append(df)
                        print(f"    ✓ Weather data: {len(df)} records")
                    else:
                        print(f"    ✗ Failed to convert weather data")
                else:
                    print(f"    ✗ No weather data received")
                    
            except Exception as e:
                print(f"    ✗ Error: {e}")
            
            # Respectful delay
            if idx < len(locations):
                time.sleep(1)
        
        # Combine weather data
        if all_weather_data:
            weather_df = pd.concat(all_weather_data, ignore_index=False)
            weather_df.reset_index(inplace=True)
            weather_df.rename(columns={'index': 'Date'}, inplace=True)
            
            # Reorder columns
            cols = ['Date', 'Location', 'Latitude', 'Longitude'] + \
                   [col for col in weather_df.columns if col not in 
                    ['Date', 'Location', 'Latitude', 'Longitude']]
            weather_df = weather_df[cols]
            
            print(f"\n✅ Weather data combined: {len(weather_df)} total records")
        else:
            weather_df = None
            print("\n❌ No weather data extracted")
        
        # Download satellite images if requested
        image_info = []
        
        if download_images:
            print("\n" + "=" * 90)
            print("🛰️  STEP 2: DOWNLOADING SATELLITE IMAGES")
            print("=" * 90)
            print("\n⚠️  Note: Using DEMO_KEY (limited to 30 requests/hour)")
            print("    For production, get free API key from: https://api.nasa.gov/")
            
            # Select specific dates to download (not all days to avoid rate limits)
            date_list = pd.date_range(start_dt, end_dt, freq='D')
            
            # If more than 5 dates, sample evenly
            if len(date_list) > 5:
                step = len(date_list) // 5
                selected_dates = [date_list[i] for i in range(0, len(date_list), step)][:5]
            else:
                selected_dates = date_list
            
            print(f"\n📸 Downloading images for {len(selected_dates)} dates (sampled from {days} days)")
            print(f"    Dates: {[d.strftime('%Y-%m-%d') for d in selected_dates]}")
            
            total_images = len(locations) * len(selected_dates)
            downloaded = 0
            
            for date in selected_dates:
                date_str = date.strftime('%Y-%m-%d')
                print(f"\n📅 Date: {date_str}")
                
                for location_name, coords in locations.items():
                    print(f"  📍 {location_name}...", end=' ')
                    
                    result = self.get_nasa_earth_image(
                        latitude=coords['lat'],
                        longitude=coords['lon'],
                        date_str=date_str
                    )
                    
                    result['location'] = location_name
                    image_info.append(result)
                    
                    if result['status'] == 'success':
                        downloaded += 1
                        print(f"✓ Downloaded")
                    else:
                        print(f"✗ {result.get('error', 'Failed')}")
                    
                    # Delay to respect rate limits (DEMO_KEY = 30/hour = 1 per 2 seconds)
                    time.sleep(2)
            
            print(f"\n✅ Downloaded {downloaded}/{total_images} satellite images")
        
        else:
            print("\n⏭️  STEP 2: SKIPPED (download_images=False)")
        
        # Create image info dataframe
        if image_info:
            image_df = pd.DataFrame(image_info)
        else:
            image_df = pd.DataFrame()
        
        return weather_df, image_df


def extract_20_samples_with_images():
    """
    Extract exactly 20 samples with both satellite images and weather data
    Strategy: 4 locations × 5 time points = 20 samples
    """
    
    extractor = SatelliteWeatherExtractor()
    
    # Select 4 strategic locations across Bangladesh
    locations = {
        'Dhaka': {'lat': 23.8103, 'lon': 90.4125},          # Capital (central)
        'Chittagong': {'lat': 22.3569, 'lon': 91.7832},     # Coastal (high lightning risk)
        'Sylhet': {'lat': 24.8949, 'lon': 91.8687},         # Northeast (high rainfall)
        'Rangpur': {'lat': 25.7439, 'lon': 89.2752},        # North (lower risk)
    }
    
    # Select 5 specific dates in August 2024 (spread across the month)
    # These dates ensure we get 4 locations × 5 dates = 20 samples
    dates = [
        '2024-08-05',
        '2024-08-10',
        '2024-08-15',
        '2024-08-20',
        '2024-08-25',
    ]
    
    # Convert to start/end for weather API
    start_date = '20240805'
    end_date = '20240825'
    
    print("=" * 90)
    print("🎯 TARGET: 20 SAMPLES (4 locations × 5 time points)")
    print("=" * 90)
    
    # Extract data
    weather_df, image_df = extractor.extract_combined_data(
        locations=locations,
        start_date=start_date,
        end_date=end_date,
        download_images=True  # Set to True to download satellite images
    )
    
    # Filter weather data to exact dates we want
    if weather_df is not None:
        weather_df['Date'] = pd.to_datetime(weather_df['Date'])
        target_dates = pd.to_datetime(dates)
        weather_df_filtered = weather_df[weather_df['Date'].isin(target_dates)]
        
        # Save weather data
        weather_filename = f"{extractor.data_dir}/weather_data_20_samples.csv"
        weather_df_filtered.to_csv(weather_filename, index=False)
        
        print("\n" + "=" * 90)
        print("💾 SAVING RESULTS")
        print("=" * 90)
        print(f"\n✅ Weather Data:")
        print(f"   File: {weather_filename}")
        print(f"   Records: {len(weather_df_filtered)}")
        print(f"   Columns: {weather_df_filtered.shape[1]}")
        
        # Display sample
        print(f"\n📊 Weather Data Preview:")
        print(weather_df_filtered[['Date', 'Location', 'Temperature_2m_C', 
                                   'Precipitation_mm', 'Relative_Humidity_%']].head(10))
    
    # Save image info
    if not image_df.empty:
        image_filename = f"{extractor.data_dir}/satellite_images_info.csv"
        image_df.to_csv(image_filename, index=False)
        
        print(f"\n✅ Satellite Image Info:")
        print(f"   File: {image_filename}")
        print(f"   Records: {len(image_df)}")
        
        successful = len(image_df[image_df['status'] == 'success'])
        print(f"   Successfully downloaded: {successful}")
        
        if successful > 0:
            print(f"   Images saved in: {extractor.satellite_dir}/")
            print(f"\n📸 Sample images:")
            for _, row in image_df[image_df['status'] == 'success'].head(5).iterrows():
                print(f"      • {row.get('filename', 'N/A')}")
    
    print("\n" + "=" * 90)
    print("✅ EXTRACTION COMPLETE!")
    print("=" * 90)
    
    print(f"""
📁 OUTPUT FILES:
   1. {weather_filename} - Weather data (CSV)
   2. {image_filename} - Image metadata (CSV)
   3. {extractor.satellite_dir}/ - Satellite images (PNG)

📊 SUMMARY:
   • Weather samples: {len(weather_df_filtered) if weather_df is not None else 0}
   • Satellite images: {successful if not image_df.empty else 0}
   • Locations: {len(locations)}
   • Time points: {len(dates)}

🎯 Ready for analysis and ML model training!
    """)
    
    return weather_df_filtered, image_df


def extract_custom_data():
    """
    Custom extraction - modify parameters as needed
    """
    
    extractor = SatelliteWeatherExtractor()
    
    # Define your locations
    locations = {
        'Location1': {'lat': 23.8103, 'lon': 90.4125},
        'Location2': {'lat': 22.3569, 'lon': 91.7832},
        # Add more locations...
    }
    
    # Define date range
    start_date = '20240801'  # YYYYMMDD
    end_date = '20240810'    # YYYYMMDD
    
    # Extract
    weather_df, image_df = extractor.extract_combined_data(
        locations=locations,
        start_date=start_date,
        end_date=end_date,
        download_images=False  # Set to True to download images
    )
    
    # Save
    if weather_df is not None:
        weather_df.to_csv(f"{extractor.data_dir}/custom_weather_data.csv", index=False)
        print(f"✅ Saved: custom_weather_data.csv")
    
    if not image_df.empty:
        image_df.to_csv(f"{extractor.data_dir}/custom_image_info.csv", index=False)
        print(f"✅ Saved: custom_image_info.csv")
    
    return weather_df, image_df


if __name__ == "__main__":
    print("\n🛰️  SATELLITE + WEATHER DATA EXTRACTOR")
    print("=" * 90)
    print("\nChoose extraction mode:")
    print("  1. Extract 20 samples (4 locations × 5 dates)")
    print("  2. Custom extraction (modify code)")
    print("\n⚠️  NOTE: Satellite image download uses DEMO_KEY")
    print("     Limited to 30 requests/hour")
    print("     Get your free API key at: https://api.nasa.gov/")
    print("=" * 90)
    
    # Run the 20-sample extraction
    extract_20_samples_with_images()
