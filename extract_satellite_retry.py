"""
Enhanced Satellite Image Extractor with Retry Logic
Fixes timeout issues with multiple retry attempts and longer timeouts
"""

import requests
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

class RobustSatelliteExtractor:
    def __init__(self, api_key='DEMO_KEY'):
        self.api_key = api_key
        self.satellite_api_url = "https://api.nasa.gov/planetary/earth/imagery"
        
        # Create directories
        self.base_dir = Path(__file__).parent
        self.image_dir = self.base_dir / 'satellite_images'
        self.data_dir = self.base_dir / 'weather_data'
        self.image_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        self.results = []
    
    def download_satellite_image_robust(self, lat, lon, date, location_name, 
                                       max_retries=3, timeout=60):
        """
        Download satellite image with robust retry logic and longer timeout
        
        Args:
            lat: Latitude
            lon: Longitude  
            date: Date in YYYY-MM-DD format
            location_name: Name of location
            max_retries: Number of retry attempts (default 3)
            timeout: Timeout in seconds (default 60)
        """
        params = {
            'lon': lon,
            'lat': lat,
            'date': date,
            'dim': 0.15,  # ~15km coverage
            'api_key': self.api_key
        }
        
        for attempt in range(max_retries):
            try:
                print(f"  📡 Attempt {attempt + 1}/{max_retries}: {location_name} on {date}...", end=" ")
                
                # Make request with longer timeout
                response = requests.get(
                    self.satellite_api_url, 
                    params=params, 
                    timeout=timeout,  # Increased from 30 to 60 seconds
                    stream=True  # Stream large images
                )
                
                if response.status_code == 200:
                    # Save image
                    filename = f"{location_name}_{date}.png"
                    filepath = self.image_dir / filename
                    
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    file_size = filepath.stat().st_size / 1024  # KB
                    print(f"✅ Success ({file_size:.1f} KB)")
                    
                    return {
                        'status': 'success',
                        'filename': filename,
                        'file_size_kb': file_size,
                        'date': date,
                        'location': location_name,
                        'attempts': attempt + 1
                    }
                else:
                    print(f"❌ HTTP {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(5)  # Wait before retry
                        continue
                    
                    return {
                        'status': 'error',
                        'error': f'HTTP {response.status_code}',
                        'date': date,
                        'location': location_name,
                        'attempts': attempt + 1
                    }
                    
            except requests.exceptions.Timeout:
                print(f"⏱️ Timeout")
                if attempt < max_retries - 1:
                    print(f"    ⏳ Waiting 10 seconds before retry...")
                    time.sleep(10)  # Longer wait after timeout
                    continue
                    
                return {
                    'status': 'error',
                    'error': 'Timeout after all retries',
                    'date': date,
                    'location': location_name,
                    'attempts': attempt + 1
                }
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                    
                return {
                    'status': 'error',
                    'error': str(e),
                    'date': date,
                    'location': location_name,
                    'attempts': attempt + 1
                }
        
        return {
            'status': 'error',
            'error': 'Max retries exceeded',
            'date': date,
            'location': location_name,
            'attempts': max_retries
        }
    
    def extract_20_samples_with_retry(self):
        """Extract 20 samples with robust retry logic"""
        
        # Configuration
        locations = {
            'Dhaka': {'lat': 23.8103, 'lon': 90.4125},
            'Chittagong': {'lat': 22.3569, 'lon': 91.7832},
            'Sylhet': {'lat': 24.8949, 'lon': 91.8687},
            'Rangpur': {'lat': 25.7439, 'lon': 89.2752},
        }
        
        target_dates = ['2024-08-05', '2024-08-10', '2024-08-15', '2024-08-20', '2024-08-25']
        
        print("=" * 70)
        print("🛰️  ENHANCED SATELLITE IMAGE EXTRACTOR")
        print("=" * 70)
        print(f"📍 Locations: {len(locations)}")
        print(f"📅 Dates: {len(target_dates)}")
        print(f"📊 Total samples: {len(locations) * len(target_dates)}")
        print(f"⏱️  Timeout: 60 seconds per image")
        print(f"🔄 Max retries: 3 attempts per image")
        print("=" * 70)
        print()
        
        start_time = time.time()
        
        for idx, (location_name, coords) in enumerate(locations.items(), 1):
            print(f"\n[{idx}/{len(locations)}] 📍 {location_name} ({coords['lat']:.4f}°N, {coords['lon']:.4f}°E)")
            print("-" * 70)
            
            for date in target_dates:
                result = self.download_satellite_image_robust(
                    lat=coords['lat'],
                    lon=coords['lon'],
                    date=date,
                    location_name=location_name,
                    max_retries=3,
                    timeout=60
                )
                
                self.results.append(result)
                
                # Delay between requests to be respectful to API
                time.sleep(3)
        
        # Save results
        self.save_results()
        
        # Print summary
        elapsed = time.time() - start_time
        self.print_summary(elapsed)
    
    def save_results(self):
        """Save extraction results to CSV"""
        df = pd.DataFrame(self.results)
        output_file = self.data_dir / 'satellite_images_metadata_retry.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        return df
    
    def print_summary(self, elapsed_time):
        """Print extraction summary"""
        df = pd.DataFrame(self.results)
        
        total = len(df)
        successful = len(df[df['status'] == 'success'])
        failed = len(df[df['status'] == 'error'])
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print("\n" + "=" * 70)
        print("📊 EXTRACTION SUMMARY")
        print("=" * 70)
        print(f"✅ Successful: {successful}/{total} ({success_rate:.1f}%)")
        print(f"❌ Failed: {failed}/{total}")
        print(f"⏱️  Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        
        if successful > 0:
            avg_size = df[df['status'] == 'success']['file_size_kb'].mean()
            total_size = df[df['status'] == 'success']['file_size_kb'].sum()
            print(f"💾 Total data: {total_size:.1f} KB ({total_size/1024:.2f} MB)")
            print(f"📏 Avg image size: {avg_size:.1f} KB")
        
        print("=" * 70)
        
        if successful > 0:
            print(f"\n✅ Images saved in: {self.image_dir}")
        
        if failed > 0:
            print(f"\n⚠️  {failed} images failed to download.")
            print("💡 Tips:")
            print("   • Check your internet connection")
            print("   • Some dates may not have satellite coverage")
            print("   • Try again later or use different dates")
            print("   • Consider getting a personal NASA API key (free)")
            print("     Visit: https://api.nasa.gov/")


def main():
    print("\n" + "=" * 70)
    print("🛰️  ROBUST SATELLITE IMAGE EXTRACTOR")
    print("    With Enhanced Retry Logic & Longer Timeouts")
    print("=" * 70)
    
    # Get API key
    print("\n🔑 NASA API Key Setup:")
    print("   • Press ENTER to use DEMO_KEY (30 requests/hour)")
    print("   • Or paste your API key (1000 requests/hour - free from api.nasa.gov)")
    api_key = input("\nAPI Key [DEMO_KEY]: ").strip()
    
    if not api_key:
        api_key = 'DEMO_KEY'
        print("   Using DEMO_KEY")
    else:
        print("   Using your API key")
    
    # Start extraction
    print("\n🚀 Starting extraction with:")
    print("   • 60-second timeout (2x longer)")
    print("   • 3 retry attempts per image")
    print("   • 10-second wait between retries")
    print("   • 3-second delay between requests")
    
    input("\nPress ENTER to start...")
    
    extractor = RobustSatelliteExtractor(api_key=api_key)
    extractor.extract_20_samples_with_retry()
    
    print("\n✅ Done! Check the satellite_images folder for downloaded images.")


if __name__ == "__main__":
    main()
