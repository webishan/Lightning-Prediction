"""
Alternative: Extract with NASA POWER API Only
Since GEE requires manual authentication, use this for weather data extraction
"""

from nasa_power_api import NASAPowerAPI
import pandas as pd
from pathlib import Path
import time

def extract_weather_only_20_samples():
    """Extract 20 weather samples without satellite images"""
    
    # Initialize API
    api = NASAPowerAPI()
    
    # Configuration - Same as hybrid script
    locations = {
        'Dhaka': {'lat': 23.8103, 'lon': 90.4125},
        'Chittagong': {'lat': 22.3569, 'lon': 91.7832},
        'Sylhet': {'lat': 24.8949, 'lon': 91.8687},
        'Rangpur': {'lat': 25.7439, 'lon': 89.2752},
    }
    
    dates = ['2024-08-05', '2024-08-10', '2024-08-15', '2024-08-20', '2024-08-25']
    
    print("=" * 70)
    print("☁️  WEATHER DATA EXTRACTION (20 SAMPLES)")
    print("   NASA POWER API - No Satellite Images")
    print("=" * 70)
    print(f"📍 Locations: {len(locations)}")
    print(f"📅 Dates: {len(dates)}")
    print(f"📊 Total samples: {len(locations) * len(dates)}")
    print("=" * 70)
    print()
    
    weather_data = []
    
    for idx, (location_name, coords) in enumerate(locations.items(), 1):
        print(f"[{idx}/{len(locations)}] 📍 {location_name}")
        
        for date in dates:
            try:
                # Get weather data
                data = api.get_daily_data(
                    latitude=coords['lat'],
                    longitude=coords['lon'],
                    start_date=date,
                    end_date=date
                )
                
                if data and 'properties' in data and 'parameter' in data['properties']:
                    params = data['properties']['parameter']
                    
                    weather_record = {
                        'Date': date,
                        'Location': location_name,
                        'Latitude': coords['lat'],
                        'Longitude': coords['lon'],
                        'Temperature_2m_C': params.get('T2M', {}).get(date),
                        'Temperature_Max_C': params.get('T2M_MAX', {}).get(date),
                        'Temperature_Min_C': params.get('T2M_MIN', {}).get(date),
                        'Temperature_Dewpoint_C': params.get('T2MDEW', {}).get(date),
                        'Temperature_Range_C': params.get('T2M_RANGE', {}).get(date),
                        'Relative_Humidity_%': params.get('RH2M', {}).get(date),
                        'Specific_Humidity_g/kg': params.get('QV2M', {}).get(date),
                        'Precipitation_mm': params.get('PRECTOTCORR', {}).get(date),
                        'Wind_Speed_2m_m/s': params.get('WS2M', {}).get(date),
                        'Wind_Speed_10m_m/s': params.get('WS10M', {}).get(date),
                        'Wind_Direction_deg': params.get('WD2M', {}).get(date),
                        'Surface_Pressure_kPa': params.get('PS', {}).get(date),
                        'Solar_Radiation_Shortwave_W/m2': params.get('ALLSKY_SFC_SW_DWN', {}).get(date),
                        'Solar_Radiation_Longwave_W/m2': params.get('ALLSKY_SFC_LW_DWN', {}).get(date),
                    }
                    
                    weather_data.append(weather_record)
                    print(f"   ✅ {date}: Weather data retrieved")
                else:
                    print(f"   ❌ {date}: No data available")
                    
            except Exception as e:
                print(f"   ❌ {date}: Error - {e}")
            
            time.sleep(1)  # Be respectful to API
        
        print()
    
    # Save to CSV
    df = pd.DataFrame(weather_data)
    output_dir = Path(__file__).parent / 'weather_data'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'weather_data_20_samples_only.csv'
    df.to_csv(output_file, index=False)
    
    print("=" * 70)
    print("📊 EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"✅ Extracted: {len(df)} weather samples")
    print(f"💾 Saved to: {output_file}")
    print("=" * 70)
    print()
    print("📈 Data Summary:")
    print(f"   Locations: {df['Location'].nunique()}")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   Parameters: 14 weather variables")
    print()
    print("✅ Ready for analysis!")
    print()
    print("💡 To get satellite images:")
    print("   1. Authenticate with Google Earth Engine")
    print("   2. Run: python extract_hybrid_gee_nasa.py")
    
    return df

if __name__ == "__main__":
    extract_weather_only_20_samples()
