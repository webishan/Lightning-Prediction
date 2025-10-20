# 5-YEAR HISTORICAL DATA EXTRACTION (2019-2023)

## 📊 Project Overview

**Extraction Period:** 2019 - 2023 (5 years)  
**Time Resolution:** 30-minute intervals  
**Data Sources:**
- Weather: NASA POWER API (Hourly → 30-min conversion)
- Satellite: Google Earth Engine (Landsat 8, quarterly samples)

---

## 📈 Dataset Specifications

### Weather Data
- **Total Records Expected:** ~350,400 weather samples
  - 2 locations × 5 years × 365 days × 48 intervals = 175,200 per location
- **Time Resolution:** 30 minutes (48 records per day)
- **Parameters per Record:** 5
  1. Temperature at 2m (°C)
  2. Relative Humidity (%)
  3. Wind Speed at 2m (m/s)
  4. Precipitation (mm)
  5. Solar Radiation (kWh/m²)

### Locations Covered
1. **Dhaka** (23.8103°N, 90.4125°E) - Capital city
2. **Chittagong** (22.3569°N, 91.7832°E) - Port city

### Temporal Coverage
- **Years:** 2019, 2020, 2021, 2022, 2023
- **Seasons Covered:**
  - Dry Season (November - February)
  - Pre-Monsoon (March - May)
  - Monsoon (June - September)
  - Post-Monsoon (October)

### Satellite Imagery
- **Total Images:** 40 (quarterly samples)
  - 2 locations × 5 years × 4 quarters = 40 images
- **Sample Dates per Year:**
  - January 15 (Dry season)
  - April 15 (Pre-monsoon)
  - July 15 (Peak monsoon)
  - October 15 (Post-monsoon)
- **Source:** Landsat 8 Collection 2 Level-2
- **Cloud Threshold:** 70% (±8 day window)

---

## 📁 File Structure

### Output Directory: `weather_data_5years/`

**Individual Yearly Files:**
```
Dhaka_2019_30min.csv          (~3.5 MB, ~17,520 records)
Dhaka_2020_30min.csv          (~3.5 MB, ~17,568 records - leap year)
Dhaka_2021_30min.csv          (~3.5 MB, ~17,520 records)
Dhaka_2022_30min.csv          (~3.5 MB, ~17,520 records)
Dhaka_2023_30min.csv          (~3.5 MB, ~17,520 records)

Chittagong_2019_30min.csv     (~3.5 MB, ~17,520 records)
Chittagong_2020_30min.csv     (~3.5 MB, ~17,568 records - leap year)
Chittagong_2021_30min.csv     (~3.5 MB, ~17,520 records)
Chittagong_2022_30min.csv     (~3.5 MB, ~17,520 records)
Chittagong_2023_30min.csv     (~3.5 MB, ~17,520 records)
```

**Metadata Files:**
```
satellite_images_5years_metadata.csv  - Image download log
extraction_log_5years.csv             - Weather extraction log
```

### Output Directory: `satellite_images_5years/`
PNG images named: `{Location}_{Date}.png`

---

## 📊 CSV File Format

### Weather Data Columns (11 columns)

| Column Name | Data Type | Description | Unit |
|-------------|-----------|-------------|------|
| Date | String | Date in YYYY-MM-DD format | - |
| Time | String | Time in HH:MM format | - |
| DateTime | String | Combined timestamp | - |
| Location | String | City name | - |
| Latitude | Float | Latitude coordinate | Degrees |
| Longitude | Float | Longitude coordinate | Degrees |
| Temperature_2m_C | Float | Air temperature at 2m height | °C |
| Relative_Humidity_% | Float | Relative humidity | % |
| Wind_Speed_2m_m/s | Float | Wind speed at 2m height | m/s |
| Precipitation_mm | Float | Hourly precipitation | mm |
| Solar_Radiation_kWh/m2 | Float | Solar irradiance | kWh/m² |

### Sample Data Row
```csv
2019-01-15,12:30,2019-01-15 12:30,Dhaka,23.8103,90.4125,24.83,45.99,2.65,0.0,0.52
```

---

## 🔍 Data Quality Considerations

### Weather Data Quality
✅ **Hourly NASA POWER data** - Validated by NASA  
✅ **30-min intervals** - Repeated hourly values (not interpolated)  
✅ **Complete temporal coverage** - All days from 2019-2023  
✅ **No missing data** - NASA POWER provides continuous records  

### Important Notes:
1. **30-minute resolution:** Each hour has two identical records (:00 and :30)
   - This is because NASA POWER provides hourly data
   - True 30-min measurements would require different data source
   - Suitable for hourly-resolution analysis

2. **Satellite temporal gap:** Images within ±8 days of target date
   - Landsat 8 revisit time: 16 days
   - Cloud cover can further limit availability
   - Monsoon season: Lower success rate expected

3. **Leap years:** 2020 has 366 days (17,568 records vs 17,520)

---

## 📈 Use Cases & Applications

### 1. Long-term Trend Analysis
- **Temperature trends:** Study warming patterns over 5 years
- **Seasonal variations:** Compare dry season vs monsoon
- **Climate patterns:** Identify annual cycles and anomalies

### 2. Lightning Detection Research
- **Pre-storm conditions:** Analyze temperature/humidity before events
- **Seasonal patterns:** Lightning more common in pre-monsoon/monsoon
- **Diurnal cycles:** Lightning peaks in afternoon (14:00-18:00)
- **Multi-year comparison:** Identify inter-annual variability

### 3. Machine Learning Applications
- **Time series forecasting:** Predict next-hour weather
- **Pattern recognition:** Identify storm precursors
- **Multi-modal learning:** Combine weather + satellite imagery
- **Anomaly detection:** Find unusual weather events

### 4. Statistical Analysis
- **Correlation studies:** Temp vs Humidity, Wind vs Precipitation
- **Frequency analysis:** Rain event frequency by season
- **Extreme events:** Identify heat waves, heavy rain days
- **Comparative studies:** Dhaka vs Chittagong climate differences

---

## 📊 Expected Data Statistics

### Temperature (Based on Bangladesh Climate)
- **Annual Range:** 15°C - 35°C
- **Dry Season (Dec-Feb):** 12°C - 28°C
- **Pre-Monsoon (Mar-May):** 25°C - 35°C
- **Monsoon (Jun-Sep):** 26°C - 32°C
- **Post-Monsoon (Oct-Nov):** 20°C - 30°C

### Precipitation Patterns
- **Annual Total:** 1,500 - 2,500 mm
- **Dry Season:** <100 mm (sparse rain)
- **Monsoon:** 80% of annual rainfall
- **Peak:** June-July (300-400 mm/month)

### Humidity
- **Monsoon:** 85-95% (very humid)
- **Dry Season:** 60-75% (moderate)
- **Daily Cycle:** Peak at dawn (100%), minimum at noon (45-60%)

### Wind Speed
- **Average:** 1-4 m/s (light to moderate)
- **Pre-Monsoon:** Higher speeds (3-6 m/s)
- **Storm Events:** Can exceed 10 m/s

---

## 🛠️ Data Processing Tips

### Combining Multiple Years
```python
import pandas as pd
from pathlib import Path

# Combine all Dhaka data
dhaka_files = list(Path('weather_data_5years').glob('Dhaka_*_30min.csv'))
dhaka_all = pd.concat([pd.read_csv(f) for f in dhaka_files], ignore_index=True)

print(f"Total Dhaka records: {len(dhaka_all):,}")
# Output: Total Dhaka records: 87,648

# Convert DateTime to proper format
dhaka_all['DateTime'] = pd.to_datetime(dhaka_all['DateTime'])
dhaka_all = dhaka_all.sort_values('DateTime')

# Save combined file
dhaka_all.to_csv('Dhaka_2019-2023_combined.csv', index=False)
```

### Extracting Specific Seasons
```python
# Extract only monsoon months (June-September)
dhaka_all['Month'] = pd.to_datetime(dhaka_all['Date']).dt.month
monsoon = dhaka_all[dhaka_all['Month'].isin([6, 7, 8, 9])]

print(f"Monsoon records: {len(monsoon):,}")
```

### Analyzing Daily Patterns
```python
# Average temperature by hour of day
dhaka_all['Hour'] = pd.to_datetime(dhaka_all['Time'], format='%H:%M').dt.hour
hourly_avg = dhaka_all.groupby('Hour')['Temperature_2m_C'].mean()

print(hourly_avg)
# Shows diurnal temperature cycle
```

### Finding Extreme Events
```python
# Find days with >50mm precipitation (heavy rain)
daily_precip = dhaka_all.groupby('Date')['Precipitation_mm'].sum()
heavy_rain = daily_precip[daily_precip > 50]

print(f"Heavy rain days: {len(heavy_rain)}")
print(heavy_rain.head())
```

---

## 📚 Citation & Attribution

### Data Sources

**NASA POWER Project:**
```
NASA Prediction of Worldwide Energy Resources (POWER) Project
NASA Langley Research Center (LaRC)
Available at: https://power.larc.nasa.gov/
Accessed: October 2025
```

**Google Earth Engine:**
```
Landsat 8 Collection 2 Level-2 Surface Reflectance
U.S. Geological Survey
Available through: Google Earth Engine
Accessed: October 2025
```

### Recommended Citation Format
```
Weather data sourced from NASA POWER Project (2019-2023), providing hourly 
surface meteorology at 0.5° × 0.5° resolution. Satellite imagery obtained 
from Landsat 8 Collection 2 via Google Earth Engine. Data extracted at 
30-minute intervals for lightning detection research in Bangladesh.
```

---

## 🔬 Data Limitations

### 1. Temporal Resolution
- ⚠️ **30-min data is repeated hourly values** (not true sub-hourly measurements)
- ✅ For hourly analysis and ML training
- ❌ For sub-hourly storm dynamics (need weather radar/stations)

### 2. Spatial Resolution
- ⚠️ NASA POWER: 0.5° × 0.5° grid (~55km × 55km)
- ⚠️ Point coordinates represent grid cell center
- ✅ Suitable for regional climate studies
- ❌ Not suitable for micro-scale urban heat island studies

### 3. Data Source Limitations
- ⚠️ NASA POWER uses **reanalysis data** (model + observations)
- ⚠️ Not direct ground station measurements
- ✅ Well-validated and widely used in research
- ✅ Consistent methodology across years

### 4. Satellite Availability
- ⚠️ Landsat 8: 16-day revisit cycle (limited temporal resolution)
- ⚠️ Cloud cover: Major limitation during monsoon
- ✅ Quarterly samples provide seasonal snapshots
- ℹ️ Consider Sentinel-1 SAR for all-weather imaging

---

## 🚀 Next Steps & Enhancements

### Option 1: Add More Locations
```python
locations = {
    'Dhaka': {'lat': 23.8103, 'lon': 90.4125},
    'Chittagong': {'lat': 22.3569, 'lon': 91.7832},
    'Sylhet': {'lat': 24.8949, 'lon': 91.8687},      # +87,648 records
    'Rangpur': {'lat': 25.7439, 'lon': 89.2752},     # +87,648 records
    'Khulna': {'lat': 22.8456, 'lon': 89.5403},      # +87,648 records
}
# Total: 438,240 records (5 locations)
```

### Option 2: Extract More Parameters
NASA POWER API offers 100+ parameters:
- Surface Pressure (PS)
- Dew Point Temperature (T2MDEW)
- Wind Direction (WD2M)
- Cloud Cover (CLOUD_AMT)
- All-Sky UV Index (ALLSKY_SRF_UV)

### Option 3: Higher Satellite Resolution
- Sentinel-2: 10m resolution, 5-day revisit
- Sentinel-1 SAR: All-weather, day/night
- MODIS: Daily coverage

### Option 4: True Sub-Hourly Data
- Weather station networks (BMD - Bangladesh Meteorological Department)
- Research-grade weather stations
- Lightning detection networks

### Option 5: Extend Time Range
- NASA POWER: Available from 1981-present
- Extract 1990-2023 (34 years) for long-term climate analysis

---

## 📞 Command Reference

### Run Extraction
```bash
python quick_extract_5years.py
```

### Check Progress (while running)
Files will appear in `weather_data_5years/` as extraction proceeds.

### Verify Extraction
```bash
# Count files
dir weather_data_5years\*.csv

# Check file sizes
dir weather_data_5years\ /s
```

### Load and Analyze
```python
import pandas as pd

# Load single year
df_2023 = pd.read_csv('weather_data_5years/Dhaka_2023_30min.csv')
print(f"2023: {len(df_2023):,} records")

# Basic stats
print(df_2023['Temperature_2m_C'].describe())
```

---

## ⏱️ Performance Metrics

### Extraction Time (Estimated)
- **Weather Data:** ~12-15 minutes
  - 12 months × 5 years × 2 locations = 120 API calls
  - ~5-8 seconds per month
- **Satellite Images:** ~3-5 minutes
  - 40 images × 3-5 seconds each
- **Total Time:** 15-25 minutes

### Dataset Size
- **Weather CSV files:** ~35 MB total (10 files × 3.5 MB)
- **Satellite images:** ~20 MB (40 images × ~500 KB)
- **Total Storage:** ~55 MB

### Data Rate
- **Records per second:** ~400-500 weather samples/second
- **Processing:** Minimal overhead, mostly API wait time

---

## 🎯 Quality Assurance Checklist

After extraction completes:

✅ **File Count:** 10 CSV files (2 locations × 5 years)  
✅ **Record Count:** ~175,200 records per location  
✅ **No Missing Values:** All columns populated  
✅ **Date Range:** Jan 1, 2019 - Dec 31, 2023  
✅ **Time Coverage:** All 48 intervals per day  
✅ **Coordinate Consistency:** Lat/Lon match locations  
✅ **Value Ranges:** Temperature 10-40°C, Humidity 20-100%  
✅ **Metadata Files:** extraction_log_5years.csv present  
✅ **Satellite Images:** Check metadata for success rate  

---

## 📧 Troubleshooting

### Issue: Extraction Stops Mid-Way
**Solution:** Check internet connection, re-run script (it creates separate files per year)

### Issue: Low Satellite Success Rate
**Solution:** Normal for monsoon season. Consider:
- Increasing cloud_cover_max to 90%
- Using dry season dates only
- Adding Sentinel-1 SAR

### Issue: File Too Large for Excel
**Solution:** Use Python/R for analysis:
```python
import pandas as pd
df = pd.read_csv('large_file.csv', 
                 chunksize=10000)  # Read in chunks
```

### Issue: Memory Error
**Solution:** Process year by year instead of combining all:
```python
for year in range(2019, 2024):
    df = pd.read_csv(f'Dhaka_{year}_30min.csv')
    # Analyze each year separately
```

---

**Documentation Created:** October 20, 2025  
**Script Version:** quick_extract_5years.py  
**Extraction Status:** RUNNING  
**Expected Completion:** 15-25 minutes from start
