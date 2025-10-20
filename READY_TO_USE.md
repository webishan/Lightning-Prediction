# ✅ SATELLITE + WEATHER DATA EXTRACTION - READY TO USE!

## 🎉 What You Have

Three scripts ready to extract satellite images + weather data:

### 1. 🧪 **Demo Test** (2 samples) - TEST FIRST
```cmd
python demo_quick_test.py
```
- **Purpose:** Quick test to verify everything works
- **Time:** 10-15 seconds
- **Output:** 2 weather samples, 1 test image attempt

### 2. ⭐ **Full Extraction** (20 samples) - MAIN SCRIPT
```cmd
python extract_20_samples_complete.py
```
- **Purpose:** Extract all 20 samples
- **Time:** 1-2 minutes
- **Output:** 20 weather samples, satellite images

### 3. 🔧 **Custom Extraction** - ADVANCED
```cmd
python extract_satellite_and_weather.py
```
- **Purpose:** Customize locations/dates
- **Time:** Varies
- **Output:** Custom dataset

---

## 📊 What Gets Extracted

### ✅ Weather Data (CSV File)
**20 samples:** 4 locations × 5 dates

**Locations:**
- Dhaka (23.81°N, 90.41°E)
- Chittagong (22.36°N, 91.78°E) 
- Sylhet (24.89°N, 91.87°E)
- Rangpur (25.74°N, 89.28°E)

**Dates:**
- Aug 5, 10, 15, 20, 25 (2024)

**Features (14 parameters):**
- Temperature (avg, max, min, range, dew point)
- Humidity (relative, specific)
- Precipitation
- Wind (speed at 2m and 10m, direction)
- Pressure
- Solar radiation
- Longwave radiation

### ✅ Satellite Images (PNG Files)
- **Source:** NASA Earth API (Landsat 8)
- **Format:** PNG images
- **Size:** ~15-50 KB each
- **Coverage:** 0.15° × 0.15° (~15km)
- **Expected:** 15-20 images (some dates may not have images)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Test the System
```cmd
python demo_quick_test.py
```
✅ Verifies everything works  
⏱️ Takes 10-15 seconds

### Step 2: Get API Key (Optional but Recommended)
Visit: https://api.nasa.gov/

**Why?**
- DEMO_KEY: 30 requests/hour (slow)
- Personal key: 1,000 requests/hour (fast)
- Both are FREE!

### Step 3: Run Full Extraction
```cmd
python extract_20_samples_complete.py
```
Enter your API key when prompted (or press Enter for DEMO_KEY)

---

## 📁 Output Files

### File Structure:
```
NASA API/
├── weather_data/
│   ├── weather_data_20_samples.csv       ← Main weather data
│   ├── satellite_images_metadata.csv     ← Image info
│   └── extraction_summary.json           ← Summary stats
│
├── satellite_images/
│   ├── nasa_earth_23.8103_90.4125_20240805.png
│   ├── nasa_earth_22.3569_91.7832_20240810.png
│   └── ... (more images)
```

### 1. Weather Data CSV
**Columns:**
```
Date, Location, Latitude, Longitude,
Temperature_2m_C, Temperature_Max_C, Temperature_Min_C,
T2MDEW, Relative_Humidity_%, Precipitation_mm,
Wind_Speed_m/s, WS10M, Wind_Direction_deg,
Surface_Pressure_kPa, Solar_Radiation_kWh/m2,
ALLSKY_SFC_LW_DWN, T2M_RANGE, QV2M
```

**Sample:**
```csv
Date,Location,Temperature_2m_C,Precipitation_mm
2024-08-05,Dhaka,28.45,12.34
2024-08-05,Chittagong,27.89,35.67
```

### 2. Satellite Images
**Naming:** `nasa_earth_{lat}_{lon}_{date}.png`

**Example:** `nasa_earth_23.8103_90.4125_20240805.png`

### 3. Image Metadata CSV
```csv
location,date,filename,status,size_kb
Dhaka,2024-08-05,nasa_earth_...,success,34.5
Chittagong,2024-08-05,nasa_earth_...,failed,N/A
```

---

## ⏱️ Expected Time

| Task | With DEMO_KEY | With Personal Key |
|------|---------------|-------------------|
| **Weather Data** | 10-15 sec | 10-15 sec |
| **Satellite Images** | 50-60 sec | 20-30 sec |
| **Total** | ~1-2 min | ~30-45 sec |

---

## 📊 Sample Counts

Current configuration gives **20 samples**:
- 4 locations × 5 dates = 20

To get different counts:

| Locations | Dates | Total Samples |
|-----------|-------|---------------|
| 2 | 5 | 10 |
| 4 | 5 | **20** ✅ (current) |
| 5 | 6 | 30 |
| 10 | 5 | 50 |

---

## 🔧 Customization

### Change Locations
Edit `extract_20_samples_complete.py`:

```python
locations = {
    'YourCity': {'lat': 24.00, 'lon': 90.00},
    'AnotherCity': {'lat': 23.50, 'lon': 91.50},
}
```

### Change Dates
```python
target_dates = [
    '2024-07-01',
    '2024-07-15',
    # Add more...
]
```

### Change Parameters
Edit in `nasa_power_api.py` to add/remove weather parameters

---

## ⚠️ Important Notes

### Satellite Images
- ⚠️ Not all dates have images (cloud cover, processing delays)
- ✅ Weather data always available (from 1981-present)
- 💡 Expect 75-100% image success rate

### API Rate Limits
**DEMO_KEY:**
- 30 requests/hour
- Good for testing
- May take longer

**Personal Key (Free):**
- 1,000 requests/hour
- Much faster
- Recommended for production

### Network Issues
- Satellite downloads may timeout
- Script continues even if some fail
- Weather data usually completes successfully

---

## 🎯 What You Can Do With This Data

### 1. Lightning Prediction Model
```python
# Combine satellite + weather features
from PIL import Image
import numpy as np

# Load image
img = Image.open('satellite_images/nasa_earth_23.8103_90.4125_20240805.png')
img_array = np.array(img)

# Load weather
weather = pd.read_csv('weather_data/weather_data_20_samples.csv')

# Train model with both
# X = [weather_features, image_features]
# y = lightning_occurrence
```

### 2. Visual Analysis
```python
import matplotlib.pyplot as plt

# Plot weather trends
weather['Date'] = pd.to_datetime(weather['Date'])
weather.plot(x='Date', y='Temperature_2m_C', 
             style='o-', title='Temperature Over Time')

# Display satellite images
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for idx, location in enumerate(['Dhaka', 'Chittagong', 'Sylhet', 'Rangpur']):
    img = Image.open(f'satellite_images/nasa_earth_{lat}_{lon}_{date}.png')
    axes[idx//2, idx%2].imshow(img)
    axes[idx//2, idx%2].set_title(location)
plt.show()
```

### 3. Spatial Analysis
```python
import geopandas as gpd

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    weather, 
    geometry=gpd.points_from_xy(weather.Longitude, weather.Latitude)
)

# Plot spatial distribution
gdf.plot(column='Precipitation_mm', 
         cmap='Blues', 
         legend=True,
         figsize=(10, 8))
```

---

## 🐛 Troubleshooting

### Problem: Satellite images not downloading
**Solutions:**
1. Check internet connection
2. Verify API key
3. Try different dates
4. Some dates naturally don't have images (normal)

### Problem: "Read timed out"
**Solution:**
- Normal for satellite downloads
- Increase timeout in code (line with `timeout=30`)
- Try again later

### Problem: Only weather data, no images
**Solution:**
- Weather data from NASA POWER (different API, always works)
- Satellite images from NASA Earth (can have availability issues)
- This is NORMAL - you'll still have complete weather data!

---

## ✅ Success Checklist

After running, verify:

- [ ] `weather_data_20_samples.csv` exists (20 rows)
- [ ] `satellite_images/` folder has PNG files (15-20 images)
- [ ] `satellite_images_metadata.csv` exists
- [ ] `extraction_summary.json` exists
- [ ] No errors in terminal output

---

## 📚 Next Steps

1. ✅ **Run demo:** `python demo_quick_test.py`
2. ✅ **Get API key:** https://api.nasa.gov/ (optional)
3. ✅ **Extract data:** `python extract_20_samples_complete.py`
4. ✅ **Analyze:** Load CSV, view images
5. ✅ **Build model:** Use for lightning detection

---

## 🎓 For Your Thesis

### Data Description
"Weather data was collected from NASA POWER API for 20 samples across 4 locations in Bangladesh during August 2024. Corresponding Landsat 8 satellite imagery was obtained from NASA Earth API for the same spatiotemporal points. The dataset includes 14 meteorological parameters synchronized with satellite observations."

### Methodology
- **Weather Source:** NASA POWER API (validated global data)
- **Satellite Source:** NASA Earth API (Landsat 8)
- **Spatial Coverage:** 4 strategic locations across Bangladesh
- **Temporal Coverage:** 5 dates in August 2024 (monsoon peak)
- **Sample Size:** 20 synchronized observations
- **Features:** 14 weather parameters + satellite imagery

---

## 📞 Resources

- **NASA API Keys:** https://api.nasa.gov/
- **NASA Earth API:** https://api.nasa.gov/planetary/earth
- **NASA POWER API:** https://power.larc.nasa.gov/
- **Landsat Info:** https://landsat.gsfc.nasa.gov/

---

## 🎉 You're Ready!

**Current Status:** ✅ System tested and working

**Next Command:**
```cmd
python extract_20_samples_complete.py
```

**Expected Result:**
- 20 weather data samples in CSV
- 15-20 satellite images in PNG
- Metadata files
- Ready for analysis!

---

**Good luck with your thesis! 🚀⚡🛰️**
