# 🛰️ Satellite Images + Weather Data Extraction Guide

## 🎯 What This Does

Extracts **20 synchronized samples** with both:
1. **Weather data** → Saved in CSV file
2. **Satellite images** → Saved as PNG files

**Configuration:** 4 locations × 5 dates = 20 samples

---

## 📁 Files Created

### Main Scripts

1. **`extract_20_samples_complete.py`** ⭐ **RECOMMENDED**
   - Extracts weather data + satellite images
   - 4 locations × 5 dates = 20 samples
   - Interactive, with progress tracking

2. **`extract_satellite_and_weather.py`**
   - Alternative version with more options
   - Customizable parameters

---

## 🚀 Quick Start

### Step 1: Run the Extraction

```cmd
python extract_20_samples_complete.py
```

### Step 2: Enter API Key (Optional)

```
Enter NASA API key (or press Enter to use DEMO_KEY): [Press Enter]
```

**Note:** DEMO_KEY is limited to 30 requests/hour. For production use, get a free API key at https://api.nasa.gov/

---

## 📊 Output Files

### 1. Weather Data (CSV)
**File:** `weather_data/weather_data_20_samples.csv`

**Contains:**
- Date, Location, Latitude, Longitude
- Temperature (avg, max, min)
- Humidity
- Precipitation
- Wind speed
- Solar radiation
- Pressure
- And more...

**Records:** 20 (4 locations × 5 dates)

### 2. Satellite Images (PNG)
**Folder:** `satellite_images/`

**Format:** `nasa_earth_{lat}_{lon}_{date}.png`

**Example:**
```
nasa_earth_23.8103_90.4125_20240805.png
nasa_earth_22.3569_91.7832_20240810.png
```

### 3. Image Metadata (CSV)
**File:** `weather_data/satellite_images_metadata.csv`

**Contains:**
- Location name
- Date
- Image filename
- Download status
- File size
- Error messages (if any)

### 4. Extraction Summary (JSON)
**File:** `weather_data/extraction_summary.json`

**Contains:**
- Extraction timestamp
- Total samples
- Success/failure counts
- File locations
- Configuration details

---

## 🌍 Locations

| Location | Coordinates | Characteristics |
|----------|-------------|-----------------|
| **Dhaka** | 23.81°N, 90.41°E | Capital, Central |
| **Chittagong** | 22.36°N, 91.78°E | Coastal, High Lightning |
| **Sylhet** | 24.89°N, 91.87°E | Northeast, High Rainfall |
| **Rangpur** | 25.74°N, 89.28°E | North, Lower Risk |

---

## 📅 Time Points (5 Dates)

- **2024-08-05** - Early monsoon
- **2024-08-10** - Peak monsoon
- **2024-08-15** - Mid-monsoon
- **2024-08-20** - Late monsoon
- **2024-08-25** - Monsoon declining

**Why August?** Peak lightning activity in Bangladesh

---

## ⚙️ Configuration

### Locations (4 × 5 = 20 samples)

Edit in the script:
```python
locations = {
    'Location1': {'lat': 23.8103, 'lon': 90.4125},
    'Location2': {'lat': 22.3569, 'lon': 91.7832},
    # Add or modify...
}
```

### Dates

Edit in the script:
```python
target_dates = [
    '2024-08-05',
    '2024-08-10',
    # Add or modify...
]
```

### Sample Count

To get different sample counts:
- **10 samples:** 2 locations × 5 dates
- **20 samples:** 4 locations × 5 dates ✅ (current)
- **30 samples:** 6 locations × 5 dates
- **50 samples:** 10 locations × 5 dates

---

## ⏱️ Execution Time

### With DEMO_KEY
- **Weather data:** ~10-15 seconds (fast)
- **Satellite images:** ~50-60 seconds (20 images × 2.5 sec delay)
- **Total:** ~1-2 minutes

### With Personal API Key
- Can be faster (higher rate limits)
- Up to 1,000 requests/hour

---

## 🛰️ About Satellite Images

### Source
**NASA Earth API** - Landsat 8 imagery
- Resolution: ~30 meters per pixel
- Coverage: Global
- Free to use

### Image Details
- **Format:** PNG
- **Size:** ~15-50 KB per image
- **Coverage:** 0.15° × 0.15° (~15km × 15km)
- **Type:** True color composite

### Alternative Sources
If NASA Earth doesn't work, consider:
1. **Sentinel Hub** (ESA) - Higher resolution
2. **Google Earth Engine** - More data options
3. **USGS EarthExplorer** - Landsat archive

---

## ⚠️ Important Notes

### API Rate Limits

**DEMO_KEY:**
- 30 requests per hour
- 50 requests per day
- Good for testing

**Personal API Key (Free):**
- 1,000 requests per hour
- No daily limit
- Get at: https://api.nasa.gov/

### Image Availability

Not all dates may have images:
- Cloud cover
- Satellite orbit
- Processing delays

**Solution:** Script continues even if some images fail

### Data Synchronization

Weather data and images are for:
- ✅ Same locations
- ✅ Same dates
- ✅ Matched by coordinates

---

## 🔧 Troubleshooting

### Problem: "HTTP 429 Too Many Requests"
**Solution:** You've hit rate limit
- Wait 1 hour (DEMO_KEY)
- Get personal API key
- Reduce sample count

### Problem: No satellite images downloaded
**Solution:** 
- Check API key
- Verify internet connection
- Try different dates
- Images may not be available for all dates

### Problem: Weather data missing
**Solution:**
- Check date format (YYYYMMDD)
- Verify coordinates are in Bangladesh
- Check internet connection

### Problem: "ModuleNotFoundError"
**Solution:**
```cmd
pip install requests pandas python-dotenv
```

---

## 📊 Data Usage

### For Machine Learning

```python
import pandas as pd

# Load weather data
weather = pd.read_csv('weather_data/weather_data_20_samples.csv')

# Load image metadata
images = pd.read_csv('weather_data/satellite_images_metadata.csv')

# Merge on location and date
combined = weather.merge(images, on=['location', 'date'])
```

### For Analysis

```python
# Group by location
by_location = weather.groupby('Location').mean()

# Time series analysis
weather['Date'] = pd.to_datetime(weather['Date'])
weather.plot(x='Date', y='Temperature_2m_C')
```

### Display Satellite Images

```python
from PIL import Image
import matplotlib.pyplot as plt

# Load and display image
img = Image.open('satellite_images/nasa_earth_23.8103_90.4125_20240805.png')
plt.imshow(img)
plt.title('Dhaka - Aug 5, 2024')
plt.show()
```

---

## 🎯 Sample Outputs

### Weather Data CSV Structure
```
Date       | Location   | Latitude | Longitude | Temperature_2m_C | Precipitation_mm | ...
2024-08-05 | Dhaka      | 23.8103  | 90.4125   | 28.45           | 12.34           | ...
2024-08-05 | Chittagong | 22.3569  | 91.7832   | 27.89           | 35.67           | ...
...
```

### Image Metadata CSV Structure
```
location   | date       | filename                            | status  | size_kb
Dhaka      | 2024-08-05 | nasa_earth_23.8103_90.4125_...png | success | 34.5
Chittagong | 2024-08-05 | nasa_earth_22.3569_91.7832_...png | success | 28.3
...
```

---

## 📈 Expected Results

### Success Metrics
- ✅ Weather data: 20/20 records (100%)
- ✅ Satellite images: 15-20/20 (75-100%)
  - Some dates may not have images available
  - Cloud cover can affect availability

### Quality Checks
```python
# Check completeness
print(f"Weather records: {len(weather)}")
print(f"Missing values: {weather.isnull().sum().sum()}")
print(f"Unique locations: {weather['Location'].nunique()}")
print(f"Date range: {weather['Date'].min()} to {weather['Date'].max()}")
```

---

## 🚀 Next Steps

1. **Run the script:**
   ```cmd
   python extract_20_samples_complete.py
   ```

2. **Check outputs:**
   - `weather_data/weather_data_20_samples.csv`
   - `satellite_images/` folder

3. **Analyze data:**
   - Load CSV in Python/Excel
   - View images
   - Combine for ML

4. **Scale up (optional):**
   - Get NASA API key
   - Increase sample count
   - Add more locations

---

## 📞 Resources

- **NASA API Key:** https://api.nasa.gov/
- **NASA Earth API:** https://api.nasa.gov/planetary/earth
- **Landsat Info:** https://landsat.gsfc.nasa.gov/
- **NASA POWER API:** https://power.larc.nasa.gov/

---

## ✅ Quick Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install requests pandas`)
- [ ] Script downloaded
- [ ] Internet connection active
- [ ] (Optional) NASA API key obtained
- [ ] Ready to run!

---

**🎉 You're ready to extract satellite images + weather data!**

Run: `python extract_20_samples_complete.py`
