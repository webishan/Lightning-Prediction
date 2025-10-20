# 🗂️ NASA API Project - Complete File Index

## 📋 Overview

Complete toolkit for extracting weather data and satellite images from NASA APIs for lightning detection research in Bangladesh.

---

## 🎯 QUICK START FILES

### ⭐ Start Here!
1. **`READY_TO_USE.md`** - **READ THIS FIRST**
   - Complete setup guide
   - Quick start instructions
   - What to expect

2. **`demo_quick_test.py`** - **RUN THIS FIRST**
   - Quick 2-sample test
   - Verifies everything works
   - Takes 10-15 seconds
   ```cmd
   python demo_quick_test.py
   ```

3. **`extract_20_samples_complete.py`** - **MAIN EXTRACTION**
   - Extract 20 samples (4 locations × 5 dates)
   - Satellite images + weather data
   - Takes 1-2 minutes
   ```cmd
   python extract_20_samples_complete.py
   ```

---

## 📚 DOCUMENTATION

### General Guides
- **`README.md`** - Overview of NASA POWER API
- **`QUICKSTART.md`** - Quick reference for weather data
- **`QUICK_REFERENCE.md`** - Lightning dataset quick reference

### Dataset Documentation
- **`DATASET_SUMMARY.md`** - 64 districts dataset documentation
- **`SATELLITE_EXTRACTION_GUIDE.md`** - Satellite + weather extraction guide

---

## 🐍 PYTHON SCRIPTS

### Core API
- **`nasa_power_api.py`** - Main API client class
  - Used by all other scripts
  - Contains NASAPowerAPI class

### Weather Data Extraction
1. **`example_usage.py`** - 6 example scenarios
2. **`extract_64_districts_lightning.py`** - All 64 districts (1,024 samples)
3. **`view_data.py`** - View/analyze CSV data
4. **`analyze_lightning_data.py`** - Comprehensive analysis

### Satellite + Weather Extraction
5. **`extract_20_samples_complete.py`** ⭐ - 20 samples with images
6. **`extract_satellite_and_weather.py`** - Alternative version
7. **`demo_quick_test.py`** - Quick test (2 samples)

### Utilities
8. **`estimate_time.py`** - Time estimation calculator
9. **`test_api.py`** - API endpoint tester

---

## 📊 DATA FILES

### Extracted Datasets
- **`bangladesh_64_districts_lightning_data.csv`** - 1,024 samples
  - All 64 districts
  - Aug 16-31, 2024
  - 14 lightning parameters
  - Size: ~450 KB

- **`dhaka_august_2024.csv`** - 31 samples
  - Dhaka only
  - August 2024
  - Example dataset

### Output Folders
- **`weather_data/`** - CSV files go here
- **`satellite_images/`** - PNG images go here

---

## ⚙️ CONFIGURATION

- **`requirements.txt`** - Python dependencies
  ```cmd
  pip install -r requirements.txt
  ```

---

## 📂 FOLDER STRUCTURE

```
NASA API/
│
├── 📖 Documentation
│   ├── READY_TO_USE.md ⭐ START HERE
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── QUICK_REFERENCE.md
│   ├── DATASET_SUMMARY.md
│   └── SATELLITE_EXTRACTION_GUIDE.md
│
├── 🐍 Scripts - Weather Only
│   ├── nasa_power_api.py (core)
│   ├── example_usage.py
│   ├── extract_64_districts_lightning.py
│   ├── view_data.py
│   ├── analyze_lightning_data.py
│   └── estimate_time.py
│
├── 🛰️ Scripts - Satellite + Weather
│   ├── extract_20_samples_complete.py ⭐ MAIN
│   ├── extract_satellite_and_weather.py
│   └── demo_quick_test.py 🧪 TEST FIRST
│
├── 📊 Data Files
│   ├── bangladesh_64_districts_lightning_data.csv
│   ├── dhaka_august_2024.csv
│   ├── weather_data/ (output folder)
│   └── satellite_images/ (output folder)
│
└── ⚙️ Configuration
    └── requirements.txt
```

---

## 🎯 USE CASES

### Use Case 1: Extract Weather Data Only
**Goal:** Get weather parameters for lightning detection

**Files:**
1. `extract_64_districts_lightning.py` - All 64 districts
2. `analyze_lightning_data.py` - Analyze results

**Output:**
- CSV with 1,024 weather samples
- 14 lightning parameters
- All Bangladesh districts

**Time:** ~3-4 minutes

---

### Use Case 2: Extract Satellite Images + Weather
**Goal:** Get synchronized satellite images and weather data

**Files:**
1. `demo_quick_test.py` - Test (2 samples)
2. `extract_20_samples_complete.py` - Full extraction (20 samples)

**Output:**
- CSV with 20 weather samples
- 15-20 PNG satellite images
- Metadata files

**Time:** ~1-2 minutes

---

### Use Case 3: Custom Extraction
**Goal:** Specific locations/dates

**Files:**
1. Edit `extract_satellite_and_weather.py`
2. Modify locations and dates
3. Run custom extraction

**Output:**
- Customized dataset
- Your specific parameters

---

## 🚀 QUICK START COMMANDS

### 1. Install Dependencies
```cmd
pip install -r requirements.txt
```

### 2. Test System (2 samples)
```cmd
python demo_quick_test.py
```

### 3. Extract Satellite + Weather (20 samples)
```cmd
python extract_20_samples_complete.py
```

### 4. Extract All Districts (1,024 samples)
```cmd
python extract_64_districts_lightning.py
```

### 5. Analyze Results
```cmd
python analyze_lightning_data.py
```

---

## 📊 AVAILABLE DATASETS

### Dataset 1: 64 Districts (Already Extracted ✅)
- **File:** `bangladesh_64_districts_lightning_data.csv`
- **Samples:** 1,024
- **Coverage:** All 64 districts
- **Period:** Aug 16-31, 2024
- **Features:** 19 columns (14 weather + 5 metadata)
- **Status:** ✅ Ready to use

### Dataset 2: 20 Samples with Images (To Be Extracted)
- **Command:** `python extract_20_samples_complete.py`
- **Samples:** 20
- **Coverage:** 4 key locations
- **Period:** Aug 5, 10, 15, 20, 25 (2024)
- **Features:** 18 weather + satellite images
- **Status:** 📥 Run script to extract

---

## 🌍 LOCATIONS COVERED

### Already Extracted (64 Districts)
- ✅ All 8 divisions
- ✅ All 64 districts
- ✅ Full Bangladesh coverage

### Satellite Extraction (4 Key Locations)
- Dhaka (capital)
- Chittagong (coastal, high lightning)
- Sylhet (high rainfall)
- Rangpur (northern)

---

## ⚡ FEATURES EXTRACTED

### Weather Parameters (14)
1. Temperature (avg, max, min)
2. Temperature range
3. Dew point
4. Relative humidity
5. Specific humidity
6. Precipitation
7. Wind speed (2m)
8. Wind speed (10m)
9. Wind direction
10. Surface pressure
11. Solar radiation
12. Longwave radiation

### Metadata (5)
- Date
- Location name
- Division
- Latitude
- Longitude

### Satellite Images (when extracted)
- Landsat 8 imagery
- PNG format
- ~15-50 KB per image
- 0.15° × 0.15° coverage

---

## 📈 DATASET STATISTICS

### 64 Districts Dataset
```
Records:          1,024
Locations:        64 districts
Date Range:       Aug 16-31, 2024 (16 days)
Features:         19
Missing Values:   0 (100% complete)
High Risk Days:   48.5%
File Size:        ~450 KB
```

### 20 Sample Dataset (After Extraction)
```
Records:          20
Locations:        4 cities
Date Range:       5 dates in Aug 2024
Features:         18 weather + images
Satellite Images: 15-20 PNG files
Total Size:       ~500 KB - 1 MB
```

---

## 🎓 FOR YOUR THESIS

### Already Available
✅ **1,024 weather samples** from 64 districts  
✅ **14 lightning-specific parameters**  
✅ **Complete geographic coverage**  
✅ **Monsoon season data**  
✅ **Ready for ML training**

### After Running Satellite Extraction
✅ **20 synchronized samples**  
✅ **Satellite imagery**  
✅ **Multi-modal dataset**  
✅ **Image + tabular data**  
✅ **Perfect for deep learning**

---

## 🔧 CUSTOMIZATION

### Change Sample Count
Edit number of locations × dates:
- 10 samples: 2 locations × 5 dates
- 20 samples: 4 locations × 5 dates ✅
- 50 samples: 10 locations × 5 dates

### Change Time Period
Modify date ranges in scripts:
- Monsoon: June-September
- Pre-monsoon: March-May
- Post-monsoon: October-November

### Change Locations
Add/remove cities in location dictionaries

### Change Parameters
Edit parameter lists in `nasa_power_api.py`

---

## ⚠️ IMPORTANT NOTES

### API Keys
- **NASA POWER:** No key needed (free)
- **NASA Earth:** DEMO_KEY (30/hour) or personal key (1000/hour)
- **Get key:** https://api.nasa.gov/

### Data Availability
- **Weather:** Always available (1981-present)
- **Satellite:** Subject to cloud cover, orbits
- **Expect:** 100% weather, 75-95% images

### Rate Limits
- **DEMO_KEY:** 30 requests/hour
- **Personal key:** 1,000 requests/hour
- **Both FREE!**

---

## 📞 HELP & RESOURCES

### Documentation Files
1. `READY_TO_USE.md` - Main guide
2. `SATELLITE_EXTRACTION_GUIDE.md` - Satellite details
3. `DATASET_SUMMARY.md` - 64 districts info

### Online Resources
- NASA API: https://api.nasa.gov/
- NASA POWER: https://power.larc.nasa.gov/
- Landsat: https://landsat.gsfc.nasa.gov/

### Scripts Help
- Run with `--help` flag (if implemented)
- Read docstrings in code
- Check comments in scripts

---

## ✅ COMPLETION CHECKLIST

### Initial Setup
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Files downloaded
- [ ] Internet connection active

### Testing
- [ ] Run `demo_quick_test.py`
- [ ] Verify output files created
- [ ] Check CSV data quality

### Main Extraction
- [ ] (Optional) Get NASA API key
- [ ] Run `extract_20_samples_complete.py`
- [ ] Verify 20 weather samples
- [ ] Check satellite images downloaded

### Analysis
- [ ] Load CSV in Python/Excel
- [ ] View satellite images
- [ ] Combine data for ML
- [ ] Start thesis analysis

---

## 🎯 RECOMMENDED WORKFLOW

1. **Read:** `READY_TO_USE.md`
2. **Test:** `python demo_quick_test.py`
3. **Extract:** `python extract_20_samples_complete.py`
4. **Analyze:** Use existing 64-district dataset + new 20-sample dataset
5. **Model:** Build lightning prediction model

---

## 📊 SUMMARY

| Feature | Available |
|---------|-----------|
| **Weather Data API** | ✅ Working |
| **Satellite Image API** | ✅ Working |
| **64 Districts Dataset** | ✅ Extracted (1,024 samples) |
| **20 Samples + Images** | 📥 Ready to extract |
| **Documentation** | ✅ Complete |
| **Test Scripts** | ✅ Working |
| **Analysis Tools** | ✅ Available |

---

## 🎉 YOU'RE READY!

**Status:** ✅ All systems operational  
**Next Step:** Run `python demo_quick_test.py`  
**Then:** Run `python extract_20_samples_complete.py`  
**Finally:** Build your lightning detection model!

---

**Good luck with your thesis! 🚀⚡🛰️**

Last updated: October 19, 2025
