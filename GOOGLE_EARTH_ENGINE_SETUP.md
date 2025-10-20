# 🌍 Google Earth Engine Setup Guide
## Hybrid Solution: Google Earth Engine + NASA POWER API

This guide helps you set up Google Earth Engine (GEE) for satellite images while keeping NASA POWER for weather data.

---

## ✅ Why Google Earth Engine?

### Advantages over NASA Earth API:
- ✅ **More reliable** - Better uptime and faster responses
- ✅ **More data sources** - Landsat, Sentinel, MODIS, etc.
- ✅ **Better filtering** - Cloud masking, quality filtering
- ✅ **Free for research** - Academic/non-commercial use
- ✅ **Python API** - Easy integration with existing code
- ✅ **Large catalog** - 40+ years of satellite data

### What You'll Keep:
- ✅ **NASA POWER API** for weather data (already working great!)
- ✅ **Your existing 1,024-sample dataset**
- ✅ **All your analysis scripts**

---

## 📋 Setup Steps (10-15 minutes)

### **Step 1: Sign Up for Google Earth Engine (5 minutes)**

1. Visit: https://earthengine.google.com/signup/

2. Click **"Register a Noncommercial or Commercial Cloud project"**

3. Choose: **"Unpaid usage" → "Academia & Research"**

4. Fill in:
   - Project name: `Bangladesh Lightning Detection`
   - Organization: Your university name
   - Project type: `Academic Research`
   - Description: `Lightning detection using satellite imagery and weather data for thesis research`

5. Click **"Continue to Summary"** → **"Confirm"**

6. Wait for approval email (usually **instant to 24 hours**)

### **Step 2: Install Google Earth Engine Python API**

```cmd
pip install earthengine-api
```

Or if you need all dependencies:
```cmd
pip install earthengine-api google-auth google-auth-oauthlib google-auth-httplib2
```

### **Step 3: Authenticate**

Run this once to authenticate:
```cmd
earthengine authenticate
```

This will:
1. Open browser for Google login
2. Ask for permission
3. Give you a code
4. Paste code in terminal

---

## 🚀 Quick Start - Test GEE Access

### Test Script: `test_gee_access.py`

```python
import ee

try:
    # Initialize Earth Engine
    ee.Initialize()
    
    # Test by getting an image
    image = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_137044_20240815')
    
    print("✅ Google Earth Engine is working!")
    print("✅ You can access satellite imagery!")
    print(f"✅ Test image: {image.get('system:index').getInfo()}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Make sure you:")
    print("   1. Signed up at https://earthengine.google.com/signup/")
    print("   2. Ran: earthengine authenticate")
    print("   3. Installed: pip install earthengine-api")
```

---

## 🎯 What Satellites to Use?

### **Recommended: Landsat 8 (Best for Bangladesh)**

**Why Landsat 8?**
- ✅ Free and open
- ✅ 30m resolution (good for regional analysis)
- ✅ 16-day revisit time
- ✅ Available since 2013
- ✅ Good for monsoon/lightning studies

**Alternative Options:**

| Satellite | Resolution | Revisit | Best For |
|-----------|-----------|---------|----------|
| **Landsat 8** | 30m | 16 days | General research (RECOMMENDED) |
| **Sentinel-2** | 10m | 5 days | Higher detail |
| **MODIS** | 250m-1km | Daily | Daily monitoring |
| **Sentinel-1** | 10m | 6 days | Radar (works through clouds!) |

---

## 📊 Data You'll Get

### From Google Earth Engine:
- 🛰️ Satellite images (Landsat 8, Sentinel, etc.)
- ☁️ Cloud-free or cloud-masked imagery
- 📏 Multiple spectral bands
- 🎨 RGB composites for visualization
- 📈 Calculated indices (NDVI, etc.)

### From NASA POWER API (Existing):
- ⚡ 14 weather parameters
- 🌡️ Temperature, humidity, pressure
- 🌧️ Precipitation
- 💨 Wind speed and direction
- ☀️ Solar radiation

---

## 💰 Cost

**Google Earth Engine:**
- 🆓 **FREE** for research and education
- 🆓 **FREE** for non-commercial use
- 💳 Paid only for commercial applications

**NASA POWER API:**
- 🆓 **FREE** (no limits for weather data)

---

## ⏱️ Time Required

1. **GEE Signup**: 5 minutes (+ approval wait)
2. **Install Python API**: 2 minutes
3. **Authentication**: 3 minutes
4. **Test access**: 1 minute
5. **Create extraction script**: Already done! (next step)

**Total**: ~15 minutes active time

---

## 🎓 Learning Resources

### Official Documentation:
- **Earth Engine Guide**: https://developers.google.com/earth-engine/guides
- **Python API Docs**: https://developers.google.com/earth-engine/guides/python_install
- **Code Examples**: https://github.com/google/earthengine-api/tree/master/python/examples

### Tutorials:
- **GEE Catalog**: https://developers.google.com/earth-engine/datasets/
- **Landsat 8 Collection**: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2

---

## 🔧 Troubleshooting

### "User not registered" Error
**Solution**: Wait for approval email from Google Earth Engine team
**Time**: Usually instant, max 24 hours

### "Authentication failed" Error
**Solution**: 
```cmd
earthengine authenticate --force
```

### "Module 'ee' not found" Error
**Solution**:
```cmd
pip install earthengine-api --upgrade
```

### Connection timeout
**Solution**: Google Earth Engine has better reliability than NASA Earth API
- GEE servers are more robust
- Better caching
- Regional servers

---

## 📝 Next Steps

After setup is complete:

1. ✅ Run authentication: `earthengine authenticate`
2. ✅ Test access: `python test_gee_access.py`
3. ✅ Run hybrid extraction: `python extract_hybrid_gee_nasa.py` (I'll create this next!)

---

## 🎯 Why This is Better for Your Thesis

### Research Quality:
- **More reliable data collection** - No timeouts
- **Better spatial coverage** - More satellites available
- **Industry-standard tool** - GEE is widely used in research
- **Reproducible** - Other researchers can replicate your work

### Citations for Thesis:
```
Satellite imagery was obtained from Google Earth Engine platform 
(Gorelick et al., 2017), specifically using Landsat 8 Collection 2 
Level-2 surface reflectance products. Meteorological data was acquired 
from NASA POWER API (Stackhouse et al., 2018).

References:
Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017). 
Google Earth Engine: Planetary-scale geospatial analysis for everyone. 
Remote Sensing of Environment, 202, 18-27.
```

---

## ⚡ Ready to Start?

Run these commands in order:

```cmd
# 1. Install Earth Engine
pip install earthengine-api

# 2. Authenticate
earthengine authenticate

# 3. Test (I'll create this next)
python test_gee_access.py

# 4. Extract data (I'll create this next)
python extract_hybrid_gee_nasa.py
```

---

**Let me know when you're approved for GEE, and I'll create the hybrid extraction script!** 🚀
