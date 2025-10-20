# Enable Earth Engine API for Your Project

## Quick Fix (2 minutes):

### **Option 1: Click This Link** ⭐ (Easiest)

Visit this URL to enable the Earth Engine API:

```
https://console.cloud.google.com/apis/library/earthengine.googleapis.com?project=bangladesh-lightning-detection
```

Then click the big blue **"ENABLE"** button.

---

### **Option 2: Manual Steps**

1. Go to: https://console.cloud.google.com/
2. Make sure "Bangladesh Lightning detection" project is selected (top left)
3. Click on "APIs & Services" in the left menu
4. Click "+ ENABLE APIS AND SERVICES" (top of page)
5. Search for "Earth Engine API"
6. Click on it
7. Click "ENABLE"

---

### **Option 3: Use Cloud Shell** (From your screenshot)

You already have Cloud Shell open! Run this command:

```bash
gcloud services enable earthengine.googleapis.com --project=bangladesh-lightning-detection
```

---

## After Enabling:

Wait 1-2 minutes for the API to fully activate, then run:

```cmd
python test_gee_access.py
```

It should work! ✅
