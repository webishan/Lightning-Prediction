"""
Quick Hybrid Extraction - No Prompts
Extracts 20 samples with GEE + NASA POWER (non-interactive)
"""

import sys
sys.path.append(r'd:\Study mat\THESIS\NASA API')

from extract_hybrid_gee_nasa import HybridDataExtractor

# Configuration
locations = {
    'Dhaka': {'lat': 23.8103, 'lon': 90.4125},
    'Chittagong': {'lat': 22.3569, 'lon': 91.7832},
    'Sylhet': {'lat': 24.8949, 'lon': 91.8687},
    'Rangpur': {'lat': 25.7439, 'lon': 89.2752},
}

# Use 2023 dates (confirmed data availability)
dates = ['2023-08-15', '2023-08-18', '2023-08-21', '2023-08-24', '2023-08-27']

print("=" * 70)
print("🚀 QUICK HYBRID EXTRACTION (Non-Interactive)")
print("=" * 70)
print(f"📍 Locations: {len(locations)}")
print(f"📅 Dates: {len(dates)}")
print(f"📊 Total samples: {len(locations) * len(dates)}")
print(f"☁️  Cloud cover: 70% (monsoon season)")
print("=" * 70)
print()

# Create extractor and run
extractor = HybridDataExtractor()
extractor.extract_hybrid_data(
    locations=locations,
    dates=dates,
    cloud_cover_max=70  # Higher threshold for monsoon season
)

print("\n✅ Extraction complete!")
