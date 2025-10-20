"""
Visualize 30-Minute Interval Weather Data
Shows diurnal patterns and temporal variations
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_30min_data():
    """Create visualizations of 30-minute interval weather data"""
    
    # Load data
    data_file = Path(__file__).parent / 'weather_data' / 'weather_data_30min_intervals.csv'
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df)} weather records")
    
    # Convert Time to datetime for plotting
    df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour + \
                 pd.to_datetime(df['Time'], format='%H:%M').dt.minute / 60
    
    # Filter for one location and one date for clarity
    dhaka_jan15 = df[(df['Location'] == 'Dhaka') & (df['Date'] == '2023-01-15')]
    
    if len(dhaka_jan15) == 0:
        print("❌ No data found for Dhaka 2023-01-15")
        return
    
    print(f"\n📊 Visualizing {len(dhaka_jan15)} records for Dhaka on 2023-01-15")
    print(f"   Temperature range: {dhaka_jan15['Temperature_2m_C'].min():.1f} - {dhaka_jan15['Temperature_2m_C'].max():.1f}°C")
    print(f"   Humidity range: {dhaka_jan15['Relative_Humidity_%'].min():.1f} - {dhaka_jan15['Relative_Humidity_%'].max():.1f}%")
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Dhaka Weather Patterns - 30-Minute Intervals (2023-01-15)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Temperature
    ax1 = axes[0, 0]
    ax1.plot(dhaka_jan15['Hour'], dhaka_jan15['Temperature_2m_C'], 
             marker='o', markersize=3, linewidth=2, color='#e74c3c')
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title('Temperature Variation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 3))
    
    # Add annotations for key times
    min_temp_idx = dhaka_jan15['Temperature_2m_C'].idxmin()
    max_temp_idx = dhaka_jan15['Temperature_2m_C'].idxmax()
    ax1.annotate(f"Min: {dhaka_jan15.loc[min_temp_idx, 'Temperature_2m_C']:.1f}°C",
                xy=(dhaka_jan15.loc[min_temp_idx, 'Hour'], 
                    dhaka_jan15.loc[min_temp_idx, 'Temperature_2m_C']),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red'))
    ax1.annotate(f"Max: {dhaka_jan15.loc[max_temp_idx, 'Temperature_2m_C']:.1f}°C",
                xy=(dhaka_jan15.loc[max_temp_idx, 'Hour'], 
                    dhaka_jan15.loc[max_temp_idx, 'Temperature_2m_C']),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # Plot 2: Humidity
    ax2 = axes[0, 1]
    ax2.plot(dhaka_jan15['Hour'], dhaka_jan15['Relative_Humidity_%'], 
             marker='o', markersize=3, linewidth=2, color='#3498db')
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Relative Humidity (%)', fontsize=12)
    ax2.set_title('Humidity Variation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 24)
    ax2.set_xticks(range(0, 25, 3))
    ax2.set_ylim(0, 110)
    
    # Plot 3: Wind Speed
    ax3 = axes[1, 0]
    ax3.plot(dhaka_jan15['Hour'], dhaka_jan15['Wind_Speed_2m_m/s'], 
             marker='o', markersize=3, linewidth=2, color='#2ecc71')
    ax3.set_xlabel('Hour of Day', fontsize=12)
    ax3.set_ylabel('Wind Speed (m/s)', fontsize=12)
    ax3.set_title('Wind Speed Variation', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 24)
    ax3.set_xticks(range(0, 25, 3))
    
    # Plot 4: Temperature vs Humidity (inverse relationship)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(dhaka_jan15['Temperature_2m_C'], 
                         dhaka_jan15['Relative_Humidity_%'],
                         c=dhaka_jan15['Hour'], cmap='twilight', 
                         s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Temperature (°C)', fontsize=12)
    ax4.set_ylabel('Relative Humidity (%)', fontsize=12)
    ax4.set_title('Temperature vs Humidity', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar for time
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Hour of Day', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'weather_30min_patterns.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_file}")
    
    # Show plot
    plt.show()
    
    # Print statistics
    print("\n" + "="*70)
    print("📈 DAILY STATISTICS (Dhaka, 2023-01-15)")
    print("="*70)
    print(f"Temperature:")
    print(f"  Min: {dhaka_jan15['Temperature_2m_C'].min():.2f}°C at {dhaka_jan15.loc[min_temp_idx, 'Time']}")
    print(f"  Max: {dhaka_jan15['Temperature_2m_C'].max():.2f}°C at {dhaka_jan15.loc[max_temp_idx, 'Time']}")
    print(f"  Range: {dhaka_jan15['Temperature_2m_C'].max() - dhaka_jan15['Temperature_2m_C'].min():.2f}°C")
    print(f"  Mean: {dhaka_jan15['Temperature_2m_C'].mean():.2f}°C")
    
    print(f"\nHumidity:")
    print(f"  Min: {dhaka_jan15['Relative_Humidity_%'].min():.2f}%")
    print(f"  Max: {dhaka_jan15['Relative_Humidity_%'].max():.2f}%")
    print(f"  Mean: {dhaka_jan15['Relative_Humidity_%'].mean():.2f}%")
    
    print(f"\nWind Speed:")
    print(f"  Min: {dhaka_jan15['Wind_Speed_2m_m/s'].min():.2f} m/s")
    print(f"  Max: {dhaka_jan15['Wind_Speed_2m_m/s'].max():.2f} m/s")
    print(f"  Mean: {dhaka_jan15['Wind_Speed_2m_m/s'].mean():.2f} m/s")
    
    print(f"\nPrecipitation:")
    total_precip = dhaka_jan15['Precipitation_mm'].sum()
    print(f"  Total: {total_precip:.2f} mm")
    if total_precip > 0:
        print(f"  Rain events: {(dhaka_jan15['Precipitation_mm'] > 0).sum()} intervals")
    else:
        print(f"  Status: Dry day (no precipitation)")
    
    print("="*70)


if __name__ == "__main__":
    try:
        import matplotlib
        visualize_30min_data()
    except ImportError:
        print("❌ Error: matplotlib not installed")
        print("Install with: pip install matplotlib")
