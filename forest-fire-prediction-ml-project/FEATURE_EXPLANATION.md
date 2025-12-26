# 🔍 Understanding the Input Features

The prediction model uses 12 features to assess fire risk. Understanding these helps in interpreting why a specific risk level is predicted.

### 📍 Spatial & Temporal Features
*   **X & Y Coordinates:** These represent the position in the Montesinho Natural Park map (Portugal). Fires often follow geographic patterns based on terrain or vegetation density in specific areas.
*   **Month & Day:** Fire risk is highly seasonal. Most fires occur during the dry summer months (August/September).

### 🌡️ Meteorological Features (The Weather)
*   **Temperature (°C):** Hotter weather dries out vegetation, making it easier to ignite.
*   **Relative Humidity (%):** Measures air moisture. Low humidity (< 30%) makes "fine fuels" (twigs/leaves) extremely dry and flammable.
*   **Wind Speed (km/h):** High wind provides oxygen to a fire and carries embers, causing the fire to spread rapidly.
*   **Rain (mm/m²):** Recent rain effectively resets fire risk by soaking the fuels.

### 🔥 FWI (Fire Weather Index) System
These are specialized indexes used by foresters worldwide to track fire danger:

1.  **FFMC (Fine Fuel Moisture Code):**
    *   Represents the moisture content of litter and other cured fine fuels.
    *   **Scale:** High values (> 90) mean very dry conditions where a single spark can start a fire.
2.  **DMC (Duff Moisture Code):**
    *   Represents moisture in loosely compacted organic layers (deeper than FFMC).
    *   Indicates how much fuel is available "deep down."
3.  **DC (Drought Code):**
    *   Represents moisture in deep, compact organic layers.
    *   This is a **seasonal** index. High values (> 600) indicate long-term dryness (drought).
4.  **ISI (Initial Spread Index):**
    *   Calculated from wind speed and FFMC.
    *   It estimates how fast a fire will spread immediately after ignition.
