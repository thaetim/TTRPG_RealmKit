
## Prerequisites

Download Planet Generator .zip file from http://hjemmesider.diku.dk/~torbenm/Planet/ and unpack it's contents into `planet` folder.

# PROCESS
1. Planet Generator + Map Painter workflow
   Using VS Code, install Live Preview extension and run `preview-watcher.py`
   <screenshot>
      > preview-watcher.py
2. Generate maps:
   - bathy
   - biome (logarythmic?)
   - realistic both log and linear
   - grayscale both log and linear
3. Process images in GIMP cmd:
   1. Extract ocean-land
      - based on bathy.col elevations ??
   2. Recolor ocean and land into separate layers
   3. Recolor biomemap with realistic colors (allow for stopping at that point to custom recolor - based on koppen palette)
   4. Generate xgrayscale (log * lin chyba)
   5. Add gaussian blur based on elevation (grayscale)
   6. Add snow layer using grayscale
   7. Add bumpmap using grayscale(s)
   8. Render
4. Reproject with G-Projector
5. Trim with GIMP
6. Upscale with Upscayl
7. Cut into tiles with PhotoShop script
8. Optimize with pyngyu
9.  Load to repo