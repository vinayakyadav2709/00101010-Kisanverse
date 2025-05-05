"use client";
"use client"
import React from 'react';
import { 
  ComposableMap, 
  Geographies, 
  Geography, 
  Marker 
} from 'react-simple-maps';

// Define a function to stringify and parse the coordinates to ensure consistent precision
const normalizeCoordinates = (coords) => {
  // Convert to string and back to number with fixed precision
  if (Array.isArray(coords)) {
    return coords.map(coord => parseFloat(coord.toFixed(10)));
  }
  return coords;
};

interface WorldMapProps {
  markers: Array<{
    name: string;
    coordinates: [number, number];
    users: number;
  }>;
}

export function WorldMap({ markers }: WorldMapProps) {
  // Normalize all marker coordinates to ensure consistent precision
  const normalizedMarkers = React.useMemo(() => {
    return markers.map(marker => ({
      ...marker,
      coordinates: normalizeCoordinates(marker.coordinates),
    }));
  }, [markers]);

  return (
    <div className="h-[300px]">
      <ComposableMap>
        {/* Use suppressHydrationWarning to ignore any remaining minor differences */}
        <g width={800} height={600} projection="geoEqualEarth" projectionConfig={{}} suppressHydrationWarning>
          <svg viewBox="0 0 800 600" className="rsm-svg">
            <Geographies geography="/world-110m.json">
              {({ geographies }) =>
                geographies.map((geo) => (
                  <Geography key={geo.rsmKey} geography={geo} fill="#EAEAEC" stroke="#D6D6DA" />
                ))
              }
            </Geographies>
            
            {normalizedMarkers.map((marker, index) => (
              <Marker 
                key={`marker-${index}`}
                coordinates={marker.coordinates}
                // Add suppressHydrationWarning to the Marker component
                suppressHydrationWarning
              >
                <circle r={Math.sqrt(marker.users) / 10} fill="#4ade80" opacity={0.8} />
                <text
                  textAnchor="middle"
                  y={-10}
                  style={{ fontFamily: "system-ui", fill: "#5D5A6D", fontSize: "8px" }}
                >
                  {marker.name}
                </text>
              </Marker>
            ))}
          </svg>
        </g>
      </ComposableMap>
    </div>
  );
}
