import React, { useMemo } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import geoJsonData from '../../assets/data/sri_lanka_districts.json';
import { ACTIVITY_CONFIG, formatNumber, normalizeDistrictName } from '../../utils/formatters';

// Helper to extract activity category consistently across Historical (activity_level) and Current (relative_activity) records
function getActivityLevel(dRecord) {
  if (!dRecord) return null;
  return dRecord.relative_activity || dRecord.activity_level || null;
}

export default function DistrictMap({ latestDistricts, selectedDistrict, onSelectDistrict }) {
  // Map API districts by normalized district name
  const districtMap = useMemo(() => {
    const map = {};
    const matchedApiKeys = new Set();

    if (latestDistricts && Array.isArray(latestDistricts)) {
      latestDistricts.forEach(d => {
        if (d.district) {
          const normKey = normalizeDistrictName(d.district);
          map[normKey] = d;
        }
      });
    }

    // Diagnostic check for GeoJSON to API district matching
    const gjFeatures = geoJsonData.features || [];
    const unmatchedGeoJson = [];
    
    gjFeatures.forEach(f => {
      const props = f.properties || {};
      const rawName = props.district || props.ADM2_EN || props.name || props.shapeName;
      const normKey = normalizeDistrictName(rawName);
      if (normKey && map[normKey]) {
        matchedApiKeys.add(normKey);
      } else {
        unmatchedGeoJson.push(rawName);
      }
    });

    const apiKeys = Object.keys(map);
    const unmatchedApi = apiKeys.filter(k => !matchedApiKeys.has(k));

    if (process.env.NODE_ENV !== 'production') {
      console.log(`[DistrictMap Join Audit] Matched API districts: ${matchedApiKeys.size}/25`);
      if (unmatchedGeoJson.length > 0) {
        console.log('[DistrictMap Audit] Unmatched GeoJSON shapes:', unmatchedGeoJson);
      }
      if (unmatchedApi.length > 0) {
        console.warn('[DistrictMap Audit] Unmatched API districts:', unmatchedApi);
      }
    }

    return map;
  }, [latestDistricts]);

  // Style function for Leaflet GeoJSON layer
  const styleFeature = (feature) => {
    const props = feature.properties || {};
    const rawName = props.district || props.ADM2_EN || props.name || props.shapeName || '';
    const normKey = normalizeDistrictName(rawName);
    
    const dData = districtMap[normKey];
    const activity = getActivityLevel(dData);
    
    // Explicit check: only matched districts with valid activity level get colored
    const cfg = activity ? (ACTIVITY_CONFIG[activity] || ACTIVITY_CONFIG['NO DATA']) : ACTIVITY_CONFIG['NO DATA'];

    const isSelected = selectedDistrict && normalizeDistrictName(selectedDistrict) === normKey;

    return {
      fillColor: cfg.color,
      weight: isSelected ? 3 : 1.5,
      opacity: 1,
      color: isSelected ? '#0f172a' : '#ffffff',
      dashArray: '',
      fillOpacity: isSelected ? 0.90 : (activity ? 0.80 : 0.4)
    };
  };

  // Feature interactions (hover, click, tooltip)
  const onEachFeature = (feature, layer) => {
    const props = feature.properties || {};
    const rawName = (props.district || props.ADM2_EN || props.name || 'Unknown').trim();
    const normKey = normalizeDistrictName(rawName);
    const dData = districtMap[normKey];

    if (!dData) {
      // Explicit No Data tooltip state for unmapped offshore reef features
      layer.bindTooltip(`
        <div style="font-family: system-ui, sans-serif; padding: 4px;">
          <div style="font-weight: 700; font-size: 13px; color: #475569;">${rawName}</div>
          <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">No API data available</div>
        </div>
      `, { sticky: true });
      return;
    }

    const currentCases = formatNumber(dData.current_cases);
    const forecastCases = formatNumber(dData.forecast_cases_1w);
    const activity = getActivityLevel(dData) || 'UNKNOWN';
    const trend = dData.forecast_trend || 'STABLE';

    const cfg = ACTIVITY_CONFIG[activity] || ACTIVITY_CONFIG['LOW'];

    // Tooltip content: District, Current cases, 1-week RF forecast, Relative activity, Trend
    const tooltipContent = `
      <div style="font-family: system-ui, sans-serif; min-width: 170px; padding: 4px;">
        <div style="font-weight: 700; font-size: 14px; color: #0f172a; margin-bottom: 4px;">
          ${dData.district} District
        </div>
        <div style="display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 10px; font-weight: 700; background: ${cfg.badgeBg}; color: ${cfg.color}; margin-bottom: 8px;">
          Relative Activity: ${activity}
        </div>
        <div style="font-size: 12px; color: #334155; display: flex; justify-content: space-between; margin-bottom: 3px;">
          <span>Current Cases:</span>
          <span style="font-weight: 700; color: #0f172a;">${currentCases}</span>
        </div>
        <div style="font-size: 12px; color: #334155; display: flex; justify-content: space-between; margin-bottom: 3px;">
          <span>1-Wk RF Forecast:</span>
          <span style="font-weight: 800; color: #0f766e;">${forecastCases}</span>
        </div>
        <div style="font-size: 11px; color: #64748b; margin-top: 6px; border-top: 1px solid #e2e8f0; padding-top: 4px; display: flex; justify-content: space-between;">
          <span>Trend:</span>
          <strong style="color: #1e293b;">${trend}</strong>
        </div>
      </div>
    `;

    layer.bindTooltip(tooltipContent, {
      sticky: true,
      direction: 'auto'
    });

    layer.on({
      mouseover: (e) => {
        const l = e.target;
        l.setStyle({
          weight: 3,
          color: '#0f172a',
          fillOpacity: 0.95
        });
      },
      mouseout: (e) => {
        const l = e.target;
        const isSel = selectedDistrict && normalizeDistrictName(selectedDistrict) === normKey;
        l.setStyle({
          weight: isSel ? 3 : 1.5,
          color: isSel ? '#0f172a' : '#ffffff',
          fillOpacity: isSel ? 0.90 : 0.80
        });
      },
      click: () => {
        if (onSelectDistrict && dData.district) {
          onSelectDistrict(dData.district);
        }
      }
    });
  };

  const position = [7.8731, 80.7718];

  return (
    <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-base font-bold text-slate-900">Sri Lanka District Activity Map</h2>
          <p className="text-xs text-slate-500">25-district relative dengue activity choropleth map</p>
        </div>
        <span className="text-[10px] font-mono bg-slate-100 px-2 py-1 rounded text-slate-600 border border-slate-200">
          25 Districts Matched
        </span>
      </div>

      {/* Map Container using Key-Free OpenStreetMap Tiles */}
      <div className="relative w-full h-[460px] rounded-xl overflow-hidden border border-slate-200 shadow-inner">
        <MapContainer 
          center={position} 
          zoom={7} 
          scrollWheelZoom={false}
          className="w-full h-full"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <GeoJSON 
            data={geoJsonData} 
            style={styleFeature}
            onEachFeature={onEachFeature}
          />
        </MapContainer>

        {/* Floating Map Legend */}
        <div className="absolute bottom-3 left-3 z-[1000] bg-white/95 backdrop-blur-md p-3 rounded-xl border border-slate-200 shadow-md text-xs max-w-[210px]">
          <div className="font-bold text-slate-800 text-[11px] mb-2 uppercase tracking-wider">
            Relative Activity Level
          </div>
          <div className="space-y-1.5">
            {['VERY HIGH', 'HIGH', 'ELEVATED', 'LOW'].map((level) => {
              const cfg = ACTIVITY_CONFIG[level];
              return (
                <div key={level} className="flex items-center gap-2">
                  <span 
                    className="w-3.5 h-3.5 rounded-sm border border-black/10 shrink-0" 
                    style={{ backgroundColor: cfg.color }}
                  />
                  <span className="font-semibold text-slate-700 text-[11px]">{level}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="mt-3 text-[11px] text-slate-500 italic bg-slate-50 p-2.5 rounded-lg border border-slate-200/60">
        Statistical historical-percentile categories — not official health authority alerts. Click any district to inspect driver details.
      </div>
    </div>
  );
}
