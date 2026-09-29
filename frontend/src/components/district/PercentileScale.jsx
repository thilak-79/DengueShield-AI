import React from 'react';
import { ACTIVITY_CONFIG, formatNumber } from '../../utils/formatters';

export default function PercentileScale({ districtDetail }) {
  if (!districtDetail) return null;

  const p50 = districtDetail.historical_p50 || 0;
  const p75 = districtDetail.historical_p75 || 0;
  const p90 = districtDetail.historical_p90 || 0;
  const current = districtDetail.current_cases || 0;
  const forecast = districtDetail.forecast_cases_1w || 0;
  const activity = districtDetail.relative_activity || districtDetail.activity_level || 'LOW';

  // Calculate percentage positions for scale (0% to 100%)
  const maxVal = Math.max(p90 * 1.3, current * 1.1, forecast * 1.1, 10);
  
  const getPos = (val) => Math.min(Math.max((val / maxVal) * 100, 0), 100);

  const posP50 = getPos(p50);
  const posP75 = getPos(p75);
  const posP90 = getPos(p90);
  const posForecast = getPos(forecast);

  const cfg = ACTIVITY_CONFIG[activity] || ACTIVITY_CONFIG['LOW'];

  return (
    <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
        <div>
          <h3 className="text-sm font-bold text-slate-900">Historical Distribution & Activity Thresholds</h3>
          <p className="text-xs text-slate-500">Comparison against district-specific historical percentiles</p>
        </div>
        <div className={`px-3 py-1 rounded-full text-xs font-bold ${cfg.badgeBg} ${cfg.badgeText} self-start sm:self-auto`}>
          Relative Activity: {activity}
        </div>
      </div>

      {/* Metric Badges Grid */}
      <div className="grid grid-cols-3 gap-3 mb-6 text-center">
        <div className="p-2.5 bg-emerald-50/60 border border-emerald-200/60 rounded-xl">
          <div className="text-[10px] uppercase font-bold text-emerald-800">P50 (Median)</div>
          <div className="text-sm font-extrabold text-emerald-950 mt-0.5">{formatNumber(p50)}</div>
          <div className="text-[10px] text-emerald-700">LOW: &lt; P50</div>
        </div>
        <div className="p-2.5 bg-amber-50/60 border border-amber-200/60 rounded-xl">
          <div className="text-[10px] uppercase font-bold text-amber-800">P75 Threshold</div>
          <div className="text-sm font-extrabold text-amber-950 mt-0.5">{formatNumber(p75)}</div>
          <div className="text-[10px] text-amber-700">ELEVATED: P50 - P75</div>
        </div>
        <div className="p-2.5 bg-orange-50/60 border border-orange-200/60 rounded-xl">
          <div className="text-[10px] uppercase font-bold text-orange-800">P90 Threshold</div>
          <div className="text-sm font-extrabold text-orange-950 mt-0.5">{formatNumber(p90)}</div>
          <div className="text-[10px] text-orange-700">HIGH: P75 - P90</div>
        </div>
      </div>

      {/* Visual Indicator Bar */}
      <div className="relative pt-6 pb-8 px-2">
        {/* Scale track */}
        <div className="h-3 w-full bg-slate-100 rounded-full relative overflow-hidden flex">
          <div style={{ width: `${posP50}%` }} className="bg-emerald-400/70 h-full" title="Low zone" />
          <div style={{ width: `${posP75 - posP50}%` }} className="bg-amber-400/70 h-full" title="Elevated zone" />
          <div style={{ width: `${posP90 - posP75}%` }} className="bg-orange-400/70 h-full" title="High zone" />
          <div style={{ width: `${100 - posP90}%` }} className="bg-red-400/70 h-full" title="Very High zone" />
        </div>

        {/* P50 Marker */}
        <div 
          style={{ left: `${posP50}%` }} 
          className="absolute top-4 -translate-x-1/2 flex flex-col items-center"
        >
          <div className="w-0.5 h-6 bg-slate-400" />
          <span className="text-[10px] font-bold text-slate-500 mt-1">P50 ({formatNumber(p50)})</span>
        </div>

        {/* P75 Marker */}
        <div 
          style={{ left: `${posP75}%` }} 
          className="absolute top-4 -translate-x-1/2 flex flex-col items-center"
        >
          <div className="w-0.5 h-6 bg-slate-400" />
          <span className="text-[10px] font-bold text-slate-500 mt-1">P75 ({formatNumber(p75)})</span>
        </div>

        {/* P90 Marker */}
        <div 
          style={{ left: `${posP90}%` }} 
          className="absolute top-4 -translate-x-1/2 flex flex-col items-center"
        >
          <div className="w-0.5 h-6 bg-slate-400" />
          <span className="text-[10px] font-bold text-slate-500 mt-1">P90 ({formatNumber(p90)})</span>
        </div>

        {/* Forecast Pointer Marker */}
        <div 
          style={{ left: `${posForecast}%` }} 
          className="absolute top-0 -translate-x-1/2 flex flex-col items-center z-10"
        >
          <div className="px-2 py-0.5 rounded bg-teal-800 text-white text-[10px] font-bold shadow-sm whitespace-nowrap">
            Forecast: {formatNumber(forecast)}
          </div>
          <div className="w-2 h-2 bg-teal-800 rotate-45 -mt-1" />
        </div>
      </div>

      <div className="text-[11px] text-slate-500 mt-2 bg-slate-50 p-2.5 rounded-lg border border-slate-200/60">
        This category compares the predicted weekly case count with the district's own historical distribution.
      </div>
    </div>
  );
}
