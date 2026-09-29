import React from 'react';
import { ACTIVITY_CONFIG } from '../../utils/formatters';
import { AlertCircle, AlertTriangle, CheckCircle2, Flame } from 'lucide-react';

const ICONS = {
  'VERY HIGH': Flame,
  'HIGH': AlertTriangle,
  'ELEVATED': AlertCircle,
  'LOW': CheckCircle2
};

export default function ActivitySummaryCards({ summaryData, districts = [] }) {
  // Initialize counts map for the 4 relative activity levels
  const counts = { 'VERY HIGH': 0, 'HIGH': 0, 'ELEVATED': 0, 'LOW': 0 };

  // 1. Try summaryData.national.activity_counts (Current mode API payload)
  if (summaryData?.national?.activity_counts) {
    const ac = summaryData.national.activity_counts;
    counts['VERY HIGH'] = ac['VERY HIGH'] || 0;
    counts['HIGH'] = ac['HIGH'] || 0;
    counts['ELEVATED'] = ac['ELEVATED'] || 0;
    counts['LOW'] = ac['LOW'] || 0;
  }
  // 2. Try summaryData.summary or summaryData.data array (Historical or Current summary array)
  else if (Array.isArray(summaryData?.summary) || Array.isArray(summaryData?.data)) {
    const arr = summaryData.summary || summaryData.data;
    arr.forEach(item => {
      const level = item.activity_level;
      if (level && counts[level] !== undefined) {
        counts[level] = Number(item.districts || 0);
      }
    });
  }
  // 3. Fallback: calculate directly from district array
  else if (districts && districts.length > 0) {
    districts.forEach(d => {
      const level = d.relative_activity || d.activity_level;
      if (level && counts[level] !== undefined) {
        counts[level]++;
      }
    });
  }

  // Calculate total districts across categories
  const totalDistricts = Object.values(counts).reduce((a, b) => a + b, 0) || 25;

  const CATEGORIES = ['VERY HIGH', 'HIGH', 'ELEVATED', 'LOW'];

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {CATEGORIES.map((cat) => {
          const cfg = ACTIVITY_CONFIG[cat];
          const Icon = ICONS[cat];
          const count = counts[cat] ?? 0;
          const pct = totalDistricts > 0 ? (count / totalDistricts) * 100 : 0;

          return (
            <div 
              key={cat} 
              className={`p-5 rounded-2xl border ${cfg.bg} ${cfg.border} shadow-xs transition-all hover:shadow-md`}
            >
              <div className="flex items-center justify-between mb-3">
                <span className={`px-2.5 py-1 rounded-full text-[11px] font-bold tracking-wider uppercase ${cfg.badgeBg} ${cfg.badgeText}`}>
                  {cat}
                </span>
                <div className={`p-2 rounded-xl ${cfg.badgeBg}`}>
                  <Icon className={`w-4 h-4 ${cfg.badgeText}`} />
                </div>
              </div>

              <div className="flex items-baseline justify-between">
                <div>
                  <div className="text-3xl font-extrabold text-slate-900 tracking-tight">
                    {count}
                  </div>
                  <div className="text-xs text-slate-600 font-medium mt-0.5">
                    Districts ({pct.toFixed(0)}%)
                  </div>
                </div>
              </div>

              <p className="text-xs font-semibold text-slate-600 mt-3 font-mono">
                {cfg.description}
              </p>
            </div>
          );
        })}
      </div>

      <div className="text-[11px] text-slate-500 text-right italic">
        * Relative activity categories are calculated based on each district's individual historical case distribution percentiles.
      </div>
    </div>
  );
}
