import React from 'react';
import { ACTIVITY_CONFIG } from '../../utils/formatters';
import { AlertCircle, AlertTriangle, CheckCircle2, Flame } from 'lucide-react';

const ICONS = {
  'VERY HIGH': Flame,
  'HIGH': AlertTriangle,
  'ELEVATED': AlertCircle,
  'LOW': CheckCircle2
};

export default function ActivitySummaryCards({ summaryData }) {
  const counts = summaryData?.counts || { 'VERY HIGH': 0, 'HIGH': 7, 'ELEVATED': 11, 'LOW': 7 };
  const percentages = summaryData?.percentages || { 'VERY HIGH': 0, 'HIGH': 28, 'ELEVATED': 44, 'LOW': 28 };

  const CATEGORIES = ['VERY HIGH', 'HIGH', 'ELEVATED', 'LOW'];

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {CATEGORIES.map((cat) => {
          const cfg = ACTIVITY_CONFIG[cat];
          const Icon = ICONS[cat];
          const count = counts[cat] ?? 0;
          const pct = percentages[cat] ?? 0;

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

              <p className="text-[11px] text-slate-500 mt-3 leading-snug line-clamp-2">
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
