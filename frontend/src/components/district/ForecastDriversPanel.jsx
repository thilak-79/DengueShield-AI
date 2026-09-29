import React from 'react';
import { ArrowUpRight, ArrowDownRight, Info, HelpCircle } from 'lucide-react';
import { getFriendlyFeatureName, formatNumber } from '../../utils/formatters';

export default function ForecastDriversPanel({ districtDetail, explanationData }) {
  if (!districtDetail) return null;

  // Process top 3 drivers from detail or explanation payload
  let drivers = [];

  if (explanationData?.drivers && explanationData.drivers.length > 0) {
    drivers = explanationData.drivers.slice(0, 3).map(d => ({
      technical: d.feature,
      friendly: getFriendlyFeatureName(d.feature),
      value: d.feature_value,
      shap: d.shap_value,
      direction: d.direction || (d.shap_value >= 0 ? 'increases_prediction' : 'decreases_prediction')
    }));
  } else {
    // Fallback to top_driver_1, top_driver_2, top_driver_3 in districtDetail
    for (let i = 1; i <= 3; i++) {
      if (districtDetail[`top_driver_${i}_technical`]) {
        drivers.push({
          technical: districtDetail[`top_driver_${i}_technical`],
          friendly: districtDetail[`top_driver_${i}`] || getFriendlyFeatureName(districtDetail[`top_driver_${i}_technical`]),
          value: districtDetail[`top_driver_${i}_value`],
          shap: districtDetail[`top_driver_${i}_shap`],
          direction: districtDetail[`top_driver_${i}_direction`]
        });
      }
    }
  }

  return (
    <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-bold text-slate-900">Key Model Drivers (SHAP Analysis)</h3>
          <p className="text-xs text-slate-500">Factors influencing this model prediction for {districtDetail.district}</p>
        </div>
        <span className="text-[10px] font-mono bg-teal-50 text-teal-800 border border-teal-200 px-2 py-0.5 rounded">
          Local SHAP
        </span>
      </div>

      {drivers.length === 0 ? (
        <div className="p-4 bg-slate-50 rounded-xl text-center text-xs text-slate-500">
          No local feature driver data available for this district.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
          {drivers.map((drv, idx) => {
            const isIncrease = drv.direction === 'increases_prediction' || drv.shap >= 0;
            const ArrowIcon = isIncrease ? ArrowUpRight : ArrowDownRight;
            const badgeColor = isIncrease 
              ? 'bg-rose-50 text-rose-700 border-rose-200' 
              : 'bg-emerald-50 text-emerald-700 border-emerald-200';

            return (
              <div 
                key={idx} 
                className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 hover:bg-slate-50 transition-colors relative group"
              >
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="flex items-center gap-1.5 text-xs font-bold text-slate-900">
                    <span>#{idx + 1} {drv.friendly}</span>
                  </div>
                  <div 
                    className="text-slate-400 hover:text-slate-600 cursor-help"
                    title={`Raw feature symbol: ${drv.technical}`}
                  >
                    <HelpCircle className="w-3.5 h-3.5" />
                  </div>
                </div>

                <div className="text-xs text-slate-500 mb-3">
                  Observed Value: <span className="font-mono font-semibold text-slate-800">{formatNumber(drv.value, 1)}</span>
                </div>

                <div className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-bold border ${badgeColor}`}>
                  <ArrowIcon className="w-3.5 h-3.5" />
                  <span>
                    {isIncrease ? '+' : ''}{formatNumber(drv.shap, 1)} SHAP
                  </span>
                  <span className="text-[10px] font-normal opacity-80">
                    ({isIncrease ? 'Increases' : 'Decreases'})
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="p-3 bg-amber-50/70 border border-amber-200/80 rounded-xl text-xs text-amber-900 flex items-start gap-2">
        <Info className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
        <div>
          <strong>Scientific Notice:</strong> SHAP explains the model's prediction, not causal relationships. Features identified contribute mathematically to the Random Forest prediction score.
        </div>
      </div>
    </div>
  );
}
