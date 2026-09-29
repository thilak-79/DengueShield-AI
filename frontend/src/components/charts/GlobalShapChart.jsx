import React from 'react';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid, 
  Cell 
} from 'recharts';
import { getFriendlyFeatureName, formatNumber } from '../../utils/formatters';

export default function GlobalShapChart({ globalExplanationData }) {
  if (!globalExplanationData?.features) return null;

  const data = globalExplanationData.features.slice(0, 12).map(f => ({
    technical: f.feature_group,
    friendly: getFriendlyFeatureName(f.feature_group),
    meanAbsShap: f.mean_abs_shap,
    signedShap: f.mean_signed_shap,
    rank: f.rank
  }));

  return (
    <div className="bg-white p-5 sm:p-6 rounded-2xl border border-slate-200/80 shadow-xs">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-base font-bold text-slate-900">Global Feature Importance (Mean |SHAP|)</h3>
          <p className="text-xs text-slate-500">Top features driving one-week district-level Random Forest predictions</p>
        </div>
        <span className="text-xs font-semibold text-slate-500 bg-slate-100 px-2.5 py-1 rounded-lg">
          Top {data.length} Features
        </span>
      </div>

      <div className="h-96 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart 
            layout="vertical" 
            data={data} 
            margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
            <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} />
            <YAxis 
              type="category" 
              dataKey="friendly" 
              tick={{ fontSize: 11, fill: '#334155' }} 
              width={160}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#0f172a', 
                borderColor: '#1e293b', 
                borderRadius: '0.75rem',
                color: '#fff',
                fontSize: '12px'
              }}
              formatter={(val) => [`${formatNumber(val, 2)} Mean |SHAP|`, 'Importance']}
              labelFormatter={(label, items) => {
                const item = items[0]?.payload;
                return `${label} (${item?.technical || ''})`;
              }}
            />
            <Bar dataKey="meanAbsShap" fill="#0d9488" radius={[0, 4, 4, 0]} maxBarSize={22}>
              {data.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={index < 3 ? '#0f766e' : '#14b8a6'} 
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-xs text-slate-500 mt-3 text-center">
        Features with larger SHAP values have a greater overall influence on the model's predictions.
      </p>
    </div>
  );
}
