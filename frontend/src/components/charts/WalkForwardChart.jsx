import React, { useState } from 'react';
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Legend, 
  CartesianGrid 
} from 'recharts';
import { formatNumber } from '../../utils/formatters';

export default function WalkForwardChart({ walkForwardDetails }) {
  const [metric, setMetric] = useState('mae'); // 'mae' | 'rmse' | 'smape'

  if (!walkForwardDetails?.data) return null;

  // Filter 1-week horizon details and group by year
  const details1w = walkForwardDetails.data.filter(d => d.horizon_weeks === 1);
  const years = [2020, 2021, 2022, 2023, 2024, 2025];

  const chartData = years.map(yr => {
    const yrRows = details1w.filter(d => d.test_year === yr);
    const rowObj = { year: yr };
    yrRows.forEach(r => {
      rowObj[r.model] = r[metric];
    });
    return rowObj;
  });

  return (
    <div className="bg-white p-5 sm:p-6 rounded-2xl border border-slate-200/80 shadow-xs">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
        <div>
          <h3 className="text-base font-bold text-slate-900">Multi-Year Walk-Forward Evaluation (2020–2025)</h3>
          <p className="text-xs text-slate-500">1-Week Horizon Out-of-Sample Performance by Year</p>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-500 font-medium">Metric:</span>
          <select 
            value={metric} 
            onChange={(e) => setMetric(e.target.value)}
            className="bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1 text-slate-700 font-semibold focus:outline-hidden focus:ring-2 focus:ring-teal-500"
          >
            <option value="mae">MAE (Mean Absolute Error)</option>
            <option value="rmse">RMSE (Root Mean Square Error)</option>
            <option value="smape">sMAPE (% Error)</option>
          </select>
        </div>
      </div>

      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
            <XAxis dataKey="year" tick={{ fontSize: 12, fill: '#64748b' }} />
            <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#0f172a', 
                borderColor: '#1e293b', 
                borderRadius: '0.75rem',
                color: '#fff',
                fontSize: '12px'
              }}
              formatter={(val, name) => [
                formatNumber(val, 2), 
                name === 'persistence' ? 'Persistence Baseline' :
                name === 'random_forest' ? 'Random Forest ML' :
                name === 'lightgbm' ? 'LightGBM ML' : name
              ]}
              labelFormatter={(label) => `Test Evaluation Year: ${label}`}
            />
            <Legend 
              verticalAlign="top" 
              align="right"
              wrapperStyle={{ fontSize: '12px', paddingBottom: '12px' }}
            />
            <Line 
              type="monotone" 
              dataKey="persistence" 
              name="Persistence Baseline" 
              stroke="#64748b" 
              strokeDasharray="4 4" 
              strokeWidth={2} 
              dot={{ r: 4 }}
            />
            <Line 
              type="monotone" 
              dataKey="random_forest" 
              name="Random Forest" 
              stroke="#0d9488" 
              strokeWidth={3} 
              dot={{ r: 5 }}
            />
            <Line 
              type="monotone" 
              dataKey="lightgbm" 
              name="LightGBM" 
              stroke="#8b5cf6" 
              strokeWidth={2} 
              dot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="text-xs text-slate-600 mt-4 bg-slate-50 p-3 rounded-xl border border-slate-200/60 leading-relaxed">
        <strong>Key Research Finding:</strong> Persistence achieved lower 1-week MAE in 2020, 2021 and 2023, while Random Forest achieved lower MAE in 2022, 2024 and 2025. Random Forest beats persistence in 3 out of 6 evaluation years.
      </div>
    </div>
  );
}
