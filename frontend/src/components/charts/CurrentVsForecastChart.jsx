import React, { useState } from 'react';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Legend, 
  CartesianGrid 
} from 'recharts';
import { ACTIVITY_CONFIG } from '../../utils/formatters';

export default function CurrentVsForecastChart({ districtsData }) {
  const [sortBy, setSortBy] = useState('current'); // 'current' | 'forecast' | 'name'

  if (!districtsData || districtsData.length === 0) return null;

  // Format and sort data
  const chartData = districtsData.map(d => ({
    name: d.district,
    current: Math.round(d.current_cases || 0),
    forecast: Math.round(d.forecast_cases_1w || 0),
    persistence: Math.round(d.persistence_forecast_1w || 0),
    activity: d.activity_level || 'LOW'
  })).sort((a, b) => {
    if (sortBy === 'forecast') return b.forecast - a.forecast;
    if (sortBy === 'name') return a.name.localeCompare(b.name);
    return b.current - a.current;
  });

  return (
    <div className="bg-white p-5 sm:p-6 rounded-2xl border border-slate-200/80 shadow-xs">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
        <div>
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            District Dengue Activity Comparison
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Current Weekly Reported Cases vs. 1-Week Random Forest Forecast across 25 Districts
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-500 font-medium">Sort by:</span>
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value)}
            className="bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1 text-slate-700 font-medium focus:outline-hidden focus:ring-2 focus:ring-teal-500"
          >
            <option value="current">Current Cases (High to Low)</option>
            <option value="forecast">1-Wk Forecast (High to Low)</option>
            <option value="name">District Name (A-Z)</option>
          </select>
        </div>
      </div>

      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
            <XAxis 
              dataKey="name" 
              angle={-45} 
              textAnchor="end" 
              interval={0} 
              tick={{ fontSize: 10, fill: '#64748b' }} 
            />
            <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#0f172a', 
                borderColor: '#1e293b', 
                borderRadius: '0.75rem',
                color: '#fff',
                fontSize: '12px',
                boxShadow: '0 10px 15px -3px rgba(0,0,0,0.3)'
              }}
              formatter={(value, name) => [
                `${value} cases`, 
                name === 'current' ? 'Current Reported' : '1-Week RF Forecast'
              ]}
              labelFormatter={(label) => `District: ${label}`}
            />
            <Legend 
              verticalAlign="top" 
              align="right"
              wrapperStyle={{ fontSize: '12px', paddingBottom: '12px' }}
            />
            <Bar 
              dataKey="current" 
              name="Current Cases" 
              fill="#94a3b8" 
              radius={[4, 4, 0, 0]} 
              maxBarSize={28} 
            />
            <Bar 
              dataKey="forecast" 
              name="1-Week RF Forecast" 
              fill="#0d9488" 
              radius={[4, 4, 0, 0]} 
              maxBarSize={28} 
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
