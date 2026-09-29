import React, { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { 
  Building2, 
  TrendingUp, 
  TrendingDown, 
  Minus
} from 'lucide-react';
import apiService from '../services/api';
import PercentileScale from '../components/district/PercentileScale';
import ForecastDriversPanel from '../components/district/ForecastDriversPanel';
import PrototypeDisclaimer from '../components/common/PrototypeDisclaimer';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import { ACTIVITY_CONFIG, formatNumber, formatPercent } from '../utils/formatters';

export default function DistrictPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [districtList, setDistrictList] = useState([]);
  const [selectedDistrict, setSelectedDistrict] = useState(searchParams.get('name') || 'Colombo');
  const [districtDetail, setDistrictDetail] = useState(null);
  const [explanationData, setExplanationData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch all 25 district names for dropdown selector
  useEffect(() => {
    apiService.getDistricts()
      .then(res => {
        // Handle both res.districts array or direct array
        const list = res?.districts || (Array.isArray(res) ? res : []);
        if (list.length > 0) {
          setDistrictList(list);
        }
      })
      .catch(err => console.error('Failed to load district list:', err));
  }, []);

  // Fetch district details and explanation when selected District changes
  const fetchDistrictData = async (districtName) => {
    setLoading(true);
    setError(null);
    try {
      const [detailRes, expRes] = await Promise.all([
        apiService.getDistrictDetail(districtName),
        apiService.getDistrictExplanation(districtName).catch(() => null)
      ]);
      setDistrictDetail(detailRes?.district || detailRes);
      setExplanationData(expRes);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedDistrict) {
      fetchDistrictData(selectedDistrict);
    }
  }, [selectedDistrict]);

  const handleDistrictChange = (e) => {
    const newName = e.target.value;
    setSelectedDistrict(newName);
    setSearchParams({ name: newName });
  };

  if (loading) return <LoadingSpinner message={`Loading forecast intelligence for ${selectedDistrict}...`} />;
  if (error) return <ErrorMessage error={error} onRetry={() => fetchDistrictData(selectedDistrict)} />;

  const d = districtDetail;
  const cfg = ACTIVITY_CONFIG[d?.activity_level] || ACTIVITY_CONFIG['LOW'];

  const trendIcon = d?.forecast_trend === 'INCREASING' ? TrendingUp :
                    d?.forecast_trend === 'DECREASING' ? TrendingDown : Minus;
  const TrendIconComp = trendIcon;

  return (
    <div className="space-y-6">
      
      {/* Top Selector Card */}
      <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-900 flex items-center gap-2">
            <Building2 className="w-5 h-5 text-teal-700" />
            District Activity Explorer
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Detailed 1-week dengue activity forecast and SHAP driver analysis
          </p>
        </div>

        {/* Dropdown Selector */}
        <div className="flex items-center gap-2">
          <label htmlFor="district-select" className="text-xs font-semibold text-slate-600 shrink-0">
            Select District:
          </label>
          <div className="relative">
            <select
              id="district-select"
              value={selectedDistrict}
              onChange={handleDistrictChange}
              className="bg-slate-50 border border-slate-300 font-bold text-slate-800 text-xs sm:text-sm rounded-xl px-3 py-2 pr-8 focus:outline-hidden focus:ring-2 focus:ring-teal-500 cursor-pointer"
            >
              {districtList.length > 0 ? (
                districtList.map(name => (
                  <option key={name} value={name}>{name}</option>
                ))
              ) : (
                <option value={selectedDistrict}>{selectedDistrict}</option>
              )}
            </select>
          </div>
        </div>
      </div>

      <PrototypeDisclaimer type="general" />

      {/* Main Metric Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        
        {/* Card 1: Current Cases */}
        <div className="p-4 bg-white rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="text-[11px] font-semibold text-slate-500 uppercase">Current Cases</div>
          <div className="text-2xl font-black text-slate-900 mt-1">{formatNumber(d?.current_cases)}</div>
          <div className="text-[10px] text-slate-400 mt-0.5">Reported weekly</div>
        </div>

        {/* Card 2: 1-Week RF Forecast */}
        <div className="p-4 bg-teal-50/70 rounded-2xl border border-teal-200 shadow-xs">
          <div className="text-[11px] font-bold text-teal-800 uppercase">1-Wk RF Forecast</div>
          <div className="text-2xl font-black text-teal-950 mt-1">{formatNumber(d?.forecast_cases_1w)}</div>
          <div className="text-[10px] text-teal-700 mt-0.5">Random Forest model</div>
        </div>

        {/* Card 3: Persistence Benchmark */}
        <div className="p-4 bg-white rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="text-[11px] font-semibold text-slate-500 uppercase">Persistence</div>
          <div className="text-2xl font-bold text-slate-700 mt-1">{formatNumber(d?.persistence_forecast_1w)}</div>
          <div className="text-[10px] text-slate-400 mt-0.5">Simple baseline</div>
        </div>

        {/* Card 4: Forecast Change % */}
        <div className="p-4 bg-white rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="text-[11px] font-semibold text-slate-500 uppercase">Forecast Change</div>
          <div className={`text-xl font-extrabold mt-1 ${(d?.forecast_percent_change || 0) >= 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
            {formatPercent(d?.forecast_percent_change, true)}
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5">Vs current week</div>
        </div>

        {/* Card 5: Trend */}
        <div className="p-4 bg-white rounded-2xl border border-slate-200/80 shadow-xs flex flex-col justify-between">
          <div className="text-[11px] font-semibold text-slate-500 uppercase">Trend</div>
          <div className="flex items-center gap-1.5 font-extrabold text-slate-800 text-sm mt-1">
            <TrendIconComp className={`w-4 h-4 ${(d?.forecast_trend === 'INCREASING') ? 'text-rose-500' : 'text-emerald-500'}`} />
            <span>{d?.forecast_trend || 'STABLE'}</span>
          </div>
          <div className="text-[10px] text-slate-400">Directional vector</div>
        </div>

        {/* Card 6: Relative Activity Badge */}
        <div className={`p-4 rounded-2xl border ${cfg.bg} ${cfg.border} shadow-xs flex flex-col justify-between`}>
          <div className="text-[10px] font-bold tracking-wider uppercase text-slate-600">Relative Activity</div>
          <div className={`text-base font-black ${cfg.text} mt-1`}>
            {d?.activity_level || 'LOW'}
          </div>
          <div className="text-[10px] text-slate-500">Historical percentile</div>
        </div>

      </div>

      {/* Historical Percentiles Visual Scale */}
      <PercentileScale districtDetail={d} />

      {/* Top 3 Forecast Drivers (SHAP Panel) */}
      <ForecastDriversPanel districtDetail={d} explanationData={explanationData} />

    </div>
  );
}
