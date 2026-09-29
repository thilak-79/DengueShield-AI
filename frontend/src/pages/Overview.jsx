import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  ArrowRight, 
  Building2, 
  MapPin
} from 'lucide-react';
import apiService from '../services/api';
import ActivitySummaryCards from '../components/cards/ActivitySummaryCards';
import CurrentVsForecastChart from '../components/charts/CurrentVsForecastChart';
import PrototypeDisclaimer from '../components/common/PrototypeDisclaimer';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import { ACTIVITY_CONFIG, formatNumber, formatPercent } from '../utils/formatters';

export default function Overview() {
  const navigate = useNavigate();
  const [latestData, setLatestData] = useState(null);
  const [summaryData, setSummaryData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [latestRes, summaryRes] = await Promise.all([
        apiService.getLatestActivity(),
        apiService.getActivitySummary()
      ]);
      setLatestData(latestRes);
      setSummaryData(summaryRes);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading) return <LoadingSpinner message="Fetching national dengue forecast intelligence..." />;
  if (error) return <ErrorMessage error={error} onRetry={fetchData} />;

  // Correct root extraction: endpoint returns district records inside `data` array
  const districts = latestData?.data || [];

  // Calculate National Aggregates from real API records
  const totalCurrentCases = districts.reduce((acc, d) => acc + (d.current_cases || 0), 0);
  const totalForecastCases = districts.reduce((acc, d) => acc + (d.forecast_cases_1w || 0), 0);
  const netChange = totalForecastCases - totalCurrentCases;
  const netChangePct = totalCurrentCases > 0 ? (netChange / totalCurrentCases) * 100 : 0;

  const increasingCount = districts.filter(d => d.forecast_trend === 'INCREASING').length;
  const stableCount = districts.filter(d => d.forecast_trend === 'STABLE').length;
  const decreasingCount = districts.filter(d => d.forecast_trend === 'DECREASING').length;

  // Top 5 Districts sorted by forecast_cases_1w
  const topDistricts = [...districts]
    .sort((a, b) => (b.forecast_cases_1w || 0) - (a.forecast_cases_1w || 0))
    .slice(0, 5);

  const sampleDistrict = districts[0];
  const originDate = latestData?.forecast_origin_date || sampleDistrict?.forecast_origin_date || '2025-12-12';
  const targetDate = sampleDistrict?.target_date || '2025-12-13';

  return (
    <div className="space-y-6">
      
      {/* Top Header Card */}
      <div className="bg-gradient-to-r from-teal-900 via-teal-800 to-slate-900 text-white p-6 rounded-3xl shadow-lg relative overflow-hidden">
        <div className="absolute right-0 top-0 bottom-0 w-1/3 bg-teal-500/10 pointer-events-none skew-x-12" />
        
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-teal-500/20 text-teal-300 border border-teal-400/30">
                HISTORICAL EVALUATION PROTOTYPE
              </span>
              <span className="text-xs text-slate-300">25 Districts</span>
            </div>
            
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">
              DengueShield AI — Activity Intelligence
            </h1>
            <p className="text-xs sm:text-sm text-teal-100/80 max-w-2xl mt-1 leading-relaxed">
              1-Week Random Forest relative activity forecast derived from historical epidemiological surveillance.
            </p>
          </div>

          {/* Quick Meta Badges */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-slate-900">
            <div className="bg-white/95 backdrop-blur-md p-3 rounded-2xl text-center shadow-xs">
              <div className="text-[10px] uppercase font-bold text-slate-400">Forecast Origin</div>
              <div className="text-xs font-extrabold text-slate-800 mt-0.5">{originDate}</div>
            </div>
            <div className="bg-white/95 backdrop-blur-md p-3 rounded-2xl text-center shadow-xs">
              <div className="text-[10px] uppercase font-bold text-slate-400">Target Date</div>
              <div className="text-xs font-extrabold text-slate-800 mt-0.5">{targetDate}</div>
            </div>
            <div className="bg-white/95 backdrop-blur-md p-3 rounded-2xl text-center shadow-xs">
              <div className="text-[10px] uppercase font-bold text-slate-400">Horizon</div>
              <div className="text-xs font-extrabold text-slate-800 mt-0.5">1 Week</div>
            </div>
            <div className="bg-white/95 backdrop-blur-md p-3 rounded-2xl text-center shadow-xs">
              <div className="text-[10px] uppercase font-bold text-slate-400">Model</div>
              <div className="text-xs font-extrabold text-teal-700 mt-0.5">Random Forest</div>
            </div>
          </div>
        </div>
      </div>

      {/* Prototype Scientific Banner */}
      <PrototypeDisclaimer type="general" />

      {/* Activity Summary Cards */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-base font-bold text-slate-900">Relative Dengue Activity Breakdown</h2>
          <span className="text-xs text-slate-500 font-medium">District Percentile Distribution</span>
        </div>
        <ActivitySummaryCards summaryData={summaryData} />
      </section>

      {/* National Aggregates & Top Districts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left 2 Cols: National Aggregates */}
        <div className="lg:col-span-2 space-y-6">
          
          <div className="bg-white p-5 sm:p-6 rounded-2xl border border-slate-200/80 shadow-xs">
            <h2 className="text-base font-bold text-slate-900 mb-4">National Case Aggregates</h2>
            
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/70">
                <div className="text-xs font-semibold text-slate-500 uppercase">Current Total Cases</div>
                <div className="text-2xl font-black text-slate-900 mt-1">{formatNumber(totalCurrentCases)}</div>
                <div className="text-[11px] text-slate-500 mt-1">Sum across 25 districts</div>
              </div>

              <div className="p-4 bg-teal-50/60 rounded-xl border border-teal-200/70">
                <div className="text-xs font-semibold text-teal-800 uppercase">1-Wk RF Predicted Total</div>
                <div className="text-2xl font-black text-teal-950 mt-1">{formatNumber(totalForecastCases)}</div>
                <div className="text-[11px] text-teal-700 mt-1">Random Forest 1-week horizon</div>
              </div>

              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/70">
                <div className="text-xs font-semibold text-slate-500 uppercase">Forecast Net Difference</div>
                <div className={`text-2xl font-black mt-1 ${netChange >= 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                  {netChange >= 0 ? '+' : ''}{formatNumber(netChange)} ({formatPercent(netChangePct, true)})
                </div>
                <div className="text-[11px] text-slate-500 mt-1">Net national difference</div>
              </div>
            </div>

            {/* Trend Direction Breakdown */}
            <div className="border-t border-slate-100 pt-4">
              <div className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-3">District Forecast Trends</div>
              <div className="grid grid-cols-3 gap-3 text-center text-xs">
                <div className="p-3 bg-rose-50/70 rounded-xl border border-rose-200/60 flex items-center justify-center gap-2">
                  <TrendingUp className="w-4 h-4 text-rose-600" />
                  <div>
                    <span className="font-extrabold text-rose-900 text-sm">{increasingCount}</span>
                    <span className="text-rose-700 text-[11px] block">Increasing</span>
                  </div>
                </div>

                <div className="p-3 bg-slate-100/70 rounded-xl border border-slate-200 flex items-center justify-center gap-2">
                  <Minus className="w-4 h-4 text-slate-500" />
                  <div>
                    <span className="font-extrabold text-slate-900 text-sm">{stableCount}</span>
                    <span className="text-slate-600 text-[11px] block">Stable</span>
                  </div>
                </div>

                <div className="p-3 bg-emerald-50/70 rounded-xl border border-emerald-200/60 flex items-center justify-center gap-2">
                  <TrendingDown className="w-4 h-4 text-emerald-600" />
                  <div>
                    <span className="font-extrabold text-emerald-900 text-sm">{decreasingCount}</span>
                    <span className="text-emerald-700 text-[11px] block">Decreasing</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

        </div>

        {/* Right Col: Top 5 Highest Activity Districts */}
        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-base font-bold text-slate-900">Highest Forecast Activity</h2>
                <p className="text-xs text-slate-500">Top 5 districts by 1-week forecast volume</p>
              </div>
              <Building2 className="w-5 h-5 text-slate-400" />
            </div>

            <div className="space-y-3">
              {topDistricts.map((d, i) => {
                const cfg = ACTIVITY_CONFIG[d.activity_level] || ACTIVITY_CONFIG['LOW'];
                return (
                  <div 
                    key={d.district}
                    onClick={() => navigate(`/district?name=${d.district}`)}
                    className="p-3 bg-slate-50 hover:bg-slate-100/80 rounded-xl border border-slate-200/70 cursor-pointer transition-colors flex items-center justify-between group"
                  >
                    <div className="flex items-center gap-3">
                      <span className="w-6 h-6 rounded-full bg-slate-200 text-slate-700 text-xs font-bold flex items-center justify-center">
                        {i + 1}
                      </span>
                      <div>
                        <div className="font-bold text-slate-900 text-sm flex items-center gap-1.5">
                          {d.district}
                          <ArrowRight className="w-3 h-3 text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                        </div>
                        <div className="text-[11px] text-slate-500">
                          Current: <span className="font-semibold text-slate-700">{formatNumber(d.current_cases)}</span>
                        </div>
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="text-sm font-extrabold text-teal-800">
                        {formatNumber(d.forecast_cases_1w)}
                      </div>
                      <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${cfg.badgeBg} ${cfg.badgeText}`}>
                        {d.activity_level}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <button
            onClick={() => navigate('/map')}
            className="w-full mt-4 py-2.5 bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold rounded-xl flex items-center justify-center gap-2 transition-colors"
          >
            <MapPin className="w-4 h-4 text-teal-400" />
            View Interactive 25-District Map
          </button>
        </div>

      </div>

      {/* Main Bar Chart */}
      <section>
        <CurrentVsForecastChart districtsData={districts} />
      </section>

    </div>
  );
}
