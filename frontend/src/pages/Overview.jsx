import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  ArrowRight, 
  Building2, 
  MapPin,
  Sparkles,
  Calendar,
  HelpCircle
} from 'lucide-react';
import apiService from '../services/api';
import { useMode } from '../context/ModeContext';
import ActivitySummaryCards from '../components/cards/ActivitySummaryCards';
import CurrentVsForecastChart from '../components/charts/CurrentVsForecastChart';
import PrototypeDisclaimer from '../components/common/PrototypeDisclaimer';
import CurrentInferenceWarning from '../components/common/CurrentInferenceWarning';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import { ACTIVITY_CONFIG, formatNumber, formatPercent } from '../utils/formatters';

export default function Overview() {
  const navigate = useNavigate();
  const { isCurrent } = useMode();
  const [latestData, setLatestData] = useState(null);
  const [summaryData, setSummaryData] = useState(null);
  const [statusData, setStatusData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      if (isCurrent) {
        const [statusRes, activityRes, summaryRes] = await Promise.all([
          apiService.getCurrentStatus(),
          apiService.getCurrentActivity(),
          apiService.getCurrentSummary()
        ]);
        setStatusData(statusRes?.data || statusRes);
        setLatestData(activityRes);
        setSummaryData(summaryRes);
      } else {
        const [latestRes, summaryRes] = await Promise.all([
          apiService.getLatestActivity(),
          apiService.getActivitySummary()
        ]);
        setLatestData(latestRes);
        setSummaryData(summaryRes);
      }
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [isCurrent]);

  if (loading) return <LoadingSpinner message={isCurrent ? "Fetching 2026 operational current dengue intelligence..." : "Fetching national dengue forecast intelligence..."} />;
  if (error) return <ErrorMessage error={error} onRetry={fetchData} />;

  // Extract district records array
  const districts = latestData?.data || [];

  // Calculate National Aggregates from status endpoint or district records
  const totalCurrentCases = isCurrent && statusData?.current_total_cases !== undefined
    ? statusData.current_total_cases
    : districts.reduce((acc, d) => acc + (d.current_cases || 0), 0);

  const totalForecastCases = isCurrent && statusData?.forecast_total_cases !== undefined
    ? statusData.forecast_total_cases
    : districts.reduce((acc, d) => acc + (d.forecast_cases_1w || 0), 0);

  const netChange = totalForecastCases - totalCurrentCases;
  const netChangePct = totalCurrentCases > 0 ? (netChange / totalCurrentCases) * 100 : 0;

  // Trend Counts (including UNKNOWN to guarantee all 25 districts are accounted for)
  let increasingCount = 0;
  let stableCount = 0;
  let decreasingCount = 0;
  let unknownCount = 0;

  if (isCurrent && statusData?.trend_counts) {
    increasingCount = statusData.trend_counts.INCREASING || 0;
    stableCount = statusData.trend_counts.STABLE || 0;
    decreasingCount = statusData.trend_counts.DECREASING || 0;
    unknownCount = statusData.trend_counts.UNKNOWN || 0;
  } else {
    increasingCount = districts.filter(d => d.forecast_trend === 'INCREASING').length;
    stableCount = districts.filter(d => d.forecast_trend === 'STABLE').length;
    decreasingCount = districts.filter(d => d.forecast_trend === 'DECREASING').length;
    unknownCount = districts.filter(d => d.forecast_trend === 'UNKNOWN' || !['INCREASING', 'STABLE', 'DECREASING'].includes(d.forecast_trend)).length;
  }

  // Top 5 Districts sorted by forecast_cases_1w
  const topDistricts = [...districts]
    .sort((a, b) => (b.forecast_cases_1w || 0) - (a.forecast_cases_1w || 0))
    .slice(0, 5);

  const sampleDistrict = districts[0];
  
  // Dates
  const originDate = isCurrent
    ? (statusData?.data_freshness_date || sampleDistrict?.forecast_origin_date || '2026-09-13')
    : (latestData?.forecast_origin_date || sampleDistrict?.forecast_origin_date || '2025-12-12');

  const surveillanceWindow = isCurrent && statusData?.surveillance_window_start && statusData?.surveillance_window_end
    ? `${statusData.surveillance_window_start} – ${statusData.surveillance_window_end}`
    : '07 Sep 2026 – 13 Sep 2026';

  const forecastWindow = isCurrent && statusData?.forecast_target_start && statusData?.forecast_target_end
    ? `${statusData.forecast_target_start} – ${statusData.forecast_target_end}`
    : '14 Sep 2026 – 20 Sep 2026';

  return (
    <div className="space-y-6">
      
      {/* Top Header Card */}
      <div className={`p-6 rounded-3xl shadow-lg relative overflow-hidden text-white ${
        isCurrent 
          ? 'bg-gradient-to-r from-amber-950 via-slate-900 to-amber-900 border border-amber-500/30' 
          : 'bg-gradient-to-r from-teal-900 via-teal-800 to-slate-900'
      }`}>
        <div className="absolute right-0 top-0 bottom-0 w-1/3 bg-amber-500/5 pointer-events-none skew-x-12" />
        
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-extrabold ${
                isCurrent 
                  ? 'bg-amber-400 text-slate-950 border border-amber-300' 
                  : 'bg-teal-500/20 text-teal-300 border border-teal-400/30'
              }`}>
                {isCurrent ? 'EXPERIMENTAL CURRENT INFERENCE (2026)' : 'HISTORICAL EVALUATION PROTOTYPE'}
              </span>
              <span className="text-xs text-slate-300">25 Districts</span>
            </div>
            
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">
              {isCurrent ? 'Experimental 2026 Dengue Activity Intelligence' : 'DengueShield AI — Activity Intelligence'}
            </h1>
            <p className="text-xs sm:text-sm text-slate-200/90 max-w-2xl mt-1 leading-relaxed">
              {isCurrent 
                ? 'Experimental 1-week Random Forest refit inference using NDCU surveillance and weather observations available through the forecast origin.'
                : '1-Week Random Forest relative activity forecast derived from historical epidemiological surveillance.'
              }
            </p>
          </div>

          {/* Quick Meta Badges */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-slate-900">
            <div className="bg-white/95 backdrop-blur-md p-3 rounded-2xl text-center shadow-xs">
              <div className="text-[10px] uppercase font-bold text-slate-400">
                {isCurrent ? 'Surveillance Wk' : 'Forecast Origin'}
              </div>
              <div className="text-xs font-extrabold text-slate-800 mt-0.5">
                {isCurrent ? `Week ${statusData?.latest_week || 37}` : originDate}
              </div>
            </div>

            <div className="bg-white/95 backdrop-blur-md p-3 rounded-2xl text-center shadow-xs">
              <div className="text-[10px] uppercase font-bold text-slate-400">
                {isCurrent ? 'Freshness Date' : 'Target Date'}
              </div>
              <div className="text-xs font-extrabold text-slate-800 mt-0.5">
                {isCurrent ? (statusData?.data_freshness_date || '2026-09-13') : (sampleDistrict?.target_date || '2025-12-13')}
              </div>
            </div>

            <div className="bg-white/95 backdrop-blur-md p-3 rounded-2xl text-center shadow-xs">
              <div className="text-[10px] uppercase font-bold text-slate-400">Horizon</div>
              <div className="text-xs font-extrabold text-slate-800 mt-0.5">1 Week</div>
            </div>

            <div className="bg-white/95 backdrop-blur-md p-3 rounded-2xl text-center shadow-xs">
              <div className="text-[10px] uppercase font-bold text-slate-400">Model</div>
              <div className="text-xs font-extrabold text-teal-700 mt-0.5">
                {isCurrent ? 'RF Refit' : 'Random Forest'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Prominent Warning Banners */}
      {isCurrent ? (
        <CurrentInferenceWarning statusData={statusData} />
      ) : (
        <PrototypeDisclaimer type="general" />
      )}

      {/* Mode-specific Context Details */}
      {isCurrent && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs bg-white p-4 rounded-2xl border border-slate-200 shadow-2xs">
          <div className="flex items-center gap-3">
            <Calendar className="w-5 h-5 text-amber-600 shrink-0" />
            <div>
              <span className="font-bold text-slate-900 block">Surveillance Window:</span>
              <span className="text-slate-600">{surveillanceWindow} (NDCU Reported Cases)</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Sparkles className="w-5 h-5 text-amber-600 shrink-0" />
            <div>
              <span className="font-bold text-slate-900 block">Forecast Target Window:</span>
              <span className="text-amber-950 font-bold">{forecastWindow} (1-Wk RF Refit)</span>
            </div>
          </div>
        </div>
      )}

      {/* Relative Activity Summary Cards */}
      <section>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-base font-bold text-slate-900">
            {isCurrent ? 'Current 2026 Relative Activity Breakdown' : 'Relative Dengue Activity Breakdown'}
          </h2>
          <span className="text-xs text-slate-500 font-medium">District Percentile Distribution</span>
        </div>
        <ActivitySummaryCards summaryData={summaryData} districts={districts} />
      </section>

      {/* National Aggregates & Top Districts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left 2 Cols: National Aggregates */}
        <div className="lg:col-span-2 space-y-6">
          
          <div className="bg-white p-5 sm:p-6 rounded-2xl border border-slate-200/80 shadow-xs">
            <h2 className="text-base font-bold text-slate-900 mb-4">National Case Aggregates</h2>
            
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/70">
                <div className="text-xs font-semibold text-slate-500 uppercase">
                  {isCurrent ? 'Current Surveillance Cases' : 'Current Total Cases'}
                </div>
                <div className="text-2xl font-black text-slate-900 mt-1">{formatNumber(totalCurrentCases)}</div>
                <div className="text-[11px] text-slate-500 mt-1">Sum across 25 districts</div>
              </div>

              <div className="p-4 bg-teal-50/60 rounded-xl border border-teal-200/70">
                <div className="text-xs font-semibold text-teal-800 uppercase">
                  {isCurrent ? 'Forecast Total Cases' : '1-Wk RF Predicted Total'}
                </div>
                <div className="text-2xl font-black text-teal-950 mt-1">{formatNumber(totalForecastCases)}</div>
                <div className="text-[11px] text-teal-700 mt-1">Random Forest 1-week horizon</div>
              </div>

              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/70">
                <div className="text-xs font-semibold text-slate-500 uppercase">
                  {isCurrent ? 'Model-Estimated Difference' : 'Forecast Net Difference'}
                </div>
                <div className={`text-2xl font-black mt-1 ${netChange >= 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                  {netChange >= 0 ? '+' : ''}{formatNumber(netChange)} ({formatPercent(netChangePct, true)})
                </div>
                <div className="text-[11px] text-slate-500 mt-1">
                  {isCurrent ? 'Model-estimated difference' : 'Net national difference'}
                </div>
              </div>
            </div>

            {/* District Forecast Trend Breakdown (All 25 Districts Accounted For) */}
            <div className="border-t border-slate-100 pt-4">
              <div className="flex items-center justify-between mb-3">
                <div className="text-xs font-bold text-slate-700 uppercase tracking-wider">District Forecast Trends</div>
                <div className="text-[11px] text-slate-500 font-mono">
                  Total: {increasingCount + stableCount + decreasingCount + unknownCount} Districts
                </div>
              </div>

              <div className={`grid gap-3 text-center text-xs ${unknownCount > 0 ? 'grid-cols-2 sm:grid-cols-4' : 'grid-cols-3'}`}>
                <div className="p-3 bg-rose-50/70 rounded-xl border border-rose-200/60 flex items-center justify-center gap-2">
                  <TrendingUp className="w-4 h-4 text-rose-600 shrink-0" />
                  <div>
                    <span className="font-extrabold text-rose-900 text-sm">{increasingCount}</span>
                    <span className="text-rose-700 text-[11px] block">Increasing</span>
                  </div>
                </div>

                <div className="p-3 bg-slate-100/70 rounded-xl border border-slate-200 flex items-center justify-center gap-2">
                  <Minus className="w-4 h-4 text-slate-500 shrink-0" />
                  <div>
                    <span className="font-extrabold text-slate-900 text-sm">{stableCount}</span>
                    <span className="text-slate-600 text-[11px] block">Stable</span>
                  </div>
                </div>

                <div className="p-3 bg-emerald-50/70 rounded-xl border border-emerald-200/60 flex items-center justify-center gap-2">
                  <TrendingDown className="w-4 h-4 text-emerald-600 shrink-0" />
                  <div>
                    <span className="font-extrabold text-emerald-900 text-sm">{decreasingCount}</span>
                    <span className="text-emerald-700 text-[11px] block">Decreasing</span>
                  </div>
                </div>

                {unknownCount > 0 && (
                  <div className="p-3 bg-amber-50/70 rounded-xl border border-amber-200 flex items-center justify-center gap-2">
                    <HelpCircle className="w-4 h-4 text-amber-600 shrink-0" />
                    <div>
                      <span className="font-extrabold text-amber-950 text-sm">{unknownCount}</span>
                      <span className="text-amber-800 text-[11px] block">Unknown</span>
                    </div>
                  </div>
                )}
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
                const actLevel = d.relative_activity || d.activity_level || 'LOW';
                const cfg = ACTIVITY_CONFIG[actLevel] || ACTIVITY_CONFIG['LOW'];
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
                        {actLevel}
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
