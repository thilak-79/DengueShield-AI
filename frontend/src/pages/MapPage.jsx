import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { MapPin, ArrowRight } from 'lucide-react';
import apiService from '../services/api';
import { useMode } from '../context/ModeContext';
import DistrictMap from '../components/map/DistrictMap';
import PrototypeDisclaimer from '../components/common/PrototypeDisclaimer';
import CurrentInferenceWarning from '../components/common/CurrentInferenceWarning';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import { ACTIVITY_CONFIG, formatNumber, normalizeDistrictName } from '../utils/formatters';

export default function MapPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { isCurrent } = useMode();
  const [latestData, setLatestData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const initialDistrict = searchParams.get('district') || 'Colombo';
  const [selectedDistrictName, setSelectedDistrictName] = useState(initialDistrict);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = isCurrent 
        ? await apiService.getCurrentActivity()
        : await apiService.getLatestActivity();
      setLatestData(data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [isCurrent]);

  if (loading) return <LoadingSpinner message={isCurrent ? "Loading 2026 current spatial activity map..." : "Loading Sri Lanka district spatial map data..."} />;
  if (error) return <ErrorMessage error={error} onRetry={fetchData} />;

  // Extract district array from latestData.data
  const districts = latestData?.data || [];
  
  const selectedDistrictData = districts.find(d => 
    normalizeDistrictName(d.district) === normalizeDistrictName(selectedDistrictName)
  ) || districts[0];

  const handleSelectDistrict = (name) => {
    setSelectedDistrictName(name);
  };

  const actLevel = selectedDistrictData?.relative_activity || selectedDistrictData?.activity_level || 'LOW';
  const selCfg = ACTIVITY_CONFIG[actLevel] || ACTIVITY_CONFIG['LOW'];

  return (
    <div className="space-y-6">
      
      {/* Header Info */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs">
        <div>
          <h1 className="text-xl font-bold text-slate-900 flex items-center gap-2">
            <MapPin className="w-5 h-5 text-teal-700" />
            {isCurrent ? 'Experimental 2026 Sri Lanka District Map' : 'Sri Lanka District Activity Map'}
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            {isCurrent 
              ? '2026 Week 37 NDCU operational surveillance relative activity spatial view' 
              : 'Geographic view of relative dengue activity across all 25 administrative districts'}
          </p>
        </div>

        <span className={`text-xs font-semibold border px-3 py-1 rounded-full self-start sm:self-auto ${
          isCurrent ? 'bg-amber-100 text-amber-900 border-amber-300' : 'bg-teal-50 text-teal-800 border-teal-200'
        }`}>
          25 Districts Loaded
        </span>
      </div>

      {isCurrent ? (
        <CurrentInferenceWarning />
      ) : (
        <PrototypeDisclaimer type="percentiles" />
      )}

      {/* Grid: Map on Left, District Summary on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Map Container (Takes 2 Cols) */}
        <div className="lg:col-span-2 min-h-[500px]">
          <DistrictMap 
            latestDistricts={districts}
            selectedDistrict={selectedDistrictName}
            onSelectDistrict={handleSelectDistrict}
          />
        </div>

        {/* Selected District Quick Summary Panel */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col justify-between">
          {selectedDistrictData ? (
            <div className="space-y-5">
              
              <div className="border-b border-slate-100 pb-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Selected District</span>
                  <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold ${selCfg.badgeBg} ${selCfg.badgeText}`}>
                    {actLevel}
                  </span>
                </div>
                <h2 className="text-2xl font-black text-slate-900 mt-1">{selectedDistrictData.district}</h2>
              </div>

              {/* Core Indicators */}
              <div className="space-y-3">
                <div className="p-3 bg-slate-50 rounded-xl border border-slate-200/70 flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-600">Current Reported Cases</span>
                  <span className="text-base font-extrabold text-slate-900">{formatNumber(selectedDistrictData.current_cases)}</span>
                </div>

                <div className="p-3 bg-teal-50/60 rounded-xl border border-teal-200/70 flex items-center justify-between">
                  <span className="text-xs font-semibold text-teal-800">1-Wk RF Forecast</span>
                  <span className="text-base font-black text-teal-950">{formatNumber(selectedDistrictData.forecast_cases_1w)}</span>
                </div>

                <div className="p-3 bg-slate-50 rounded-xl border border-slate-200/70 flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-600">Persistence Benchmark</span>
                  <span className="text-sm font-semibold text-slate-700">{formatNumber(selectedDistrictData.persistence_forecast_1w)}</span>
                </div>

                <div className="p-3 bg-slate-50 rounded-xl border border-slate-200/70 flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-600">Forecast Trend</span>
                  <span className="text-xs font-bold text-slate-800 uppercase">{selectedDistrictData.forecast_trend}</span>
                </div>
              </div>

              {/* Historical Percentiles */}
              <div className="p-4 bg-slate-50 rounded-xl border border-slate-200/70">
                <div className="text-xs font-bold text-slate-700 uppercase mb-2">District Historical Distribution</div>
                <div className="grid grid-cols-3 gap-2 text-center text-xs">
                  <div>
                    <div className="text-[10px] text-slate-400">P50 (Median)</div>
                    <div className="font-bold text-slate-800">{formatNumber(selectedDistrictData.historical_p50)}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-400">P75</div>
                    <div className="font-bold text-slate-800">{formatNumber(selectedDistrictData.historical_p75)}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-400">P90</div>
                    <div className="font-bold text-slate-800">{formatNumber(selectedDistrictData.historical_p90)}</div>
                  </div>
                </div>
              </div>

              <p className="text-[11px] text-slate-500 italic">
                * Relative activity compares this district with its own historical distribution.
              </p>

            </div>
          ) : (
            <div className="p-8 text-center text-slate-400 text-xs">
              Click any district polygon on the map to inspect detail.
            </div>
          )}

          {selectedDistrictData && (
            <button
              onClick={() => navigate(`/district?name=${selectedDistrictData.district}`)}
              className="w-full mt-6 py-2.5 bg-teal-700 hover:bg-teal-800 text-white text-xs font-semibold rounded-xl flex items-center justify-center gap-2 transition-colors shadow-xs"
            >
              <span>Explore Full {selectedDistrictData.district} Intelligence</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>

      </div>

    </div>
  );
}
