import React, { useEffect, useState } from 'react';
import { Activity, Sparkles, History } from 'lucide-react';
import apiService from '../../services/api';
import { useMode } from '../../context/ModeContext';
import ModeSelector from '../common/ModeSelector';

export default function Header({ latestData, isSidebarOpen, setIsSidebarOpen }) {
  const { isCurrent } = useMode();
  const [isHealthy, setIsHealthy] = useState(true);
  const [currentStatus, setCurrentStatus] = useState(null);

  useEffect(() => {
    apiService.getHealth()
      .then(() => setIsHealthy(true))
      .catch(() => setIsHealthy(false));
  }, []);

  useEffect(() => {
    if (isCurrent) {
      apiService.getCurrentStatus()
        .then(res => setCurrentStatus(res?.data || res))
        .catch(err => console.error('Failed to load current status in Header:', err));
    }
  }, [isCurrent]);

  // Extract dates based on mode
  const sampleDistrict = latestData?.data?.[0];
  
  const originDate = isCurrent 
    ? (currentStatus?.surveillance_window_end || '2026-09-13')
    : (latestData?.forecast_origin_date || sampleDistrict?.forecast_origin_date || '2025-12-12');

  const targetDate = isCurrent
    ? (currentStatus?.forecast_target_start ? `${currentStatus.forecast_target_start} to ${currentStatus.forecast_target_end}` : '14–20 Sep 2026')
    : (sampleDistrict?.target_date || '2025-12-13');

  const modelName = isCurrent ? 'RF Full-History Refit' : 'Random Forest';

  return (
    <header className={`sticky top-0 z-30 backdrop-blur-md border-b px-4 sm:px-6 py-3 transition-colors ${
      isCurrent ? 'bg-amber-50/90 border-amber-200' : 'bg-white/95 border-slate-200'
    }`}>
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-3">
        
        {/* Left: Mobile Toggle & App Title */}
        <div className="flex items-center justify-between md:justify-start gap-3">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="lg:hidden p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
              aria-label="Toggle navigation menu"
            >
              <Activity className="w-5 h-5 text-teal-700" />
            </button>

            <div className="flex items-center gap-2.5">
              <div className={`w-9 h-9 rounded-xl font-black text-lg flex items-center justify-center shadow-sm ${
                isCurrent ? 'bg-amber-500 text-slate-950 shadow-amber-500/20' : 'bg-teal-700 text-white shadow-teal-700/20'
              }`}>
                DS
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight leading-none">
                    DengueShield AI
                  </h1>
                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-extrabold ${
                    isCurrent ? 'bg-amber-200 text-amber-950 border border-amber-300' : 'bg-teal-100 text-teal-800 border border-teal-200'
                  }`}>
                    {isCurrent ? 'EXPERIMENTAL 2026' : 'HISTORICAL PROTOTYPE'}
                  </span>
                </div>
                <p className="text-xs text-slate-500 hidden sm:block">
                  Explainable Dengue Activity Intelligence & Decision Support for Sri Lanka
                </p>
              </div>
            </div>
          </div>

          {/* Mode Selector on Mobile */}
          <div className="md:hidden">
            <ModeSelector />
          </div>
        </div>

        {/* Center/Right: Global Mode Selector & Meta Badges */}
        <div className="flex items-center justify-between md:justify-end gap-3 text-xs">
          
          {/* Desktop Mode Selector */}
          <div className="hidden md:block">
            <ModeSelector />
          </div>

          {/* Metadata Badges */}
          <div className="hidden lg:flex items-center gap-3 bg-white/80 px-3 py-1.5 rounded-xl border border-slate-200/80 shadow-2xs">
            <div>
              <span className="text-slate-400 block text-[10px] uppercase font-semibold">
                {isCurrent ? 'Data Freshness' : 'Forecast Origin'}
              </span>
              <span className="font-semibold text-slate-800">{originDate}</span>
            </div>
            <div className="h-6 w-px bg-slate-200" />
            <div>
              <span className="text-slate-400 block text-[10px] uppercase font-semibold">
                {isCurrent ? 'Forecast Period' : 'Target Date'}
              </span>
              <span className="font-semibold text-slate-800">{targetDate}</span>
            </div>
            <div className="h-6 w-px bg-slate-200" />
            <div>
              <span className="text-slate-400 block text-[10px] uppercase font-semibold">Model</span>
              <span className="font-semibold text-teal-700">{modelName}</span>
            </div>
          </div>

          {/* API Health Pill */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-white border border-slate-200 text-slate-700 shadow-2xs">
            {isHealthy ? (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="hidden xs:inline">API Online</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-red-500" />
                <span className="text-red-700 font-semibold">Offline</span>
              </>
            )}
          </div>
        </div>

      </div>
    </header>
  );
}
