import React, { useEffect, useState } from 'react';
import { Activity } from 'lucide-react';
import apiService from '../../services/api';

export default function Header({ latestData, isSidebarOpen, setIsSidebarOpen }) {
  const [isHealthy, setIsHealthy] = useState(true);

  useEffect(() => {
    apiService.getHealth()
      .then(() => setIsHealthy(true))
      .catch(() => setIsHealthy(false));
  }, []);

  // Extract origin date and target date from payload
  const sampleDistrict = latestData?.data?.[0];
  const originDate = latestData?.forecast_origin_date || sampleDistrict?.forecast_origin_date || '2025-12-12';
  const targetDate = sampleDistrict?.target_date || '2025-12-13';
  const horizon = latestData?.forecast_horizon_weeks ? `${latestData.forecast_horizon_weeks} Week` : '1 Week';

  return (
    <header className="sticky top-0 z-30 bg-white/95 backdrop-blur-md border-b border-slate-200 px-4 sm:px-6 py-3 transition-all">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-4">
        
        {/* Left: Mobile Toggle & App Title */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="lg:hidden p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
            aria-label="Toggle navigation menu"
          >
            <Activity className="w-5 h-5 text-teal-700" />
          </button>

          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-teal-700 text-white flex items-center justify-center font-black text-lg shadow-sm shadow-teal-700/20">
              DS
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight leading-none">
                  DengueShield AI
                </h1>
                <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold bg-teal-100 text-teal-800 border border-teal-200">
                  PROTOTYPE
                </span>
              </div>
              <p className="text-xs text-slate-500 hidden sm:block">
                Explainable Dengue Activity Intelligence & Decision Support for Sri Lanka
              </p>
            </div>
          </div>
        </div>

        {/* Right: Status & Meta Details */}
        <div className="flex items-center gap-3 sm:gap-4 text-xs">
          
          <div className="hidden md:flex items-center gap-3 bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200/80">
            <div>
              <span className="text-slate-400 block text-[10px] uppercase font-semibold">Forecast Origin</span>
              <span className="font-semibold text-slate-800">{originDate}</span>
            </div>
            <div className="h-6 w-px bg-slate-200" />
            <div>
              <span className="text-slate-400 block text-[10px] uppercase font-semibold">Target Date</span>
              <span className="font-semibold text-slate-800">{targetDate}</span>
            </div>
            <div className="h-6 w-px bg-slate-200" />
            <div>
              <span className="text-slate-400 block text-[10px] uppercase font-semibold">Horizon</span>
              <span className="font-semibold text-slate-800">{horizon}</span>
            </div>
            <div className="h-6 w-px bg-slate-200" />
            <div>
              <span className="text-slate-400 block text-[10px] uppercase font-semibold">Model</span>
              <span className="font-semibold text-teal-700">Random Forest</span>
            </div>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium bg-slate-100 border border-slate-200 text-slate-700">
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
