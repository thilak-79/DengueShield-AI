import React, { useEffect, useState } from 'react';
import { useMode } from '../../context/ModeContext';
import { AlertTriangle, Info, ShieldAlert, Database, Calendar } from 'lucide-react';
import apiService from '../../services/api';

export default function CurrentInferenceWarning({ statusData: externalStatus, className = '' }) {
  const { isCurrent } = useMode();
  const [status, setStatus] = useState(externalStatus || null);

  useEffect(() => {
    if (isCurrent && !externalStatus) {
      apiService.getCurrentStatus()
        .then(res => setStatus(res?.data || res))
        .catch(err => console.error('Failed to load current status warning metadata:', err));
    }
  }, [isCurrent, externalStatus]);

  if (!isCurrent) return null;

  const caseSource = status?.case_source || 'NDCU';
  const trainingSource = status?.training_source || 'WER-aligned historical surveillance';
  const freshnessDate = status?.data_freshness_date || status?.surveillance_window_end || '13 Sep 2026';
  const forecastPeriod = status?.forecast_target_start && status?.forecast_target_end 
    ? `${status.forecast_target_start} – ${status.forecast_target_end}` 
    : '14–20 Sep 2026';

  return (
    <div className={`p-4 bg-amber-500/10 border-2 border-amber-500/40 rounded-2xl text-amber-950 shadow-xs ${className}`}>
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-amber-500/20 pb-3 mb-3">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-amber-500 text-slate-950 flex items-center justify-center font-black shrink-0 shadow-sm">
            <AlertTriangle className="w-4 h-4 text-slate-950" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-black tracking-wider uppercase text-amber-950">
                EXPERIMENTAL CURRENT INFERENCE (2026)
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-200 text-amber-900 border border-amber-300">
                NON-OFFICIAL
              </span>
            </div>
            <p className="text-[11px] text-amber-900 mt-0.5">
              Operational inference benchmark utilizing different reporting pipelines. Not an official Ministry of Health forecast or alert.
            </p>
          </div>
        </div>
      </div>

      {/* Grid of Dynamic Facts */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
        <div className="bg-white/80 p-2.5 rounded-xl border border-amber-300/60">
          <span className="text-[10px] uppercase font-bold text-amber-800 block">Surveillance Source</span>
          <span className="font-extrabold text-slate-900">{caseSource}</span>
        </div>
        
        <div className="bg-white/80 p-2.5 rounded-xl border border-amber-300/60">
          <span className="text-[10px] uppercase font-bold text-amber-800 block">Model Training Source</span>
          <span className="font-extrabold text-slate-900">{trainingSource}</span>
        </div>

        <div className="bg-white/80 p-2.5 rounded-xl border border-amber-300/60">
          <span className="text-[10px] uppercase font-bold text-amber-800 block">Data Freshness</span>
          <span className="font-extrabold text-slate-900">{freshnessDate}</span>
        </div>

        <div className="bg-white/80 p-2.5 rounded-xl border border-amber-300/60">
          <span className="text-[10px] uppercase font-bold text-amber-800 block">Forecast Period</span>
          <span className="font-extrabold text-amber-950">{forecastPeriod}</span>
        </div>
      </div>
    </div>
  );
}
