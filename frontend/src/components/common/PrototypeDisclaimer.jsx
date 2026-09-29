import React from 'react';
import { AlertTriangle, Info, ShieldAlert } from 'lucide-react';

export default function PrototypeDisclaimer({ type = 'general', className = '' }) {
  if (type === 'shap') {
    return (
      <div className={`p-4 bg-amber-50/80 border border-amber-200 rounded-xl text-amber-900 text-xs sm:text-sm flex items-start gap-3 ${className}`}>
        <Info className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-amber-950">Scientific Context: </span>
          SHAP explains model behavior and feature contributions. It does <strong>NOT</strong> prove causal relationship (e.g. recent rainfall contributed positively to this model prediction, but does not prove direct causality).
        </div>
      </div>
    );
  }

  if (type === 'percentiles') {
    return (
      <div className={`p-4 bg-slate-50 border border-slate-200 rounded-xl text-slate-700 text-xs sm:text-sm flex items-start gap-3 ${className}`}>
        <Info className="w-5 h-5 text-teal-600 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-slate-900">Statistical Threshold Notice: </span>
          Relative activity categories (Low, Elevated, High, Very High) are based on each district's historical case distribution percentiles (P50, P75, P90). They are <strong>NOT official Ministry of Health alert thresholds</strong>.
        </div>
      </div>
    );
  }

  return (
    <div className={`p-4 bg-teal-900/5 border border-teal-200/60 rounded-xl text-teal-900 text-xs sm:text-sm flex items-start gap-3 ${className}`}>
      <ShieldAlert className="w-5 h-5 text-teal-700 shrink-0 mt-0.5" />
      <div>
        <span className="font-semibold text-teal-950">Decision-Support Prototype: </span>
        DengueShield AI is a historical evaluation prototype for public-health research and decision support. Data shown represents historical evaluation cycles (e.g. 2025 benchmark) and is not official live operational surveillance.
      </div>
    </div>
  );
}
