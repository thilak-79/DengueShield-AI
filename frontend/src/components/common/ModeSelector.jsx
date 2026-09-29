import React from 'react';
import { useMode } from '../../context/ModeContext';
import { History, Sparkles, AlertTriangle } from 'lucide-react';

export default function ModeSelector({ className = '' }) {
  const { mode, setMode } = useMode();

  return (
    <div className={`inline-flex items-center p-1 bg-slate-200/80 backdrop-blur-md rounded-xl border border-slate-300/80 shadow-xs ${className}`}>
      
      {/* Historical Mode Button */}
      <button
        type="button"
        onClick={() => setMode('historical')}
        className={`
          flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all duration-200 cursor-pointer
          ${mode === 'historical'
            ? 'bg-white text-slate-900 shadow-sm border border-slate-200/80'
            : 'text-slate-600 hover:text-slate-900 hover:bg-slate-300/50'
          }
        `}
      >
        <History className={`w-3.5 h-3.5 ${mode === 'historical' ? 'text-teal-700' : 'text-slate-400'}`} />
        <span>Historical Evaluation</span>
      </button>

      {/* Experimental Current 2026 Button */}
      <button
        type="button"
        onClick={() => setMode('current')}
        className={`
          flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all duration-200 cursor-pointer
          ${mode === 'current'
            ? 'bg-amber-500 text-slate-950 shadow-sm shadow-amber-500/20 border border-amber-400'
            : 'text-slate-700 hover:text-amber-900 hover:bg-amber-100/50'
          }
        `}
      >
        <Sparkles className={`w-3.5 h-3.5 ${mode === 'current' ? 'text-slate-950 fill-amber-300' : 'text-amber-600'}`} />
        <span className="flex items-center gap-1">
          Experimental Current 2026
          <span className={`text-[9px] px-1 py-0.2 rounded font-black tracking-wider uppercase ${
            mode === 'current' ? 'bg-amber-950 text-amber-300' : 'bg-amber-200 text-amber-900'
          }`}>
            EXP
          </span>
        </span>
      </button>

    </div>
  );
}
