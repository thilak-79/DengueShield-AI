import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Map, 
  Search, 
  BrainCircuit, 
  BarChart3, 
  BookOpen, 
  ShieldAlert,
  ChevronRight
} from 'lucide-react';

const NAV_ITEMS = [
  { path: '/', label: 'Overview', icon: LayoutDashboard },
  { path: '/map', label: 'Sri Lanka Map', icon: Map },
  { path: '/district', label: 'District Explorer', icon: Search },
  { path: '/explainability', label: 'Explainability', icon: BrainCircuit },
  { path: '/performance', label: 'Model Performance', icon: BarChart3 },
  { path: '/about', label: 'About & Methodology', icon: BookOpen },
];

export default function Sidebar({ isSidebarOpen, setIsSidebarOpen }) {
  return (
    <>
      {/* Mobile Backdrop */}
      {isSidebarOpen && (
        <div 
          onClick={() => setIsSidebarOpen(false)}
          className="fixed inset-0 bg-slate-900/30 z-40 lg:hidden backdrop-blur-xs transition-opacity"
        />
      )}

      {/* Sidebar Navigation */}
      <aside className={`
        fixed top-0 bottom-0 left-0 z-50 w-64 bg-slate-900 text-slate-300 flex flex-col border-r border-slate-800 transition-transform duration-200 ease-in-out
        lg:static lg:translate-x-0
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        {/* Sidebar Brand Header */}
        <div className="p-5 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-teal-500 text-slate-950 font-black flex items-center justify-center text-sm shadow-md">
              DS
            </div>
            <div>
              <div className="font-bold text-white text-sm tracking-wide">DengueShield AI</div>
              <div className="text-[10px] text-teal-400 font-mono">v1.0.0 • PROTOTYPE</div>
            </div>
          </div>
        </div>

        {/* Mode-neutral Research Prototype Banner */}
        <div className="mx-4 mt-4 p-3 bg-slate-800/80 rounded-xl border border-slate-700/60 text-xs">
          <div className="flex items-center gap-1.5 text-amber-400 font-semibold mb-1">
            <ShieldAlert className="w-3.5 h-3.5 shrink-0" />
            <span>Research Prototype</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-snug">
            Historical evaluation and experimental current inference for decision support.<br />
            Not an official public-health alert system.
          </p>
        </div>

        {/* Nav Links */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={() => setIsSidebarOpen(false)}
                className={({ isActive }) => `
                  flex items-center justify-between px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all duration-150
                  ${isActive 
                    ? 'bg-teal-600 text-white font-semibold shadow-sm shadow-teal-900/40' 
                    : 'text-slate-400 hover:bg-slate-800/70 hover:text-slate-200'
                  }
                `}
              >
                <div className="flex items-center gap-3">
                  <Icon className="w-4 h-4 shrink-0" />
                  <span>{item.label}</span>
                </div>
                <ChevronRight className="w-3.5 h-3.5 opacity-40" />
              </NavLink>
            );
          })}
        </nav>

        {/* Footer Info */}
        <div className="p-4 border-t border-slate-800 text-[11px] text-slate-500 text-center">
          <div>Sri Lanka Dengue Intelligence</div>
          <div className="text-[10px] text-slate-600 mt-0.5">Historical ML & SHAP Evaluation</div>
        </div>
      </aside>
    </>
  );
}
