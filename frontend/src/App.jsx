import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ModeProvider, useMode } from './context/ModeContext';
import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';
import Overview from './pages/Overview';
import MapPage from './pages/MapPage';
import DistrictPage from './pages/DistrictPage';
import ExplainabilityPage from './pages/ExplainabilityPage';
import PerformancePage from './pages/PerformancePage';
import AboutPage from './pages/AboutPage';
import apiService from './services/api';

function MainAppLayout() {
  const { isCurrent } = useMode();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [latestData, setLatestData] = useState(null);

  useEffect(() => {
    const fetchData = isCurrent ? apiService.getCurrentActivity() : apiService.getLatestActivity();
    fetchData
      .then(data => setLatestData(data))
      .catch(err => console.error('App level activity fetch error:', err));
  }, [isCurrent]);

  return (
    <div className={`min-h-screen flex flex-col font-sans text-slate-900 antialiased selection:bg-teal-500 selection:text-white transition-colors duration-200 ${
      isCurrent ? 'bg-amber-50/30' : 'bg-slate-50'
    }`}>
      
      {/* Top Sticky Header */}
      <Header 
        latestData={latestData}
        isSidebarOpen={isSidebarOpen}
        setIsSidebarOpen={setIsSidebarOpen}
      />

      <div className="flex-1 flex max-w-7xl w-full mx-auto">
        
        {/* Left Navigation Sidebar */}
        <Sidebar 
          isSidebarOpen={isSidebarOpen}
          setIsSidebarOpen={setIsSidebarOpen}
        />

        {/* Main Workspace Area */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto max-w-full">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/map" element={<MapPage />} />
            <Route path="/district" element={<DistrictPage />} />
            <Route path="/explainability" element={<ExplainabilityPage />} />
            <Route path="/performance" element={<PerformancePage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>

      </div>

      {/* Global Footer */}
      <footer className={`border-t py-4 px-6 text-center text-xs text-slate-500 transition-colors ${
        isCurrent ? 'bg-amber-100/50 border-amber-200' : 'bg-white border-slate-200'
      }`}>
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            <strong>DengueShield AI</strong> • {isCurrent ? 'Experimental Current 2026 Inference' : 'Historical Decision-Support Prototype'}
          </div>
          <div className="text-[11px] text-slate-500">
            {isCurrent 
              ? 'NDCU Operational Surveillance & RF Refit Inference' 
              : 'WER Historical Surveillance & SHAP Interpretability Pipeline'}
          </div>
        </div>
      </footer>

    </div>
  );
}

export default function App() {
  return (
    <Router>
      <ModeProvider>
        <MainAppLayout />
      </ModeProvider>
    </Router>
  );
}
