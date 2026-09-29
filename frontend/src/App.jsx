import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';
import Overview from './pages/Overview';
import MapPage from './pages/MapPage';
import DistrictPage from './pages/DistrictPage';
import ExplainabilityPage from './pages/ExplainabilityPage';
import PerformancePage from './pages/PerformancePage';
import AboutPage from './pages/AboutPage';
import apiService from './services/api';

export default function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [latestData, setLatestData] = useState(null);

  useEffect(() => {
    apiService.getLatestActivity()
      .then(data => setLatestData(data))
      .catch(err => console.error('App level latest activity fetch error:', err));
  }, []);

  return (
    <Router>
      <div className="min-h-screen bg-slate-50 flex flex-col font-sans text-slate-900 antialiased selection:bg-teal-500 selection:text-white">
        
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
        <footer className="bg-white border-t border-slate-200 py-4 px-6 text-center text-xs text-slate-500">
          <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
            <div>
              <strong>DengueShield AI</strong> • Historical Decision-Support Prototype for Sri Lanka
            </div>
            <div className="text-[11px] text-slate-400">
              Random Forest ML & SHAP Explainability Evaluation Pipeline
            </div>
          </div>
        </footer>

      </div>
    </Router>
  );
}
