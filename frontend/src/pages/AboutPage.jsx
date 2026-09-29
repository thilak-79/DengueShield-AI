import React, { useEffect, useState } from 'react';
import { BookOpen, ShieldAlert, Cpu, Database, Award, Info } from 'lucide-react';
import apiService from '../services/api';
import PrototypeDisclaimer from '../components/common/PrototypeDisclaimer';

export default function AboutPage() {
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    apiService.getModelInfo()
      .then(res => setModelInfo(res))
      .catch(err => console.error('Failed to fetch model info:', err));
  }, []);

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      
      {/* Header Card */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs">
        <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2.5">
          <BookOpen className="w-6 h-6 text-teal-700" />
          About DengueShield AI & Methodology
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Explainable Dengue Activity Intelligence & Decision-Support Prototype for Sri Lanka
        </p>
      </div>

      <PrototypeDisclaimer type="general" />

      {/* Project Purpose */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs space-y-3">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <Award className="w-5 h-5 text-teal-700" />
          1. Project Purpose
        </h2>
        <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">
          DengueShield AI is a machine-learning research and decision-support prototype designed to evaluate district-level dengue activity forecasting across Sri Lanka's 25 administrative districts. The platform provides early-warning activity indicators paired with local SHAP feature explanations to assist public health researchers, epidemiologists, and decision-makers in evaluating predictive model behaviors.
        </p>
      </div>

      {/* Grid of Core Technical Components */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Forecast Horizons & Data */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs space-y-4">
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <Database className="w-5 h-5 text-teal-700" />
            2. Forecast Horizons & Data Streams
          </h2>

          <div className="space-y-3 text-xs text-slate-700">
            <div>
              <strong className="text-slate-900 block text-xs mb-0.5">Forecast Horizons:</strong>
              <div className="flex gap-2">
                <span className="px-2.5 py-1 bg-teal-100 text-teal-800 rounded font-bold">1 Week (Primary)</span>
                <span className="px-2.5 py-1 bg-slate-100 text-slate-700 rounded font-semibold">2 Weeks</span>
                <span className="px-2.5 py-1 bg-slate-100 text-slate-700 rounded font-semibold">4 Weeks</span>
              </div>
            </div>

            <div>
              <strong className="text-slate-900 block text-xs mb-1">Input Data Streams:</strong>
              <ul className="list-disc pl-4 space-y-1.5 text-slate-600">
                <li><strong>Historical Dengue Surveillance:</strong> Weekly reported district case counts.</li>
                <li><strong>Meteorological Features:</strong> Total rainfall (mm), mean temperature (°C), relative humidity (%).</li>
                <li><strong>District identity:</strong> one-hot encoded district category.</li>
                <li><strong>Seasonality:</strong> cyclical week-of-year encoding.</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Models & Validation */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs space-y-4">
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-teal-700" />
            3. Machine Learning & Validation
          </h2>

          <div className="space-y-3 text-xs text-slate-700">
            <div>
              <strong className="text-slate-900 block text-xs mb-1">Evaluated Models:</strong>
              <div className="flex flex-wrap gap-2">
                <span className="px-2.5 py-1 bg-slate-200 text-slate-800 rounded font-bold">Persistence Baseline</span>
                <span className="px-2.5 py-1 bg-teal-100 text-teal-900 rounded font-bold">Random Forest</span>
                <span className="px-2.5 py-1 bg-orange-100 text-orange-900 rounded font-semibold">XGBoost</span>
                <span className="px-2.5 py-1 bg-purple-100 text-purple-900 rounded font-semibold">LightGBM</span>
              </div>
            </div>

            <div>
              <strong className="text-slate-900 block text-xs mb-1">Validation Methodology:</strong>
              <ul className="list-disc pl-4 space-y-1.5 text-slate-600">
                <li><strong>Chronological Splits:</strong> Strict time-series splitting with no random train/test shuffling.</li>
                <li><strong>Nested Walk-Forward Evaluation:</strong> Sequential multi-year evaluation folds (2020 through 2025).</li>
                <li><strong>Leakage-safe temporal design:</strong> Temporal validation checks found no invalid train/validation/test ordering.</li>
              </ul>
            </div>
          </div>
        </div>

      </div>

      {/* SHAP Explainability */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs space-y-3">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <Info className="w-5 h-5 text-teal-700" />
          4. Explainability Framework (SHAP)
        </h2>
        <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">
          To overcome the "black box" nature of ensemble tree models, DengueShield AI incorporates <strong>SHAP (SHapley Additive exPlanations)</strong>. SHAP assigns each feature an additive importance value for a specific prediction, illustrating mathematically how input variables (e.g. current cases, recent 4-week rainfall) influenced the model's output above or below the expected baseline value.
        </p>
      </div>

      {/* Critical System Limitations */}
      <div className="p-6 bg-amber-50/80 border border-amber-200 rounded-2xl space-y-4">
        <h2 className="text-base font-bold text-amber-950 flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-amber-700" />
          5. Important System Limitations & Disclaimers
        </h2>

        <ul className="space-y-2 text-xs text-amber-900 list-disc pl-4 leading-relaxed">
          <li>
            <strong>Decision-Support Prototype:</strong> DengueShield AI is a research decision-support prototype, not a medical or diagnostic system.
          </li>
          <li>
            <strong>Not an Official Alert System:</strong> Activity categories (LOW, ELEVATED, HIGH, VERY HIGH) are statistical historical percentiles (P50, P75, P90) computed per district. They do not constitute official Ministry of Health outbreak declarations.
          </li>
          <li>
            <strong>SHAP does not prove Causality:</strong> SHAP values explain feature contributions within the ML model's internal mathematical logic. They do not prove clinical or environmental causation.
          </li>
          <li>
            <strong>Year-to-Year Performance Variation:</strong> Empirical walk-forward evaluation demonstrates that machine-learning performance varies across years and does not consistently outperform simple persistence baselines at 2-week or 4-week horizons.
          </li>
          <li>
            <strong>Historical Data Evaluation:</strong> Current prototype data represents historical evaluation benchmarks.
          </li>
        </ul>
      </div>

    </div>
  );
}
