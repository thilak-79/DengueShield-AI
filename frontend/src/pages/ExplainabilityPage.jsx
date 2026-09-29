import React, { useEffect, useState } from 'react';
import { BrainCircuit, Image as ImageIcon } from 'lucide-react';
import apiService from '../services/api';
import GlobalShapChart from '../components/charts/GlobalShapChart';
import PrototypeDisclaimer from '../components/common/PrototypeDisclaimer';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';

export default function ExplainabilityPage() {
  const [globalData, setGlobalData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.getGlobalExplanation();
      setGlobalData(data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading) return <LoadingSpinner message="Fetching global model explainability intelligence..." />;
  if (error) return <ErrorMessage error={error} onRetry={fetchData} />;

  // Served figures URLs
  const shapGlobalBarImg = apiService.getFigureUrl('/figures/explainability/shap_global_bar.png');
  const shapBeeswarmImg = apiService.getFigureUrl('/figures/explainability/shap_summary_beeswarm.png');

  return (
    <div className="space-y-6">
      
      {/* Header Card */}
      <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-900 flex items-center gap-2">
            <BrainCircuit className="w-5 h-5 text-teal-700" />
            Model Explainability & SHAP Analysis
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Understanding feature contributions and internal decision-making for 2025 one-week district forecasts
          </p>
        </div>
        <span className="text-xs font-semibold bg-teal-50 text-teal-800 border border-teal-200 px-3 py-1 rounded-full self-start sm:self-auto">
          SHAP Framework
        </span>
      </div>

      {/* Prominent Scientific Disclaimer Banner */}
      <PrototypeDisclaimer type="shap" />

      {/* Concept Explanation Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        <div className="p-4 bg-white rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="text-xs font-bold text-teal-800 uppercase mb-1">Global Explanation</div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Identifies overall feature importance across all district predictions. Shows which features matter most on average.
          </p>
        </div>

        <div className="p-4 bg-white rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="text-xs font-bold text-teal-800 uppercase mb-1">Local District Explanation</div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Explains a specific single forecast for a given district. Shows how individual variables pushed that specific prediction up or down.
          </p>
        </div>

        <div className="p-4 bg-rose-50/70 rounded-2xl border border-rose-200/70 shadow-xs">
          <div className="text-xs font-bold text-rose-800 uppercase mb-1">Positive SHAP Contribution</div>
          <p className="text-xs text-rose-900 leading-relaxed">
            A positive SHAP value (+X) increases the model's predicted dengue case count above the average baseline prediction.
          </p>
        </div>

        <div className="p-4 bg-emerald-50/70 rounded-2xl border border-emerald-200/70 shadow-xs">
          <div className="text-xs font-bold text-emerald-800 uppercase mb-1">Negative SHAP Contribution</div>
          <p className="text-xs text-emerald-900 leading-relaxed">
            A negative SHAP value (-X) lowers the model's predicted dengue case count below the average baseline prediction.
          </p>
        </div>

      </div>

      {/* Interactive Global SHAP Bar Chart */}
      <GlobalShapChart globalExplanationData={globalData} />

      {/* Saved SHAP Visualizations served by Backend */}
      <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs space-y-6">
        <div>
          <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <ImageIcon className="w-5 h-5 text-teal-700" />
            Saved Research SHAP Figures
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Static figures generated during the research validation pipeline
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-center">
            <h4 className="text-xs font-bold text-slate-800 mb-2">Global Feature Importance (Bar Plot)</h4>
            <img 
              src={shapGlobalBarImg} 
              alt="Global SHAP Bar Plot" 
              className="max-h-80 mx-auto rounded-lg shadow-xs object-contain"
              onError={(e) => {
                e.target.style.display = 'none';
              }}
            />
            <p className="text-[11px] text-slate-500 mt-2">Ranked mean absolute SHAP values across all test predictions.</p>
          </div>

          <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 text-center">
            <h4 className="text-xs font-bold text-slate-800 mb-2">SHAP Beeswarm Summary Plot</h4>
            <img 
              src={shapBeeswarmImg} 
              alt="SHAP Beeswarm Plot" 
              className="max-h-80 mx-auto rounded-lg shadow-xs object-contain"
              onError={(e) => {
                e.target.style.display = 'none';
              }}
            />
            <p className="text-[11px] text-slate-500 mt-2">Distribution of feature impacts showing value ranges (high/low) vs. impact direction.</p>
          </div>
        </div>
      </div>

    </div>
  );
}
