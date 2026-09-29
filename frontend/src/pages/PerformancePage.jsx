import React, { useEffect, useState } from 'react';
import { BarChart3, Award } from 'lucide-react';
import apiService from '../services/api';
import WalkForwardChart from '../components/charts/WalkForwardChart';
import PrototypeDisclaimer from '../components/common/PrototypeDisclaimer';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import { formatNumber } from '../utils/formatters';

export default function PerformancePage() {
  const [perfData, setPerfData] = useState(null);
  const [walkForwardData, setWalkForwardData] = useState(null);
  const [walkForwardDetails, setWalkForwardDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [perfRes, wfRes, wfDetRes] = await Promise.all([
        apiService.getModelPerformance(),
        apiService.getWalkForward(),
        apiService.getWalkForwardDetails()
      ]);
      setPerfData(perfRes);
      setWalkForwardData(wfRes);
      setWalkForwardDetails(wfDetRes);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading) return <LoadingSpinner message="Fetching model performance and walk-forward validation results..." />;
  if (error) return <ErrorMessage error={error} onRetry={fetchData} />;

  const wfSummary = walkForwardData?.summary || [];

  return (
    <div className="space-y-6">
      
      {/* Header Card */}
      <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-900 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-teal-700" />
            Model Performance & Walk-Forward Validation
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Empirical out-of-sample evaluation comparing Machine Learning models against Persistence baseline
          </p>
        </div>
        <span className="text-xs font-semibold bg-teal-50 text-teal-800 border border-teal-200 px-3 py-1 rounded-full self-start sm:self-auto">
          6-Year Walk-Forward (2020-2025)
        </span>
      </div>

      <PrototypeDisclaimer type="general" />

      {/* Honest Scientific Conclusion Card */}
      <div className="p-5 bg-teal-900 text-white rounded-2xl shadow-md border border-teal-800 space-y-2">
        <div className="flex items-center gap-2 text-teal-300 text-xs font-bold uppercase tracking-wider">
          <Award className="w-4 h-4 text-teal-400" />
          Empirical Research Findings
        </div>
        <p className="text-xs sm:text-sm text-teal-100 leading-relaxed">
          Persistence achieved lower 1-week MAE in 2020, 2021 and 2023, while Random Forest achieved lower MAE in 2022, 2024 and 2025. Random Forest beats persistence in 3 of 6 evaluation years. At 2-week and 4-week horizons, persistence remains highly competitive and may outperform machine-learning models overall.
        </p>
      </div>

      {/* Evaluation Metrics Explanation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        
        <div className="p-4 bg-white rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-bold text-slate-900 uppercase">MAE (Mean Absolute Error)</span>
            <span className="text-[10px] bg-slate-100 px-2 py-0.5 rounded font-mono text-slate-600">Primary Metric</span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed mt-2">
            Measures average prediction error magnitude in reported case units. Lower is better.
          </p>
        </div>

        <div className="p-4 bg-white rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-bold text-slate-900 uppercase">RMSE (Root Mean Square Error)</span>
            <span className="text-[10px] bg-slate-100 px-2 py-0.5 rounded font-mono text-slate-600">Secondary Metric</span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed mt-2">
            Penalizes larger prediction errors more heavily. Useful for identifying large outbreak deviations.
          </p>
        </div>

        <div className="p-4 bg-white rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-bold text-slate-900 uppercase">sMAPE (Symmetric MAPE)</span>
            <span className="text-[10px] bg-slate-100 px-2 py-0.5 rounded font-mono text-slate-600">Relative % Metric</span>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed mt-2">
            Measures symmetric percentage error bounded between 0% and 200%.
          </p>
        </div>

      </div>

      {/* Walk-Forward Interactive Line Chart */}
      <WalkForwardChart walkForwardDetails={walkForwardDetails} />

      {/* Multi-Year Walk-Forward Summary Table */}
      <div className="bg-white p-5 sm:p-6 rounded-2xl border border-slate-200/80 shadow-xs space-y-4">
        <div>
          <h3 className="text-base font-bold text-slate-900">Multi-Year Walk-Forward Model Comparison Table</h3>
          <p className="text-xs text-slate-500">Summary statistics aggregated over 6 test years (2020–2025)</p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-700 font-bold uppercase tracking-wider text-[10px]">
              <tr>
                <th className="p-3">Horizon</th>
                <th className="p-3">Model</th>
                <th className="p-3">Mean MAE</th>
                <th className="p-3">Median MAE</th>
                <th className="p-3">Mean RMSE</th>
                <th className="p-3">Mean sMAPE</th>
                <th className="p-3">Years Beating Persistence</th>
                <th className="p-3">Mean MAE Imprv %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-800">
              {wfSummary.map((row, i) => {
                const isPrimary = row.horizon_weeks === 1 && row.model === 'random_forest';
                return (
                  <tr key={i} className={isPrimary ? 'bg-teal-50/50 font-medium' : 'hover:bg-slate-50'}>
                    <td className="p-3 font-semibold">{row.horizon_weeks} Week{row.horizon_weeks > 1 ? 's' : ''}</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded font-bold uppercase text-[10px] ${
                        row.model === 'random_forest' ? 'bg-teal-100 text-teal-800' :
                        row.model === 'persistence' ? 'bg-slate-200 text-slate-700' :
                        'bg-slate-100 text-slate-700'
                      }`}>
                        {row.model.replace('_', ' ')}
                      </span>
                    </td>
                    <td className="p-3 font-mono font-bold text-slate-900">{formatNumber(row.mean_mae, 2)}</td>
                    <td className="p-3 font-mono">{formatNumber(row.median_mae, 2)}</td>
                    <td className="p-3 font-mono">{formatNumber(row.mean_rmse, 2)}</td>
                    <td className="p-3 font-mono">{formatNumber(row.mean_smape, 1)}%</td>
                    <td className="p-3 font-semibold text-slate-700">{row.years_mae_better_than_persistence} / {row.years_evaluated} Years</td>
                    <td className={`p-3 font-bold ${row.mean_mae_improvement_percent > 0 ? 'text-emerald-600' : 'text-slate-500'}`}>
                      {row.mean_mae_improvement_percent > 0 ? '+' : ''}{formatNumber(row.mean_mae_improvement_percent, 2)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
