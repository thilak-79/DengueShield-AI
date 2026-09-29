import React, { useEffect, useState } from 'react';
import { BrainCircuit, Image as ImageIcon, Building2 } from 'lucide-react';
import apiService from '../services/api';
import { useMode } from '../context/ModeContext';
import GlobalShapChart from '../components/charts/GlobalShapChart';
import ForecastDriversPanel from '../components/district/ForecastDriversPanel';
import PrototypeDisclaimer from '../components/common/PrototypeDisclaimer';
import CurrentInferenceWarning from '../components/common/CurrentInferenceWarning';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';

export default function ExplainabilityPage() {
  const { isCurrent } = useMode();
  const [globalData, setGlobalData] = useState(null);
  const [districtList, setDistrictList] = useState([]);
  const [selectedDistrict, setSelectedDistrict] = useState('Colombo');
  const [districtExp, setDistrictExp] = useState(null);
  const [districtDetail, setDistrictDetail] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      if (isCurrent) {
        const [gRes, distListRes] = await Promise.all([
          apiService.getCurrentGlobalExplanation(),
          apiService.getCurrentDistricts()
        ]);
        setGlobalData({
          features: (gRes?.data || []).map((item, index) => ({
            feature_group: item.feature,
            mean_abs_shap: item.mean_abs_shap,
            mean_signed_shap: item.mean_signed_shap,
            rank: index + 1
          }))
        });
        const dList = distListRes?.districts || (Array.isArray(distListRes) ? distListRes : []);
        setDistrictList(dList);
      } else {
        const [gRes, distListRes] = await Promise.all([
          apiService.getGlobalExplanation(),
          apiService.getDistricts()
        ]);
        setGlobalData(gRes);
        const dList = distListRes?.districts || (Array.isArray(distListRes) ? distListRes : []);
        setDistrictList(dList);
      }
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [isCurrent]);

  // Fetch selected district local SHAP drivers
  useEffect(() => {
    if (selectedDistrict) {
      if (isCurrent) {
        Promise.all([
          apiService.getCurrentDistrict(selectedDistrict),
          apiService.getCurrentDistrictExplanation(selectedDistrict).catch(() => null)
        ]).then(([det, exp]) => {
          setDistrictDetail(det?.district || det);
          setDistrictExp(exp);
        }).catch(err => console.error('Error fetching local current explanation:', err));
      } else {
        Promise.all([
          apiService.getDistrictDetail(selectedDistrict),
          apiService.getDistrictExplanation(selectedDistrict).catch(() => null)
        ]).then(([det, exp]) => {
          setDistrictDetail(det?.district || det);
          setDistrictExp(exp);
        }).catch(err => console.error('Error fetching local historical explanation:', err));
      }
    }
  }, [selectedDistrict, isCurrent]);

  if (loading) return <LoadingSpinner message="Fetching model explainability intelligence..." />;
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
            {isCurrent ? 'Current 2026 Model Explainability (SHAP)' : 'Model Explainability & SHAP Analysis'}
          </h1>
          <p className="text-xs text-slate-500 mt-0.5">
            {isCurrent 
              ? 'Understanding feature contributions for 2026 Week 37 operational inference' 
              : 'Understanding feature contributions and internal decision-making for 2025 one-week district forecasts'}
          </p>
        </div>
        <span className="text-xs font-semibold bg-teal-50 text-teal-800 border border-teal-200 px-3 py-1 rounded-full self-start sm:self-auto">
          SHAP Framework
        </span>
      </div>

      {isCurrent ? (
        <CurrentInferenceWarning />
      ) : (
        <PrototypeDisclaimer type="shap" />
      )}

      {/* Concept Explanation Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        <div className="p-4 bg-white rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="text-xs font-bold text-teal-800 uppercase mb-1">Global Feature Importance</div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Identifies overall feature importance across all district predictions. Shows which features the model relied strongly on.
          </p>
        </div>

        <div className="p-4 bg-white rounded-2xl border border-slate-200/80 shadow-xs">
          <div className="text-xs font-bold text-teal-800 uppercase mb-1">Local District Drivers</div>
          <p className="text-xs text-slate-600 leading-relaxed">
            Explains a specific single forecast for a given district. Shows how individual variables pushed that specific prediction up or down.
          </p>
        </div>

        <div className="p-4 bg-rose-50/70 rounded-2xl border border-rose-200/70 shadow-xs">
          <div className="text-xs font-bold text-rose-800 uppercase mb-1">Positive Feature Impact</div>
          <p className="text-xs text-rose-900 leading-relaxed">
            A positive SHAP value (+X) indicates the feature increased the model prediction above the expected baseline value.
          </p>
        </div>

        <div className="p-4 bg-emerald-50/70 rounded-2xl border border-emerald-200/70 shadow-xs">
          <div className="text-xs font-bold text-emerald-800 uppercase mb-1">Negative Feature Impact</div>
          <p className="text-xs text-emerald-900 leading-relaxed">
            A negative SHAP value (-X) indicates the feature decreased the model prediction below the expected baseline value.
          </p>
        </div>

      </div>

      {/* Interactive Global SHAP Bar Chart */}
      <GlobalShapChart globalExplanationData={globalData} />

      {/* Local District Driver Inspector */}
      <div className="space-y-3">
        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Building2 className="w-5 h-5 text-teal-700" />
            <div>
              <h3 className="text-sm font-bold text-slate-900">Local District Driver Inspector</h3>
              <p className="text-xs text-slate-500">Inspect top drivers for any selected district</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <label htmlFor="exp-district-select" className="text-xs font-semibold text-slate-600 shrink-0">
              District:
            </label>
            <select
              id="exp-district-select"
              value={selectedDistrict}
              onChange={(e) => setSelectedDistrict(e.target.value)}
              className="bg-slate-50 border border-slate-300 font-bold text-slate-800 text-xs rounded-xl px-3 py-1.5 focus:outline-hidden focus:ring-2 focus:ring-teal-500"
            >
              {districtList.map(name => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </div>
        </div>

        {districtDetail && (
          <ForecastDriversPanel districtDetail={districtDetail} explanationData={districtExp} />
        )}
      </div>

      {/* Saved SHAP Visualizations (Served by Backend static mount) */}
      {!isCurrent && (
        <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-xs space-y-6">
          <div>
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <ImageIcon className="w-5 h-5 text-teal-700" />
              Saved Historical Research SHAP Figures
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Static figures generated during historical research validation
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
              <p className="text-[11px] text-slate-500 mt-2">Distribution of feature impacts showing value ranges vs. impact direction.</p>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
