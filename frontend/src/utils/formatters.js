/**
 * Helper formatters and style mappings for DengueShield AI
 */

export const ACTIVITY_CONFIG = {
  'VERY HIGH': {
    label: 'VERY HIGH',
    bg: 'bg-red-50',
    border: 'border-red-200',
    text: 'text-red-800',
    badgeBg: 'bg-red-100',
    badgeText: 'text-red-700',
    color: '#dc2626',
    dot: 'bg-red-500',
    description: '≥ historical P90'
  },
  'HIGH': {
    label: 'HIGH',
    bg: 'bg-orange-50',
    border: 'border-orange-200',
    text: 'text-orange-800',
    badgeBg: 'bg-orange-100',
    badgeText: 'text-orange-700',
    color: '#ea580c',
    dot: 'bg-orange-500',
    description: 'P75 – P90'
  },
  'ELEVATED': {
    label: 'ELEVATED',
    bg: 'bg-amber-50',
    border: 'border-amber-200',
    text: 'text-amber-800',
    badgeBg: 'bg-amber-100',
    badgeText: 'text-amber-700',
    color: '#d97706',
    dot: 'bg-amber-500',
    description: 'P50 – P75'
  },
  'LOW': {
    label: 'LOW',
    bg: 'bg-emerald-50',
    border: 'border-emerald-200',
    text: 'text-emerald-800',
    badgeBg: 'bg-emerald-100',
    badgeText: 'text-emerald-700',
    color: '#059669',
    dot: 'bg-emerald-500',
    description: '< historical P50'
  },
  'NO DATA': {
    label: 'NO DATA',
    bg: 'bg-slate-50',
    border: 'border-slate-200',
    text: 'text-slate-600',
    badgeBg: 'bg-slate-200',
    badgeText: 'text-slate-700',
    color: '#94a3b8',
    dot: 'bg-slate-400',
    description: 'No data'
  }
};

export const FEATURE_NAME_MAP = {
  'cases_current': 'Current dengue cases',
  'cases_lag_1': 'Cases 1 week ago',
  'cases_lag_2': 'Cases 2 weeks ago',
  'cases_lag_3': 'Cases 3 weeks ago',
  'cases_lag_4': 'Cases 4 weeks ago',
  'cases_rolling_2': 'Recent 2-week case average',
  'cases_rolling_4': 'Recent 4-week case average',
  'cases_rolling_8': 'Recent 8-week case average',
  'rain_mm_rolling_4': 'Recent 4-week rainfall',
  'rain_mm_rolling_8': 'Recent 8-week rainfall',
  'rain_mm_lag_1': 'Rainfall 1 week ago',
  'rain_mm_lag_2': 'Rainfall 2 weeks ago',
  'rain_mm_lag_4': 'Rainfall 4 weeks ago',
  'temp_c_rolling_4': 'Recent 4-week temperature',
  'humidity_pct_rolling_4': 'Recent 4-week humidity',
  'district_enc': 'District identity & category',
  'month': 'Month of year',
  'week_of_year': 'Cyclical week-of-year'
};

export function normalizeDistrictName(name) {
  if (!name) return '';
  return name
    .toString()
    .trim()
    .replace(/\s+district$/i, '')
    .replace(/[^a-zA-Z0-9]/g, '')
    .toLowerCase();
}

export function getFriendlyFeatureName(technicalName) {
  if (!technicalName) return 'Unknown Feature';
  if (FEATURE_NAME_MAP[technicalName]) return FEATURE_NAME_MAP[technicalName];
  
  return technicalName
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (l) => l.toUpperCase());
}

export function formatNumber(val, decimals = 0) {
  if (val === null || val === undefined || isNaN(val)) return 'N/A';
  return Number(val).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}

export function formatPercent(val, showPlus = false) {
  if (val === null || val === undefined || isNaN(val)) return '0%';
  const num = Number(val);
  const prefix = showPlus && num > 0 ? '+' : '';
  return `${prefix}${num.toFixed(1)}%`;
}
