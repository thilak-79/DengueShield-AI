/**
 * DengueShield AI - Frontend API Service Layer
 * Centralized API client for communicating with the FastAPI backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

/**
 * Generic fetch wrapper with error handling
 */
async function fetchJson(endpoint) {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const response = await fetch(url);
    if (!response.ok) {
      const errorText = await response.text().catch(() => '');
      throw new Error(`HTTP Error ${response.status}: ${errorText || response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`API Error [${endpoint}]:`, error);
    throw error;
  }
}

export const apiService = {
  getHealth: () => fetchJson('/api/health'),
  getDistricts: () => fetchJson('/api/districts'),
  getLatestActivity: () => fetchJson('/api/activity/latest'),
  getActivitySummary: () => fetchJson('/api/activity/summary'),
  getDistrictDetail: (district) => fetchJson(`/api/district/${encodeURIComponent(district)}`),
  getDistrictExplanation: (district) => fetchJson(`/api/explanation/district/${encodeURIComponent(district)}`),
  getGlobalExplanation: () => fetchJson('/api/explanation/global'),
  getModelPerformance: () => fetchJson('/api/model/performance'),
  getWalkForward: () => fetchJson('/api/model/walk-forward'),
  getWalkForwardDetails: () => fetchJson('/api/model/walk-forward/details'),
  getModelInfo: () => fetchJson('/api/model/info'),
  
  // Helper for figure image URLs served by FastAPI static mount
  getFigureUrl: (path) => {
    if (!path) return '';
    if (path.startsWith('http://') || path.startsWith('https://')) return path;
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return `${API_BASE_URL}${cleanPath}`;
  }
};

export default apiService;
