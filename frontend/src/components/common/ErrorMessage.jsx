import React from 'react';
import { AlertCircle, RefreshCw, ServerCrash } from 'lucide-react';

export default function ErrorMessage({ error, onRetry }) {
  const isNetworkError = error?.message?.includes('Failed to fetch') || error?.message?.includes('NetworkError');

  return (
    <div className="p-8 bg-red-50/70 border border-red-200 rounded-2xl text-center max-w-lg mx-auto my-8 shadow-sm">
      <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4 text-red-600">
        {isNetworkError ? <ServerCrash className="w-6 h-6" /> : <AlertCircle className="w-6 h-6" />}
      </div>
      
      <h3 className="text-base font-bold text-red-900 mb-2">
        {isNetworkError ? 'DengueShield API Unavailable' : 'Unable to Load Data'}
      </h3>
      
      <p className="text-xs sm:text-sm text-red-700 mb-4 leading-relaxed">
        {isNetworkError ? (
          <>
            DengueShield API is currently unavailable.<br />
            Start the FastAPI backend at <code className="bg-red-100 px-1.5 py-0.5 rounded font-mono text-red-800">http://127.0.0.1:8000</code>.
          </>
        ) : (
          error?.message || 'An unexpected error occurred while fetching forecast intelligence.'
        )}
      </p>

      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-xs font-semibold rounded-lg transition-colors shadow-sm"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Retry Connection
        </button>
      )}
    </div>
  );
}
