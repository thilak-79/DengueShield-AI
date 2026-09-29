import React from 'react';
import { Loader2 } from 'lucide-react';

export default function LoadingSpinner({ message = 'Loading forecast intelligence...' }) {
  return (
    <div className="flex flex-col items-center justify-center p-12 min-h-[300px] text-center">
      <Loader2 className="w-10 h-10 text-teal-600 animate-spin mb-3" />
      <p className="text-sm font-medium text-slate-600">{message}</p>
      <p className="text-xs text-slate-400 mt-1">Connecting to FastAPI backend (127.0.0.1:8000)...</p>
    </div>
  );
}
