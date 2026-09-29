import React, { createContext, useContext, useState, useEffect } from 'react';

const ModeContext = createContext();

export function ModeProvider({ children }) {
  // Mode can be 'historical' or 'current'
  const [mode, setModeState] = useState(() => {
    return localStorage.getItem('dengueshield_mode') || 'historical';
  });

  const setMode = (newMode) => {
    if (newMode === 'historical' || newMode === 'current') {
      setModeState(newMode);
      localStorage.setItem('dengueshield_mode', newMode);
    }
  };

  const isCurrent = mode === 'current';
  const isHistorical = mode === 'historical';

  return (
    <ModeContext.Provider value={{ mode, setMode, isCurrent, isHistorical }}>
      {children}
    </ModeContext.Provider>
  );
}

export function useMode() {
  const context = useContext(ModeContext);
  if (!context) {
    throw new Error('useMode must be used within a ModeProvider');
  }
  return context;
}

export default ModeContext;
