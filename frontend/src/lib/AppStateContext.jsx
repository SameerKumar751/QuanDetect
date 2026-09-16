import React, { createContext, useContext, useState } from "react";

/**
 * Shares the current dataset session across pages: session_id, dataset
 * metadata, preprocessing config, and trained-model results. This is
 * intentionally simple (no redux) since the app has a single active
 * session/workflow at a time, mirroring the backend's session store.
 */
const AppStateContext = createContext(null);

export function AppStateProvider({ children }) {
  const [sessionId, setSessionId] = useState(null);
  const [dataset, setDataset] = useState(null); // upload response
  const [preprocessInfo, setPreprocessInfo] = useState(null); // preprocess response
  const [classicalResults, setClassicalResults] = useState(null);
  const [quantumResult, setQuantumResult] = useState(null);
  const [selectedFeatures, setSelectedFeatures] = useState([]);

  const resetSession = () => {
    setSessionId(null);
    setDataset(null);
    setPreprocessInfo(null);
    setClassicalResults(null);
    setQuantumResult(null);
    setSelectedFeatures([]);
  };

  const value = {
    sessionId,
    setSessionId,
    dataset,
    setDataset,
    preprocessInfo,
    setPreprocessInfo,
    classicalResults,
    setClassicalResults,
    quantumResult,
    setQuantumResult,
    selectedFeatures,
    setSelectedFeatures,
    resetSession,
  };

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error("useAppState must be used within AppStateProvider");
  return ctx;
}
