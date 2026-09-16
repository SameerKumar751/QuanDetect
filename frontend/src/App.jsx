import React from "react";
import { Routes, Route } from "react-router-dom";
import Sidebar from "@/components/Sidebar";
import { AppStateProvider } from "@/lib/AppStateContext";

import Landing from "@/pages/Landing";
import Upload from "@/pages/Upload";
import Training from "@/pages/Training";
import Prediction from "@/pages/Prediction";
import Comparison from "@/pages/Comparison";
import Results from "@/pages/Results";
import ImagingDashboard from "@/pages/imaging/ImagingDashboard";

export default function App() {
  return (
    <AppStateProvider>
      <div className="min-h-screen flex bg-background quantum-dots">
        <Sidebar />
        <main className="flex-1 min-w-0 pt-16 lg:pt-0">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-10 py-8 lg:py-10">
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/upload" element={<Upload />} />
              <Route path="/training" element={<Training />} />
              <Route path="/prediction" element={<Prediction />} />
              <Route path="/comparison" element={<Comparison />} />
              <Route path="/results" element={<Results />} />
              <Route path="/imaging" element={<ImagingDashboard />} />
            </Routes>
          </div>
        </main>
      </div>
    </AppStateProvider>
  );
}
