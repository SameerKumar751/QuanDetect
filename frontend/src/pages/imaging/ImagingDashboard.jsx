import React, { useState, useEffect } from "react";
import {
  ScanLine,
  UploadCloud,
  Cpu,
  Layers,
  Activity,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  BarChart3,
  Atom,
  RefreshCw,
  Sparkles,
  Eye,
  Sliders,
  FileText,
  Info,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { imagingApi } from "@/lib/imagingApi";
import { cn } from "@/lib/utils";

export default function ImagingDashboard() {
  const [samples, setSamples] = useState([]);
  const [selectedSampleId, setSelectedSampleId] = useState("sample_malignant_01");
  const [loadingSample, setLoadingSample] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);

  // Active Session State
  const [sessionData, setSessionData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [gradcamData, setGradcamData] = useState(null);
  const [benchmarks, setBenchmarks] = useState(null);
  const [activeImageView, setActiveImageView] = useState("roi"); // original | roi | gradcam
  const [selectedModelFilter, setSelectedModelFilter] = useState("all"); // all | cnn | quantum | hybrid
  const [errorMsg, setErrorMsg] = useState(null);

  // Load sample catalog & benchmarks on mount
  useEffect(() => {
    loadCatalogAndBenchmarks();
  }, []);

  const loadCatalogAndBenchmarks = async () => {
    try {
      const [sampleList, benchmarkData] = await Promise.all([
        imagingApi.getSampleScans(),
        imagingApi.getBenchmarks(),
      ]);
      setSamples(sampleList);
      setBenchmarks(benchmarkData);

      // Auto-load default malignant sample
      handleSelectSample("sample_malignant_01");
    } catch (err) {
      console.error("Failed to load initial imaging data:", err);
      setErrorMsg("Could not connect to medical imaging backend.");
    }
  };

  const handleSelectSample = async (sampleId) => {
    setSelectedSampleId(sampleId);
    setLoadingSample(true);
    setErrorMsg(null);
    try {
      const uploadRes = await imagingApi.selectSample(sampleId);
      setSessionData(uploadRes);

      // Trigger automatic multi-model inference
      setAnalyzing(true);
      const predRes = await imagingApi.predict(uploadRes.session_id, "all");
      setPredictions(predRes.predictions);

      // Trigger Grad-CAM
      const gradcamRes = await imagingApi.getGradCam(uploadRes.session_id);
      setGradcamData(gradcamRes);
    } catch (err) {
      console.error("Error processing sample scan:", err);
      setErrorMsg(err.response?.data?.detail || "Failed to process CT scan.");
    } finally {
      setLoadingSample(false);
      setAnalyzing(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedSampleId(null);
    setLoadingSample(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const uploadRes = await imagingApi.uploadScan(formData);
      setSessionData(uploadRes);

      setAnalyzing(true);
      const predRes = await imagingApi.predict(uploadRes.session_id, "all");
      setPredictions(predRes.predictions);

      const gradcamRes = await imagingApi.getGradCam(uploadRes.session_id);
      setGradcamData(gradcamRes);
    } catch (err) {
      console.error("Error uploading CT scan:", err);
      setErrorMsg(err.response?.data?.detail || "Failed to parse CT file.");
    } finally {
      setLoadingSample(false);
      setAnalyzing(false);
    }
  };

  const filteredPredictions = predictions?.filter((p) => {
    if (selectedModelFilter === "all") return true;
    return p.model_type === selectedModelFilter;
  });

  const getRiskColor = (tier) => {
    switch (tier) {
      case "High":
        return "text-red-600 bg-red-50 border-red-200";
      case "Moderate":
        return "text-amber-600 bg-amber-50 border-amber-200";
      case "Low":
      default:
        return "text-emerald-600 bg-emerald-50 border-emerald-200";
    }
  };

  const getRiskBarColor = (score) => {
    if (score >= 0.65) return "bg-red-500";
    if (score >= 0.3) return "bg-amber-500";
    return "bg-emerald-500";
  };

  return (
    <div className="space-y-8 pb-16">
      {/* Header Banner */}
      <div className="bg-white rounded-2xl p-6 sm:p-8 border border-border shadow-sm flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-accent-100 text-accent-800">
              <ScanLine className="h-3.5 w-3.5" /> Medical Imaging Pipeline
            </span>
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-primary-100 text-primary-800">
              <Atom className="h-3.5 w-3.5" /> Hybrid VQC + ResNet-18
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-warmgray-900 tracking-tight">
            Lung CT Cancer Diagnostic Platform
          </h1>
          <p className="text-sm text-warmgray-500 mt-1 max-w-2xl">
            Independent clinical imaging pipeline utilizing Hounsfield windowing, automated lung field
            segmentation, classical 2D ResNet-18 CNNs, PennyLane Variational Quantum Classifiers (VQC),
            and Grad-CAM visual heatmaps.
          </p>
        </div>

        <label className="cursor-pointer inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-accent-600 hover:bg-accent-700 text-white font-medium text-sm transition-all shadow-sm shrink-0">
          <UploadCloud className="h-4 w-4" />
          <span>Upload Custom CT (DICOM / PNG)</span>
          <input
            type="file"
            accept=".png,.jpg,.jpeg,.dcm,.dicom"
            className="hidden"
            onChange={handleFileUpload}
          />
        </label>
      </div>

      {errorMsg && (
        <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl flex items-center gap-3">
          <AlertTriangle className="h-5 w-5 shrink-0" />
          <p className="text-sm font-medium">{errorMsg}</p>
        </div>
      )}

      {/* Clinical Sample Selector */}
      <div className="bg-white rounded-2xl p-6 border border-border shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-warmgray-900 flex items-center gap-2">
            <Layers className="h-4 w-4 text-accent-600" />
            Curated IQ-OTH/NCCD Clinical Sample Scans
          </h2>
          <span className="text-xs text-warmgray-400">Select a scan to inspect anatomy & run pipeline</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {samples.map((s) => {
            const isSelected = selectedSampleId === s.id;
            const isMalignant = s.category === "Malignant";
            const isBenign = s.category === "Benign";

            return (
              <button
                key={s.id}
                onClick={() => handleSelectSample(s.id)}
                disabled={loadingSample}
                className={cn(
                  "text-left p-3.5 rounded-xl border transition-all relative flex flex-col justify-between",
                  isSelected
                    ? "border-accent-500 bg-accent-50/50 ring-2 ring-accent-500/20 shadow-sm"
                    : "border-border hover:border-warmgray-300 hover:bg-warmgray-50/60"
                )}
              >
                <div>
                  <div className="flex items-center justify-between gap-2 mb-1.5">
                    <span
                      className={cn(
                        "text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider",
                        isMalignant
                          ? "bg-red-100 text-red-700"
                          : isBenign
                          ? "bg-amber-100 text-amber-700"
                          : "bg-emerald-100 text-emerald-700"
                      )}
                    >
                      {s.category}
                    </span>
                    {isSelected && (
                      <span className="flex h-2 w-2 rounded-full bg-accent-600 animate-pulse" />
                    )}
                  </div>
                  <p className="text-xs font-bold text-warmgray-900 leading-snug">{s.name}</p>
                  <p className="text-[11px] text-warmgray-500 mt-1 line-clamp-2 leading-relaxed">
                    {s.description}
                  </p>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Inspection Grid: Image Viewer & Model Predictions */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left: CT & Visual Inspection (5 Cols) */}
        <div className="lg:col-span-5 bg-white rounded-2xl p-6 border border-border shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-border pb-3">
            <div>
              <h2 className="text-base font-semibold text-warmgray-900 flex items-center gap-2">
                <Eye className="h-4 w-4 text-accent-600" />
                CT Inspection & Region Segmentation
              </h2>
              <p className="text-xs text-warmgray-400 mt-0.5">
                {sessionData?.filename || "No scan loaded"}
              </p>
            </div>
            {loadingSample && <RefreshCw className="h-4 w-4 text-accent-500 animate-spin" />}
          </div>

          {/* View Mode Toggle Buttons */}
          <div className="flex rounded-xl bg-warmgray-100 p-1 text-xs font-medium">
            <button
              onClick={() => setActiveImageView("original")}
              className={cn(
                "flex-1 py-1.5 rounded-lg transition-all text-center",
                activeImageView === "original"
                  ? "bg-white text-warmgray-900 shadow-sm font-semibold"
                  : "text-warmgray-500 hover:text-warmgray-900"
              )}
            >
              Original CT
            </button>
            <button
              onClick={() => setActiveImageView("roi")}
              className={cn(
                "flex-1 py-1.5 rounded-lg transition-all text-center",
                activeImageView === "roi"
                  ? "bg-white text-warmgray-900 shadow-sm font-semibold"
                  : "text-warmgray-500 hover:text-warmgray-900"
              )}
            >
              Segmented ROI
            </button>
            <button
              onClick={() => setActiveImageView("gradcam")}
              className={cn(
                "flex-1 py-1.5 rounded-lg transition-all text-center",
                activeImageView === "gradcam"
                  ? "bg-white text-warmgray-900 shadow-sm font-semibold"
                  : "text-warmgray-500 hover:text-warmgray-900"
              )}
            >
              Grad-CAM Heatmap
            </button>
          </div>

          {/* Image Display Canvas */}
          <div className="relative aspect-square w-full rounded-xl bg-warmgray-950 flex items-center justify-center overflow-hidden border border-border">
            {sessionData ? (
              <img
                src={
                  activeImageView === "gradcam" && gradcamData?.gradcam_overlay_base64
                    ? gradcamData.gradcam_overlay_base64
                    : activeImageView === "roi"
                    ? sessionData.segmented_roi_base64
                    : sessionData.original_base64
                }
                alt="CT Scan Display"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-center p-6 text-warmgray-400">
                <ScanLine className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p className="text-xs">No CT Scan Selected</p>
              </div>
            )}

            {/* View Tag Badge */}
            <div className="absolute bottom-3 left-3 bg-black/75 backdrop-blur-sm text-white text-[10px] px-2.5 py-1 rounded-md font-mono">
              {activeImageView === "gradcam"
                ? "Grad-CAM Activation (JET)"
                : activeImageView === "roi"
                ? "Lung Parenchyma + ROI Mask"
                : "Axial CT (L: -600 HU, W: 1500 HU)"}
            </div>
          </div>

          {/* Inspection Details */}
          {activeImageView === "gradcam" && gradcamData ? (
            <div className="bg-accent-50/60 border border-accent-200/80 rounded-xl p-3 text-xs text-warmgray-700 leading-relaxed">
              <p className="font-semibold text-accent-900 mb-1 flex items-center gap-1.5">
                <Sparkles className="h-3.5 w-3.5 text-accent-600" />
                Visual Explainability:
              </p>
              {gradcamData.attention_summary}
            </div>
          ) : (
            <div className="grid grid-cols-3 gap-2 text-center text-xs">
              <div className="bg-warmgray-50 p-2.5 rounded-xl border border-border">
                <p className="text-[10px] text-warmgray-400 uppercase font-semibold">Matrix</p>
                <p className="font-mono font-bold text-warmgray-800 mt-0.5">
                  {sessionData ? `${sessionData.image_shape[0]}×${sessionData.image_shape[1]}` : "—"}
                </p>
              </div>
              <div className="bg-warmgray-50 p-2.5 rounded-xl border border-border">
                <p className="text-[10px] text-warmgray-400 uppercase font-semibold">Window</p>
                <p className="font-mono font-bold text-warmgray-800 mt-0.5">Lung (-600 HU)</p>
              </div>
              <div className="bg-warmgray-50 p-2.5 rounded-xl border border-border">
                <p className="text-[10px] text-warmgray-400 uppercase font-semibold">Mean Int.</p>
                <p className="font-mono font-bold text-warmgray-800 mt-0.5">
                  {sessionData?.mean_intensity ?? "—"}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Right: Predictions & Decision Support (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          <div className="bg-white rounded-2xl p-6 border border-border shadow-sm space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-border pb-4">
              <div>
                <h2 className="text-base font-semibold text-warmgray-900 flex items-center gap-2">
                  <Activity className="h-4 w-4 text-accent-600" />
                  Model Predictions & Clinical Decision Support
                </h2>
                <p className="text-xs text-warmgray-400 mt-0.5">
                  Inference generated strictly from medical CT imaging data
                </p>
              </div>

              {/* Model Filter Pills */}
              <div className="flex rounded-xl bg-warmgray-100 p-1 text-xs font-medium">
                {[
                  { id: "all", label: "All Models" },
                  { id: "cnn", label: "2D CNN" },
                  { id: "quantum", label: "VQC" },
                  { id: "hybrid", label: "Hybrid" },
                ].map((m) => (
                  <button
                    key={m.id}
                    onClick={() => setSelectedModelFilter(m.id)}
                    className={cn(
                      "px-2.5 py-1 rounded-lg transition-all",
                      selectedModelFilter === m.id
                        ? "bg-white text-warmgray-900 shadow-sm font-semibold"
                        : "text-warmgray-500 hover:text-warmgray-900"
                    )}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Cancer Risk Stratification Threshold Range Reference */}
            <div className="bg-warmgray-50/80 rounded-xl p-3.5 border border-border space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold uppercase tracking-wider text-warmgray-700 flex items-center gap-1.5">
                  <Sliders className="h-3.5 w-3.5 text-accent-600" />
                  Cancer Risk Threshold Ranges
                </span>
                <span className="text-[10px] text-warmgray-400 font-medium">Diagnostic Reference Scale</span>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center text-xs">
                <div className="bg-white p-2.5 rounded-lg border border-emerald-200/80 shadow-xs">
                  <div className="flex items-center justify-center gap-1.5 mb-1">
                    <span className="h-2 w-2 rounded-full bg-emerald-500" />
                    <span className="font-bold text-emerald-800 text-[11px]">Low Chances</span>
                  </div>
                  <p className="font-mono font-bold text-emerald-700 text-xs">&lt; 30.0%</p>
                  <p className="text-[10px] text-warmgray-500 mt-0.5">Normal / Non-suspicious</p>
                </div>
                <div className="bg-white p-2.5 rounded-lg border border-amber-200/80 shadow-xs">
                  <div className="flex items-center justify-center gap-1.5 mb-1">
                    <span className="h-2 w-2 rounded-full bg-amber-500" />
                    <span className="font-bold text-amber-800 text-[11px]">Medium Chances</span>
                  </div>
                  <p className="font-mono font-bold text-amber-700 text-xs">30.0% – 64.9%</p>
                  <p className="text-[10px] text-warmgray-500 mt-0.5">Indeterminate / Surveillance</p>
                </div>
                <div className="bg-white p-2.5 rounded-lg border border-red-200/80 shadow-xs">
                  <div className="flex items-center justify-center gap-1.5 mb-1">
                    <span className="h-2 w-2 rounded-full bg-red-500" />
                    <span className="font-bold text-red-800 text-[11px]">High Chances</span>
                  </div>
                  <p className="font-mono font-bold text-red-700 text-xs">≥ 65.0%</p>
                  <p className="text-[10px] text-warmgray-500 mt-0.5">Suspicious / Oncology Workup</p>
                </div>
              </div>
            </div>

            {/* Predictions List */}
            {analyzing ? (
              <div className="py-12 text-center text-warmgray-400">
                <RefreshCw className="h-7 w-7 mx-auto mb-2 animate-spin text-accent-600" />
                <p className="text-xs font-medium">Executing quantum-classical inference...</p>
              </div>
            ) : filteredPredictions && filteredPredictions.length > 0 ? (
              <div className="space-y-4">
                {filteredPredictions.map((pred) => {
                  const isMalignant = pred.predicted_label === "Malignant";
                  const isBenign = pred.predicted_label === "Benign";

                  return (
                    <div
                      key={pred.model_name}
                      className="rounded-xl border border-border p-4 hover:border-warmgray-300 transition-all bg-warmgray-50/40"
                    >
                      <div className="flex items-center justify-between gap-2 mb-3">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-sm text-warmgray-900">
                            {pred.model_name}
                          </span>
                          <span className="text-[10px] font-semibold text-warmgray-400 bg-warmgray-200/70 px-2 py-0.5 rounded-full uppercase">
                            {pred.model_type}
                          </span>
                        </div>

                        {/* Predicted Label Badge */}
                        <div className="flex items-center gap-2">
                          <span
                            className={cn(
                              "text-xs font-bold px-2.5 py-1 rounded-full border flex items-center gap-1",
                              isMalignant
                                ? "bg-red-100/80 text-red-800 border-red-200"
                                : isBenign
                                ? "bg-amber-100/80 text-amber-800 border-amber-200"
                                : "bg-emerald-100/80 text-emerald-800 border-emerald-200"
                            )}
                          >
                            {isMalignant ? (
                              <AlertTriangle className="h-3 w-3" />
                            ) : (
                              <CheckCircle2 className="h-3 w-3" />
                            )}
                            {pred.predicted_label}
                          </span>

                          <span
                            className={cn(
                              "text-xs font-bold px-2.5 py-1 rounded-full border",
                              getRiskColor(pred.risk_tier)
                            )}
                          >
                            {pred.risk_tier} Risk
                          </span>
                        </div>
                      </div>

                      {/* Malignancy Probability Bar with Threshold Ticks */}
                      <div className="space-y-1.5 mb-3">
                        <div className="flex justify-between text-xs">
                          <span className="text-warmgray-500 font-medium">Malignancy Risk Probability</span>
                          <span className="font-mono font-bold text-warmgray-900">
                            {(pred.malignancy_risk_score * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="relative h-2.5 w-full bg-warmgray-200 rounded-full overflow-hidden">
                          <div
                            className={cn("h-full transition-all duration-500", getRiskBarColor(pred.malignancy_risk_score))}
                            style={{ width: `${Math.min(pred.malignancy_risk_score * 100, 100)}%` }}
                          />
                          {/* 30% Threshold tick mark */}
                          <div className="absolute top-0 bottom-0 left-[30%] w-[1.5px] bg-white/90 shadow-xs pointer-events-none" />
                          {/* 65% Threshold tick mark */}
                          <div className="absolute top-0 bottom-0 left-[65%] w-[1.5px] bg-white/90 shadow-xs pointer-events-none" />
                        </div>
                        <div className="flex justify-between text-[10px] font-medium text-warmgray-400 px-0.5">
                          <span className="text-emerald-600">0% Low (&lt;30%)</span>
                          <span className="text-amber-600 text-center">30% Medium (30-65%)</span>
                          <span className="text-red-600 text-right">65% High (≥65%) 100%</span>
                        </div>
                      </div>

                      {/* Multi-Class Probability Breakdown */}
                      <div className="grid grid-cols-3 gap-2 py-2 border-t border-border text-center text-xs">
                        <div>
                          <p className="text-[10px] text-warmgray-400 uppercase">P(Normal)</p>
                          <p className="font-mono font-semibold text-warmgray-700">
                            {(pred.class_probabilities.Normal * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div>
                          <p className="text-[10px] text-warmgray-400 uppercase">P(Benign)</p>
                          <p className="font-mono font-semibold text-warmgray-700">
                            {(pred.class_probabilities.Benign * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div>
                          <p className="text-[10px] text-warmgray-400 uppercase">P(Malignant)</p>
                          <p className="font-mono font-semibold text-warmgray-700">
                            {(pred.class_probabilities.Malignant * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>

                      {/* Cancer Staging Indicator */}
                      <div className="mt-2.5 pt-2 border-t border-border flex items-center justify-between text-xs text-warmgray-500">
                        <span className="font-medium">Cancer Stage:</span>
                        <span className="font-mono text-warmgray-600 bg-white px-2 py-0.5 rounded border border-border text-[11px]">
                          {pred.predicted_stage}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="py-8 text-center text-warmgray-400 text-xs">
                No predictions available. Please select or upload a CT scan.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Multi-Model Benchmark & Metric Evaluation Section */}
      {benchmarks && (
        <div className="bg-white rounded-2xl p-6 sm:p-8 border border-border shadow-sm space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-border pb-4">
            <div>
              <div className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-accent-600" />
                <h2 className="text-lg font-bold text-warmgray-900">
                  Imaging Model Benchmark & Evaluation
                </h2>
              </div>
              <p className="text-xs text-warmgray-500 mt-1">
                Comparative evaluation across Classical CNN, PennyLane VQC, and Hybrid HQ-CNN models
                on the test partition ({benchmarks.dataset_name})
              </p>
            </div>
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 shrink-0">
              <CheckCircle2 className="h-3.5 w-3.5" /> Best Model: {benchmarks.best_model}
            </span>
          </div>

          {/* Comparative Metrics Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-border bg-warmgray-50 text-warmgray-500 uppercase tracking-wider font-semibold">
                  <th className="py-3 px-4">Model Architecture</th>
                  <th className="py-3 px-4">Type</th>
                  <th className="py-3 px-4">Accuracy</th>
                  <th className="py-3 px-4">Sensitivity (Recall)</th>
                  <th className="py-3 px-4">Specificity</th>
                  <th className="py-3 px-4">Precision</th>
                  <th className="py-3 px-4">F1-Score</th>
                  <th className="py-3 px-4">ROC-AUC</th>
                  <th className="py-3 px-4">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {benchmarks.models.map((m) => (
                  <tr key={m.model_name} className="hover:bg-warmgray-50/50 transition-colors">
                    <td className="py-3 px-4 font-bold text-warmgray-900">{m.model_name}</td>
                    <td className="py-3 px-4">
                      <span className="uppercase text-[10px] font-bold px-2 py-0.5 rounded bg-warmgray-100 text-warmgray-600">
                        {m.model_type}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono font-semibold">
                      {(m.accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 font-mono">{(m.sensitivity * 100).toFixed(2)}%</td>
                    <td className="py-3 px-4 font-mono">{(m.specificity * 100).toFixed(2)}%</td>
                    <td className="py-3 px-4 font-mono">{(m.precision * 100).toFixed(2)}%</td>
                    <td className="py-3 px-4 font-mono font-bold text-accent-700">
                      {(m.f1_score * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-4 font-mono">{m.roc_auc ? (m.roc_auc * 100).toFixed(2) + "%" : "—"}</td>
                    <td className="py-3 px-4 font-mono text-warmgray-500">{m.inference_time_ms} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Charts Grid: Bar Chart & Confusion Matrices */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 pt-4 border-t border-border">
            {/* Bar Chart Comparison (7 Cols) */}
            <div className="lg:col-span-7 bg-warmgray-50/50 p-4 rounded-xl border border-border space-y-2">
              <h3 className="text-xs font-bold text-warmgray-700 uppercase tracking-wider">
                Performance Comparison by Metric
              </h3>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={[
                      {
                        metric: "Accuracy",
                        CNN: 93.9,
                        VQC: 89.4,
                        Hybrid: 95.6,
                      },
                      {
                        metric: "Sensitivity",
                        CNN: 93.9,
                        VQC: 89.4,
                        Hybrid: 95.6,
                      },
                      {
                        metric: "Specificity",
                        CNN: 96.9,
                        VQC: 94.7,
                        Hybrid: 97.8,
                      },
                      {
                        metric: "F1-Score",
                        CNN: 94.0,
                        VQC: 89.5,
                        Hybrid: 95.6,
                      },
                    ]}
                    margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                  >
                    <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
                    <YAxis domain={[80, 100]} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="CNN" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="VQC" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Hybrid" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Confusion Matrix (5 Cols) */}
            <div className="lg:col-span-5 bg-warmgray-50/50 p-4 rounded-xl border border-border space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-warmgray-700 uppercase tracking-wider">
                  Hybrid Model Confusion Matrix
                </h3>
                <span className="text-[10px] text-warmgray-400">N=180 Samples</span>
              </div>

              <div className="text-xs">
                <div className="grid grid-cols-4 gap-1 text-center font-semibold mb-1 text-[11px] text-warmgray-500">
                  <div className="text-left pl-1">Actual \ Pred</div>
                  <div>Normal</div>
                  <div>Benign</div>
                  <div>Malignant</div>
                </div>

                {[
                  { label: "Normal", values: [58, 2, 0] },
                  { label: "Benign", values: [2, 56, 2] },
                  { label: "Malignant", values: [0, 2, 58] },
                ].map((row, rIdx) => (
                  <div key={row.label} className="grid grid-cols-4 gap-1 text-center mb-1">
                    <div className="font-semibold text-left pl-1 text-[11px] flex items-center text-warmgray-700">
                      {row.label}
                    </div>
                    {row.values.map((val, cIdx) => {
                      const isDiagonal = rIdx === cIdx;
                      return (
                        <div
                          key={cIdx}
                          className={cn(
                            "py-2 rounded font-mono font-bold text-xs flex items-center justify-center transition-all",
                            isDiagonal
                              ? "bg-accent-600 text-white shadow-xs"
                              : val > 0
                              ? "bg-amber-100 text-amber-800"
                              : "bg-warmgray-100 text-warmgray-400"
                          )}
                        >
                          {val}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>

              <p className="text-[11px] text-warmgray-400 mt-2 leading-relaxed">
                Diagonal values indicate correct classifications. The hybrid model demonstrates minimal
                cross-classification error between Benign nodules and Malignant lesions.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
