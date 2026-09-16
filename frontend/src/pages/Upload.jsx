import React, { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  UploadCloud,
  FileSpreadsheet,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  ArrowRight,
  Settings2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { endpoints } from "@/lib/api";
import { useAppState } from "@/lib/AppStateContext";

export default function Upload() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [preprocessing, setPreprocessing] = useState(false);
  const [featureK, setFeatureK] = useState(10);

  const { sessionId, setSessionId, dataset, setDataset, setPreprocessInfo } = useAppState();

  const handleFile = async (file) => {
    setError(null);
    setLoading(true);
    try {
      const res = await endpoints.uploadDataset(file);
      setSessionId(res.data.session_id);
      setDataset(res.data);
      setTargetColumn(res.data.detected_target_column || "");
    } catch (e) {
      setError(e?.response?.data?.detail || "Upload failed. Please check your CSV file.");
    } finally {
      setLoading(false);
    }
  };

  const handleUseSample = async () => {
    setError(null);
    setLoading(true);
    try {
      const res = await endpoints.useSampleDataset();
      setSessionId(res.data.session_id);
      setDataset(res.data);
      setTargetColumn(res.data.detected_target_column || "target");
    } catch (e) {
      setError(e?.response?.data?.detail || "Could not load sample dataset.");
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const runPreprocessing = async () => {
    if (!sessionId || !targetColumn) return;
    setPreprocessing(true);
    setError(null);
    try {
      const res = await endpoints.preprocess({
        session_id: sessionId,
        target_column: targetColumn,
        missing_strategy: "median",
        scale: true,
        feature_selection_k: Number(featureK) || 10,
      });
      setPreprocessInfo(res.data);
      navigate("/training");
    } catch (e) {
      setError(e?.response?.data?.detail || "Preprocessing failed.");
    } finally {
      setPreprocessing(false);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-warmgray-900">Dataset Upload</h1>
        <p className="text-warmgray-500 mt-1">
          Upload any tabular biomedical CSV — or start instantly with the bundled sample dataset.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload zone */}
        <Card>
          <CardHeader>
            <CardTitle>Upload CSV</CardTitle>
            <CardDescription>Hospital records, genomics panels, or any similar tabular dataset.</CardDescription>
          </CardHeader>
          <CardContent>
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition-colors ${
                dragActive ? "border-accent-400 bg-accent-50" : "border-border bg-warmgray-50 hover:bg-primary-50/40"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              />
              <UploadCloud className="h-10 w-10 mx-auto text-primary-500 mb-3" />
              <p className="text-sm font-medium text-warmgray-700">Drag & drop your CSV here</p>
              <p className="text-xs text-warmgray-400 mt-1">or click to browse files</p>
            </div>

            <div className="flex items-center gap-3 my-5">
              <div className="h-px bg-border flex-1" />
              <span className="text-xs text-warmgray-400">OR</span>
              <div className="h-px bg-border flex-1" />
            </div>

            <Button variant="subtle" className="w-full" onClick={handleUseSample} disabled={loading}>
              <Sparkles className="h-4 w-4" />
              Use Demo Dataset (Survey Lung Cancer)
            </Button>

            {loading && (
              <div className="flex items-center gap-2 text-sm text-warmgray-500 mt-4">
                <Loader2 className="h-4 w-4 animate-spin" /> Uploading & cleaning dataset...
              </div>
            )}
            {error && (
              <div className="flex items-center gap-2 text-sm text-red-600 mt-4 bg-red-50 border border-red-100 rounded-xl px-3 py-2">
                <AlertTriangle className="h-4 w-4 shrink-0" /> {error}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Config panel */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings2 className="h-5 w-5 text-accent-500" /> Preprocessing Config
            </CardTitle>
            <CardDescription>Feature engineering & pipeline settings before training.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {!dataset ? (
              <div className="space-y-3">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-4 w-1/2" />
                <Skeleton className="h-24 w-full" />
              </div>
            ) : (
              <>
                <div>
                  <label className="text-xs font-medium text-warmgray-500 mb-1.5 block">Target / Label Column</label>
                  <select
                    value={targetColumn}
                    onChange={(e) => setTargetColumn(e.target.value)}
                    className="w-full h-10 rounded-xl border border-border bg-white px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                  >
                    <option value="">Select column...</option>
                    {dataset.columns.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-xs font-medium text-warmgray-500 mb-1.5 block">
                    Top-K Features (ANOVA F-test selection)
                  </label>
                  <Input type="number" min={2} max={dataset.n_columns - 1} value={featureK} onChange={(e) => setFeatureK(e.target.value)} />
                </div>

                <div className="rounded-xl bg-warmgray-50 p-4 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-warmgray-500">Rows</span>
                    <span className="font-medium text-warmgray-900">{dataset.n_rows}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-warmgray-500">Columns</span>
                    <span className="font-medium text-warmgray-900">{dataset.n_columns}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-warmgray-500">Missing cells</span>
                    <span className="font-medium text-warmgray-900">
                      {Object.values(dataset.missing_value_summary || {}).reduce((a, b) => a + b, 0)}
                    </span>
                  </div>
                </div>

                <Button className="w-full" onClick={runPreprocessing} disabled={!targetColumn || preprocessing}>
                  {preprocessing ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" /> Running Pipeline...
                    </>
                  ) : (
                    <>
                      Run Preprocessing & Continue <ArrowRight className="h-4 w-4" />
                    </>
                  )}
                </Button>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Preview table */}
      <AnimatePresence>
        {dataset && (
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <FileSpreadsheet className="h-5 w-5 text-primary-600" /> Dataset Preview
                    </CardTitle>
                    <CardDescription>First 5 rows</CardDescription>
                  </div>
                  {dataset.detected_target_column && (
                    <Badge variant="success">
                      <CheckCircle2 className="h-3 w-3 mr-1" /> Target detected: {dataset.detected_target_column}
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto rounded-xl border border-border">
                  <table className="min-w-full text-xs">
                    <thead className="bg-warmgray-50">
                      <tr>
                        {dataset.columns.slice(0, 12).map((c) => (
                          <th key={c} className="px-3 py-2 text-left font-semibold text-warmgray-600 whitespace-nowrap">
                            {c}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {dataset.preview.map((row, i) => (
                        <tr key={i} className="border-t border-border">
                          {dataset.columns.slice(0, 12).map((c) => (
                            <td key={c} className="px-3 py-2 whitespace-nowrap text-warmgray-600">
                              {row[c] !== null && row[c] !== undefined ? String(row[c]).slice(0, 12) : "—"}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {dataset.columns.length > 12 && (
                  <p className="text-xs text-warmgray-400 mt-2">
                    + {dataset.columns.length - 12} more columns not shown in preview
                  </p>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
