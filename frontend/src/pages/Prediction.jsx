import React, { useState } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Stethoscope,
  Loader2,
  AlertTriangle,
  ActivitySquare,
  Gauge,
  ListTree,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { endpoints } from "@/lib/api";
import { useAppState } from "@/lib/AppStateContext";
import { formatPercent, riskColor } from "@/lib/utils";

const MODEL_OPTIONS = [
  { key: "random_forest", label: "Random Forest" },
  { key: "xgboost", label: "XGBoost" },
  { key: "svm", label: "SVM" },
  { key: "vqc", label: "Hybrid VQC (Quantum)" },
];

export default function Prediction() {
  const { sessionId, preprocessInfo, dataset } = useAppState();
  const [modelName, setModelName] = useState("random_forest");
  const [features, setFeatures] = useState({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const selectedFeatures = preprocessInfo?.selected_features || [];

  const handleChange = (feature, value) => {
    setFeatures((prev) => ({ ...prev, [feature]: value }));
  };

  const prefillFromSample = () => {
    if (!dataset?.preview?.length) return;
    const sample = dataset.preview[0];
    const filled = {};
    selectedFeatures.forEach((f) => {
      if (sample[f] !== undefined && sample[f] !== null) filled[f] = sample[f];
    });
    setFeatures(filled);
  };

  const runPrediction = async () => {
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const preparedFeatures = {};
      selectedFeatures.forEach((f) => {
        let val = features[f];
        if (val === undefined || val === "") {
          val = f.toUpperCase() === "AGE" ? 60 : 1;
        }
        if (f.toUpperCase() === "GENDER") {
          preparedFeatures[f] = val === "F" || val === 0 || val === "0" ? 0 : 1;
        } else {
          preparedFeatures[f] = Number(val) || 0;
        }
      });
      const res = await endpoints.predict({
        session_id: sessionId,
        model_name: modelName,
        features: preparedFeatures,
      });
      setResult(res.data);
    } catch (e) {
      setError(e?.response?.data?.detail || "Prediction failed. Ensure the selected model has been trained.");
    } finally {
      setLoading(false);
    }
  };

  if (!sessionId || !preprocessInfo) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold text-warmgray-900">Prediction</h1>
        <Card className="border-dashed">
          <CardContent className="p-10 text-center">
            <AlertTriangle className="h-8 w-8 text-amber-500 mx-auto mb-3" />
            <p className="font-medium text-warmgray-700">No preprocessed dataset found</p>
            <p className="text-sm text-warmgray-500 mt-1 mb-5">Upload a dataset and train a model before predicting.</p>
            <Link to="/upload">
              <Button>Go to Upload</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  const chartData = result?.top_contributing_features?.map((f) => ({
    name: f.feature.length > 14 ? f.feature.slice(0, 14) + "…" : f.feature,
    importance: f.importance,
  })) || [];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-warmgray-900">Prediction</h1>
        <p className="text-warmgray-500 mt-1">Enter patient survey and risk factor values to generate a lung cancer risk prediction.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Form */}
        <Card className="lg:col-span-3">
          <CardHeader>
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Stethoscope className="h-5 w-5 text-primary-600" /> Patient Clinical & Survey Features
                </CardTitle>
                <CardDescription>{selectedFeatures.length} model input features (Survey: 1 = No, 2 = Yes)</CardDescription>
              </div>
              <Button variant="ghost" size="sm" onClick={prefillFromSample}>
                Prefill from sample
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-h-[420px] overflow-y-auto scrollbar-thin pr-2">
              {selectedFeatures.map((f) => {
                const isGender = f.toUpperCase() === "GENDER";
                const isAge = f.toUpperCase() === "AGE";

                if (isGender) {
                  return (
                    <div key={f}>
                      <label className="text-xs font-medium text-warmgray-700 mb-1.5 block truncate" title={f}>
                        {f} (Gender)
                      </label>
                      <select
                        value={features[f] ?? "1"}
                        onChange={(e) => handleChange(f, e.target.value)}
                        className="w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-xs focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      >
                        <option value="1">Male (M)</option>
                        <option value="0">Female (F)</option>
                      </select>
                    </div>
                  );
                }

                if (!isAge) {
                  return (
                    <div key={f}>
                      <label className="text-xs font-medium text-warmgray-700 mb-1.5 block truncate" title={f}>
                        {f}
                      </label>
                      <select
                        value={features[f] ?? "2"}
                        onChange={(e) => handleChange(f, e.target.value)}
                        className="w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-xs focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      >
                        <option value="2">2 - Yes (Present)</option>
                        <option value="1">1 - No (Absent)</option>
                      </select>
                    </div>
                  );
                }

                return (
                  <div key={f}>
                    <label className="text-xs font-medium text-warmgray-700 mb-1.5 block truncate" title={f}>
                      {f} (Years)
                    </label>
                    <Input
                      type="number"
                      step="1"
                      value={features[f] ?? ""}
                      onChange={(e) => handleChange(f, e.target.value)}
                      placeholder="e.g. 65"
                    />
                  </div>
                );
              })}
            </div>

            <div className="mt-6 flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
              <select
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="h-10 rounded-xl border border-border bg-white px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                {MODEL_OPTIONS.map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.label}
                  </option>
                ))}
              </select>
              <Button className="flex-1" onClick={runPrediction} disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" /> Predicting...
                  </>
                ) : (
                  <>
                    <ActivitySquare className="h-4 w-4" /> Run Prediction
                  </>
                )}
              </Button>
            </div>

            {error && (
              <div className="flex items-center gap-2 text-sm text-red-600 bg-red-50 border border-red-100 rounded-xl px-4 py-3 mt-4">
                <AlertTriangle className="h-4 w-4 shrink-0" /> {error}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Result */}
        <div className="lg:col-span-2 space-y-6">
          <AnimatePresence mode="wait">
            {result ? (
              <motion.div
                key="result"
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
              >
                <Card className={`border-2 ${riskColor(result.risk_level)}`}>
                  <CardContent className="p-6 text-center">
                    <Gauge className="h-8 w-8 mx-auto mb-2 opacity-70" />
                    <p className="text-sm font-medium opacity-80">Disease Probability</p>
                    <p className="text-4xl font-extrabold mt-1">{formatPercent(result.disease_probability)}</p>
                    <Badge className="mt-3" variant={result.risk_level === "Low" ? "success" : result.risk_level === "Medium" ? "warning" : "danger"}>
                      {result.risk_level} Risk
                    </Badge>
                    <p className="text-xs opacity-60 mt-3">
                      Confidence {formatPercent(result.confidence)} &middot; Model: {result.model_name}
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            ) : (
              <Card className="border-dashed">
                <CardContent className="p-10 text-center">
                  <Gauge className="h-8 w-8 mx-auto text-warmgray-300 mb-2" />
                  <p className="text-sm text-warmgray-400">Prediction results will appear here</p>
                </CardContent>
              </Card>
            )}
          </AnimatePresence>

          {result && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <ListTree className="h-4 w-4 text-accent-500" /> Top Contributing Features
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={chartData} layout="vertical" margin={{ left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E7E5E4" horizontal={false} />
                    <XAxis type="number" tick={{ fontSize: 10 }} stroke="#A8A29E" />
                    <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={100} stroke="#A8A29E" />
                    <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #E7E5E4", fontSize: 12 }} />
                    <Bar dataKey="importance" radius={[0, 6, 6, 0]}>
                      {chartData.map((_, i) => (
                        <Cell key={i} fill={i % 2 === 0 ? "#0F766E" : "#8B5CF6"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
