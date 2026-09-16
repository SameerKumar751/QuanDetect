import React, { useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ShieldAlert,
  ShieldCheck,
  ShieldHalf,
  Loader2,
  AlertTriangle,
  Sparkles,
  CheckCircle2,
  Circle,
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
  ReferenceLine,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { endpoints } from "@/lib/api";
import { useAppState } from "@/lib/AppStateContext";

const RISK_TIERS = [
  { level: "Low", range: "0% – 33%", icon: ShieldCheck, color: "text-primary-600 bg-primary-50 border-primary-100" },
  { level: "Medium", range: "33% – 66%", icon: ShieldHalf, color: "text-amber-600 bg-amber-50 border-amber-100" },
  { level: "High", range: "66% – 100%", icon: ShieldAlert, color: "text-red-600 bg-red-50 border-red-100" },
];

export default function Results() {
  const { sessionId, dataset, preprocessInfo, classicalResults, quantumResult } = useAppState();
  const [modelName, setModelName] = useState("random_forest");
  const [sampleIndex, setSampleIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [explainData, setExplainData] = useState(null);
  const [error, setError] = useState(null);

  const steps = [
    { label: "Dataset uploaded", done: !!dataset },
    { label: "Preprocessing complete", done: !!preprocessInfo },
    { label: "Classical models trained", done: !!classicalResults },
    { label: "Quantum VQC trained", done: !!quantumResult },
  ];

  const runExplain = async () => {
    setError(null);
    setLoading(true);
    setExplainData(null);
    try {
      const res = await endpoints.explain({
        session_id: sessionId,
        model_name: modelName,
        sample_index: Number(sampleIndex) || 0,
      });
      setExplainData(res.data);
    } catch (e) {
      setError(e?.response?.data?.detail || "Explainability failed. Train the selected model first.");
    } finally {
      setLoading(false);
    }
  };

  const chartData = explainData
    ? explainData.feature_names.map((f, i) => ({
        name: f.length > 14 ? f.slice(0, 14) + "…" : f,
        shap: explainData.shap_values[i],
      })).sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
    : [];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-warmgray-900">Risk Stratification & Explainability</h1>
        <p className="text-warmgray-500 mt-1">Understand how risk tiers are defined and why the model made a given prediction.</p>
      </div>

      {/* Pipeline status */}
      <Card>
        <CardHeader>
          <CardTitle>Pipeline Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {steps.map((s, i) => (
              <motion.div
                key={s.label}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.08 }}
                className={`flex items-center gap-2 rounded-xl px-4 py-3 border ${
                  s.done ? "bg-primary-50 border-primary-100 text-primary-700" : "bg-warmgray-50 border-border text-warmgray-400"
                }`}
              >
                {s.done ? <CheckCircle2 className="h-4 w-4 shrink-0" /> : <Circle className="h-4 w-4 shrink-0" />}
                <span className="text-sm font-medium">{s.label}</span>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Risk tiers legend */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
        {RISK_TIERS.map((tier, i) => (
          <motion.div key={tier.level} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}>
            <Card className={`border-2 ${tier.color}`}>
              <CardContent className="p-6 text-center">
                <tier.icon className="h-8 w-8 mx-auto mb-2" />
                <p className="text-lg font-bold">{tier.level} Risk</p>
                <p className="text-sm opacity-70 mt-1">Probability {tier.range}</p>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Explainability */}
      {!sessionId ? (
        <Card className="border-dashed">
          <CardContent className="p-10 text-center">
            <AlertTriangle className="h-8 w-8 text-amber-500 mx-auto mb-3" />
            <p className="font-medium text-warmgray-700">No active session</p>
            <p className="text-sm text-warmgray-500 mt-1 mb-5">Upload a dataset to unlock explainability tools.</p>
            <Link to="/upload">
              <Button>Go to Upload</Button>
            </Link>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-accent-500" /> Explainability (SHAP)
            </CardTitle>
            <CardDescription>
              See which features pushed a specific test-set prediction toward or away from a positive diagnosis.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="flex flex-wrap items-end gap-3">
              <div>
                <label className="text-xs font-medium text-warmgray-500 mb-1.5 block">Classical Model</label>
                <select
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  className="h-10 rounded-xl border border-border bg-white px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value="random_forest">Random Forest</option>
                  <option value="xgboost">XGBoost</option>
                  <option value="svm">SVM</option>
                </select>
              </div>
              <div>
                <label className="text-xs font-medium text-warmgray-500 mb-1.5 block">Test Sample Index</label>
                <Input type="number" min={0} value={sampleIndex} onChange={(e) => setSampleIndex(e.target.value)} className="w-32" />
              </div>
              <Button onClick={runExplain} disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" /> Explaining...
                  </>
                ) : (
                  "Explain Prediction"
                )}
              </Button>
            </div>

            {error && (
              <div className="flex items-center gap-2 text-sm text-red-600 bg-red-50 border border-red-100 rounded-xl px-4 py-3">
                <AlertTriangle className="h-4 w-4 shrink-0" /> {error}
              </div>
            )}

            {explainData && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                <div className="flex items-center gap-4 text-sm">
                  <Badge variant="accent">Base value: {explainData.base_value.toFixed(3)}</Badge>
                  <Badge variant="default">Prediction: {(explainData.prediction * 100).toFixed(1)}%</Badge>
                </div>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={chartData} layout="vertical" margin={{ left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E7E5E4" horizontal={false} />
                    <XAxis type="number" tick={{ fontSize: 10 }} stroke="#A8A29E" />
                    <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={110} stroke="#A8A29E" />
                    <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #E7E5E4", fontSize: 12 }} />
                    <ReferenceLine x={0} stroke="#A8A29E" />
                    <Bar dataKey="shap" radius={[0, 6, 6, 0]}>
                      {chartData.map((d, i) => (
                        <Cell key={i} fill={d.shap >= 0 ? "#DC2626" : "#0F766E"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <p className="text-xs text-warmgray-400">
                  Red bars push the prediction toward disease-positive; teal bars push toward disease-negative.
                </p>
              </motion.div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
