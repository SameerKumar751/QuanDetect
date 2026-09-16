import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  BarChart3,
  Trophy,
  AlertTriangle,
  Loader2,
  Atom,
  Cpu,
} from "lucide-react";
import {
  BarChart,
  Bar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { endpoints } from "@/lib/api";
import { useAppState } from "@/lib/AppStateContext";
import { formatPercent } from "@/lib/utils";

const METRIC_KEYS = ["accuracy", "sensitivity", "specificity", "precision", "f1_score"];
const METRIC_LABELS = {
  accuracy: "Accuracy",
  sensitivity: "Sensitivity",
  specificity: "Specificity",
  precision: "Precision",
  f1_score: "F1-Score",
};
const MODEL_COLORS = {
  random_forest: "#0F766E",
  xgboost: "#0D9488",
  svm: "#14B8A6",
  vqc: "#8B5CF6",
};

export default function Comparison() {
  const { sessionId, classicalResults, quantumResult } = useAppState();
  const [loading, setLoading] = useState(false);
  const [compareData, setCompareData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (sessionId && (classicalResults || quantumResult)) {
      fetchComparison();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const fetchComparison = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await endpoints.compare(sessionId);
      setCompareData(res.data);
    } catch (e) {
      setError(e?.response?.data?.detail || "Could not load comparison data.");
    } finally {
      setLoading(false);
    }
  };

  if (!sessionId) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold text-warmgray-900">Comparison Dashboard</h1>
        <Card className="border-dashed">
          <CardContent className="p-10 text-center">
            <AlertTriangle className="h-8 w-8 text-amber-500 mx-auto mb-3" />
            <p className="font-medium text-warmgray-700">No trained models found</p>
            <p className="text-sm text-warmgray-500 mt-1 mb-5">Train classical and quantum models first.</p>
            <Link to="/training">
              <Button>Go to Training</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  const allResults = [...(compareData?.classical_results || []), ...(compareData?.quantum_result ? [compareData.quantum_result] : [])];

  const barData = METRIC_KEYS.map((metric) => {
    const entry = { metric: METRIC_LABELS[metric] };
    allResults.forEach((r) => {
      entry[r.model_name] = Number((r[metric] * 100).toFixed(1));
    });
    return entry;
  });

  const radarData = METRIC_KEYS.map((metric) => {
    const entry = { metric: METRIC_LABELS[metric] };
    allResults.forEach((r) => {
      entry[r.model_name] = Number((r[metric] * 100).toFixed(1));
    });
    return entry;
  });

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-bold text-warmgray-900">Comparison Dashboard</h1>
          <p className="text-warmgray-500 mt-1">Classical baselines vs. Hybrid Quantum Classifier — head to head.</p>
        </div>
        {compareData?.best_model && (
          <Badge variant="accent" className="text-sm px-4 py-2">
            <Trophy className="h-3.5 w-3.5 mr-1.5" /> Best model: {compareData.best_model.replace("_", " ")}
          </Badge>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-2 text-sm text-red-600 bg-red-50 border border-red-100 rounded-xl px-4 py-3">
          <AlertTriangle className="h-4 w-4 shrink-0" /> {error}
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Skeleton className="h-80 w-full" />
          <Skeleton className="h-80 w-full" />
        </div>
      ) : allResults.length === 0 ? (
        <Card className="border-dashed">
          <CardContent className="p-10 text-center text-warmgray-400">No results yet — train models to see the comparison.</CardContent>
        </Card>
      ) : (
        <>
          {/* Metric cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {allResults.map((r, i) => (
              <motion.div key={r.model_name} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.08 }}>
                <Card>
                  <CardContent className="p-5">
                    <div className="flex items-center gap-2 mb-3">
                      {r.model_name === "vqc" ? (
                        <Atom className="h-4 w-4 text-accent-500" />
                      ) : (
                        <Cpu className="h-4 w-4 text-primary-600" />
                      )}
                      <span className="text-sm font-semibold text-warmgray-800 capitalize">{r.model_name.replace("_", " ")}</span>
                    </div>
                    <p className="text-3xl font-extrabold text-warmgray-900">{formatPercent(r.f1_score)}</p>
                    <p className="text-xs text-warmgray-400">F1-Score</p>
                    <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
                      <div>
                        <span className="text-warmgray-400">Acc </span>
                        <span className="font-medium text-warmgray-700">{formatPercent(r.accuracy)}</span>
                      </div>
                      <div>
                        <span className="text-warmgray-400">AUC </span>
                        <span className="font-medium text-warmgray-700">{r.roc_auc ? formatPercent(r.roc_auc) : "--"}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>

          {/* Bar chart */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-primary-600" /> Metric-by-Metric Benchmark
              </CardTitle>
              <CardDescription>Accuracy, Sensitivity, Specificity, Precision & F1-Score (%)</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={340}>
                <BarChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E7E5E4" />
                  <XAxis dataKey="metric" tick={{ fontSize: 11 }} stroke="#A8A29E" />
                  <YAxis tick={{ fontSize: 11 }} stroke="#A8A29E" domain={[0, 100]} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #E7E5E4", fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v) => v.replace("_", " ")} />
                  {allResults.map((r) => (
                    <Bar key={r.model_name} dataKey={r.model_name} fill={MODEL_COLORS[r.model_name] || "#78716C"} radius={[6, 6, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Radar chart */}
          <Card>
            <CardHeader>
              <CardTitle>Performance Radar</CardTitle>
              <CardDescription>Holistic shape comparison across all metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={360}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#E7E5E4" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: "#78716C" }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} />
                  {allResults.map((r) => (
                    <Radar
                      key={r.model_name}
                      name={r.model_name.replace("_", " ")}
                      dataKey={r.model_name}
                      stroke={MODEL_COLORS[r.model_name] || "#78716C"}
                      fill={MODEL_COLORS[r.model_name] || "#78716C"}
                      fillOpacity={0.15}
                    />
                  ))}
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #E7E5E4", fontSize: 12 }} />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
