import React, { useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Cpu,
  Atom,
  Loader2,
  PlayCircle,
  CheckCircle2,
  ArrowRight,
  AlertTriangle,
  Zap,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { endpoints } from "@/lib/api";
import { useAppState } from "@/lib/AppStateContext";
import { formatPercent } from "@/lib/utils";

const CLASSICAL_MODELS = [
  { key: "random_forest", label: "Random Forest" },
  { key: "xgboost", label: "XGBoost" },
  { key: "svm", label: "SVM" },
];

export default function Training() {
  const { sessionId, preprocessInfo, classicalResults, setClassicalResults, quantumResult, setQuantumResult } =
    useAppState();

  const [selectedModels, setSelectedModels] = useState(CLASSICAL_MODELS.map((m) => m.key));
  const [trainingClassical, setTrainingClassical] = useState(false);
  const [trainingQuantum, setTrainingQuantum] = useState(false);
  const [error, setError] = useState(null);
  const [epochs, setEpochs] = useState(25);
  const [layers, setLayers] = useState(3);
  const [lossHistory, setLossHistory] = useState(null);
  const [circuitDiagram, setCircuitDiagram] = useState(null);

  const toggleModel = (key) => {
    setSelectedModels((prev) => (prev.includes(key) ? prev.filter((m) => m !== key) : [...prev, key]));
  };

  const trainClassical = async () => {
    setError(null);
    setTrainingClassical(true);
    try {
      const res = await endpoints.trainClassical({ session_id: sessionId, models: selectedModels });
      setClassicalResults(res.data.results);
    } catch (e) {
      setError(e?.response?.data?.detail || "Classical training failed.");
    } finally {
      setTrainingClassical(false);
    }
  };

  const trainQuantum = async () => {
    setError(null);
    setTrainingQuantum(true);
    try {
      const res = await endpoints.trainQuantum({
        session_id: sessionId,
        epochs: Number(epochs),
        n_layers: Number(layers),
      });
      setQuantumResult(res.data.metrics);
      setLossHistory(res.data.loss_history.map((v, i) => ({ epoch: i + 1, loss: v })));
      setCircuitDiagram(res.data.circuit_diagram);
    } catch (e) {
      setError(e?.response?.data?.detail || "Quantum training failed.");
    } finally {
      setTrainingQuantum(false);
    }
  };

  if (!sessionId || !preprocessInfo) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold text-warmgray-900">Model Training</h1>
        <Card className="border-dashed">
          <CardContent className="p-10 text-center">
            <AlertTriangle className="h-8 w-8 text-amber-500 mx-auto mb-3" />
            <p className="font-medium text-warmgray-700">No preprocessed dataset found</p>
            <p className="text-sm text-warmgray-500 mt-1 mb-5">Upload and preprocess a dataset before training models.</p>
            <Link to="/upload">
              <Button>Go to Upload</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-bold text-warmgray-900">Model Training</h1>
          <p className="text-warmgray-500 mt-1">
            {preprocessInfo.n_train} training samples &middot; {preprocessInfo.n_features_selected} selected features
          </p>
        </div>
        {classicalResults && quantumResult && (
          <Link to="/comparison">
            <Button variant="accent">
              View Comparison <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-2 text-sm text-red-600 bg-red-50 border border-red-100 rounded-xl px-4 py-3">
          <AlertTriangle className="h-4 w-4 shrink-0" /> {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Classical training */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5 text-primary-600" /> Classical Baselines
            </CardTitle>
            <CardDescription>Random Forest, XGBoost & SVM trained on the full selected feature set.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2">
              {CLASSICAL_MODELS.map((m) => (
                <button
                  key={m.key}
                  onClick={() => toggleModel(m.key)}
                  className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                    selectedModels.includes(m.key)
                      ? "bg-primary-50 border-primary-200 text-primary-700"
                      : "bg-white border-border text-warmgray-400"
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>

            <Button className="w-full" onClick={trainClassical} disabled={trainingClassical || selectedModels.length === 0}>
              {trainingClassical ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Training models...
                </>
              ) : (
                <>
                  <PlayCircle className="h-4 w-4" /> Train Classical Models
                </>
              )}
            </Button>

            {trainingClassical && (
              <div className="space-y-2">
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
              </div>
            )}

            {classicalResults && !trainingClassical && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-2 pt-2">
                {classicalResults.map((r) => (
                  <div key={r.model_name} className="flex items-center justify-between rounded-xl bg-warmgray-50 px-4 py-3">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-primary-600" />
                      <span className="text-sm font-medium text-warmgray-800 capitalize">
                        {r.model_name.replace("_", " ")}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 text-xs">
                      <span className="text-warmgray-500">Acc</span>
                      <span className="font-semibold text-primary-700">{formatPercent(r.accuracy)}</span>
                      <span className="text-warmgray-500">F1</span>
                      <span className="font-semibold text-primary-700">{formatPercent(r.f1_score)}</span>
                    </div>
                  </div>
                ))}
              </motion.div>
            )}
          </CardContent>
        </Card>

        {/* Quantum training */}
        <Card className="relative overflow-hidden">
          <div className="absolute top-0 right-0 h-32 w-32 bg-accent-100/40 blur-3xl rounded-full" />
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Atom className="h-5 w-5 text-accent-500" /> Hybrid Variational Quantum Classifier
            </CardTitle>
            <CardDescription>PennyLane + PyTorch, simulated on default.qubit.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium text-warmgray-500 mb-1.5 block">Epochs</label>
                <Input type="number" min={5} max={100} value={epochs} onChange={(e) => setEpochs(e.target.value)} />
              </div>
              <div>
                <label className="text-xs font-medium text-warmgray-500 mb-1.5 block">Entangling Layers</label>
                <Input type="number" min={1} max={6} value={layers} onChange={(e) => setLayers(e.target.value)} />
              </div>
            </div>

            <Button variant="accent" className="w-full" onClick={trainQuantum} disabled={trainingQuantum}>
              {trainingQuantum ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Training on quantum simulator...
                </>
              ) : (
                <>
                  <Zap className="h-4 w-4" /> Train VQC Model
                </>
              )}
            </Button>

            {trainingQuantum && (
              <div className="space-y-2">
                <Skeleton className="h-32 w-full" />
              </div>
            )}

            {lossHistory && !trainingQuantum && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <p className="text-xs font-medium text-warmgray-500 mb-2">Training Loss Curve</p>
                <ResponsiveContainer width="100%" height={140}>
                  <LineChart data={lossHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E7E5E4" />
                    <XAxis dataKey="epoch" tick={{ fontSize: 10 }} stroke="#A8A29E" />
                    <YAxis tick={{ fontSize: 10 }} stroke="#A8A29E" />
                    <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #E7E5E4", fontSize: 12 }} />
                    <Line type="monotone" dataKey="loss" stroke="#8B5CF6" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </motion.div>
            )}

            {quantumResult && !trainingQuantum && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center justify-between rounded-xl bg-accent-50 px-4 py-3"
              >
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-accent-600" />
                  <span className="text-sm font-medium text-warmgray-800">VQC trained</span>
                </div>
                <div className="flex items-center gap-3 text-xs">
                  <span className="text-warmgray-500">Acc</span>
                  <span className="font-semibold text-accent-700">{formatPercent(quantumResult.accuracy)}</span>
                  <span className="text-warmgray-500">F1</span>
                  <span className="font-semibold text-accent-700">{formatPercent(quantumResult.f1_score)}</span>
                </div>
              </motion.div>
            )}

            {circuitDiagram && (
              <pre className="text-[10px] leading-snug bg-warmgray-900 text-emerald-300 rounded-xl p-4 overflow-x-auto font-mono whitespace-pre">
                {circuitDiagram}
              </pre>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
