import React, { useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { gsap } from "gsap";
import { motion } from "framer-motion";
import {
  Atom,
  ArrowRight,
  Database,
  Cpu,
  Stethoscope,
  BarChart3,
  ShieldCheck,
  Sparkles,
  Network,
  FlaskConical,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const PIPELINE_STEPS = [
  {
    icon: Database,
    title: "1. Data Pre-processing",
    desc: "Automatic cleaning, imputation, scaling & ANOVA-based feature selection on any biomedical CSV.",
  },
  {
    icon: Network,
    title: "2. Hybrid Architecture",
    desc: "Classical PCA feature reduction feeds a genuine quantum variational circuit — the best of both worlds.",
  },
  {
    icon: Atom,
    title: "3. Quantum ML (VQC)",
    desc: "A Variational Quantum Classifier built with PennyLane + PyTorch, trained on the default.qubit simulator.",
  },
  {
    icon: Stethoscope,
    title: "4. Prediction & Decision Support",
    desc: "Disease probability scores with Low / Medium / High risk stratification and top contributing features.",
  },
];

const TECH_STACK = [
  "FastAPI", "PennyLane", "PyTorch", "scikit-learn", "XGBoost", "SHAP",
  "React", "Vite", "Tailwind CSS", "Recharts", "GSAP", "Framer Motion",
];

export default function Landing() {
  const heroRef = useRef(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.fromTo(
        ".hero-badge",
        { opacity: 0, y: -10 },
        { opacity: 1, y: 0, duration: 0.5, ease: "power2.out" }
      );
      gsap.fromTo(
        ".hero-title",
        { opacity: 0, y: 24 },
        { opacity: 1, y: 0, duration: 0.7, delay: 0.1, ease: "power3.out" }
      );
      gsap.fromTo(
        ".hero-sub",
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.7, delay: 0.25, ease: "power3.out" }
      );
      gsap.fromTo(
        ".hero-cta",
        { opacity: 0, y: 16 },
        { opacity: 1, y: 0, duration: 0.6, delay: 0.4, ease: "power3.out" }
      );
      gsap.to(".orb-1", { y: -18, x: 10, duration: 4, repeat: -1, yoyo: true, ease: "sine.inOut" });
      gsap.to(".orb-2", { y: 14, x: -12, duration: 5, repeat: -1, yoyo: true, ease: "sine.inOut" });
    }, heroRef);
    return () => ctx.revert();
  }, []);

  return (
    <div ref={heroRef} className="space-y-16">
      {/* Hero */}
      
      <section className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-primary-50 via-white to-accent-50 border border-border px-6 sm:px-12 py-16 sm:py-20">
        <div className="orb-1 absolute -top-10 -right-10 h-56 w-56 rounded-full bg-accent-200/30 blur-3xl" />
        <div className="orb-2 absolute bottom-0 left-0 h-64 w-64 rounded-full bg-primary-200/30 blur-3xl" />
        

        <div className="relative max-w-3xl">
          <div className="hero-badge inline-flex items-center gap-2 rounded-full bg-white border border-border px-4 py-1.5 shadow-soft mb-6">
            <Sparkles className="h-3.5 w-3.5 text-accent-500" />
            <span className="text-xs font-medium text-warmgray-600">Smart India Hackathon &middot; SIH26139</span>
          </div>

          <div className="mb-4 max-w-2xl rounded-lg border border-amber-200 bg-amber-50 px-3.5 py-2.5">
  <p className="text-xs leading-relaxed text-amber-800">
    <span className="font-semibold">Demo Note:</span>{" "}
    Some platform features may be unavailable due to backend deployment
    and server connectivity limitations in the current demo environment.
  </p>
</div>

          <h1 className="hero-title text-4xl sm:text-5xl font-extrabold tracking-tight text-warmgray-900 leading-[1.1]">
            Hybrid Quantum-Classical
            <br />
            <span className="text-gradient">Early Disease Detection</span>
          </h1>

          <p className="hero-sub mt-6 text-lg text-warmgray-500 leading-relaxed max-w-xl">
            QuanDetect fuses classical machine learning with a PennyLane-powered Variational
            Quantum Classifier to flag disease risk earlier and more transparently —
            benchmarked side-by-side, always explainable.
          </p>

          <div className="hero-cta mt-8 flex flex-wrap items-center gap-3">
            <Link to="/upload">
              <Button size="lg" className="group">
                Get Started
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
            </Link>
            <Link to="/comparison">
              <Button size="lg" variant="outline">
                View Benchmark Demo
              </Button>
            </Link>
          </div>

        </div>
      </section>

      {/* Pipeline */}
      <section>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-warmgray-900">How the Platform Works</h2>
          <Badge variant="accent">4-Stage Pipeline</Badge>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {PIPELINE_STEPS.map((step, i) => (
            <motion.div
              key={step.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
            >
              <Card className="h-full hover:-translate-y-1 hover:shadow-glow transition-all duration-300">
                <CardHeader>
                  <div className="h-11 w-11 rounded-xl bg-primary-50 flex items-center justify-center mb-2">
                    <step.icon className="h-5 w-5 text-primary-600" />
                  </div>
                  <CardTitle className="text-base">{step.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription>{step.desc}</CardDescription>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Feature highlights */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <Card className="lg:col-span-2 bg-gradient-to-br from-warmgray-900 to-primary-700 text-white border-none overflow-hidden relative">
          <div className="absolute inset-0 quantum-dots opacity-10" />
          <CardContent className="p-8 relative">
            <FlaskConical className="h-8 w-8 text-accent-300 mb-4" />
            <h3 className="text-xl font-bold mb-2">Classical vs. Quantum, side by side</h3>
            <p className="text-white/70 leading-relaxed max-w-lg">
              Train Random Forest, XGBoost and SVM baselines alongside a Hybrid Variational
              Quantum Classifier, then compare accuracy, sensitivity, specificity and F1-score
              on identical train/test splits — no guesswork.
            </p>
            <Link to="/training">
              <Button variant="accent" className="mt-6">
                Train Models <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-8">
            <ShieldCheck className="h-8 w-8 text-primary-600 mb-4" />
            <h3 className="text-lg font-bold text-warmgray-900 mb-2">Decision Support</h3>
            <p className="text-sm text-warmgray-500 leading-relaxed">
              Every prediction returns a probability score, a Low / Medium / High risk tier,
              and the top contributing features — with SHAP explainability underneath.
            </p>
          </CardContent>
        </Card>
      </section>

      {/* Tech stack strip */}
      <section>
        <h2 className="text-lg font-semibold text-warmgray-900 mb-4">Built With</h2>
        <div className="flex flex-wrap gap-2">
          {TECH_STACK.map((t) => (
            <span
              key={t}
              className="px-3 py-1.5 rounded-full text-xs font-medium bg-white border border-border text-warmgray-600 shadow-soft"
            >
              {t}
            </span>
          ))}
        </div>
      </section>

      {/* Dataset note */}
      <section>
        <Card className="border-dashed">
          <CardContent className="p-6 flex flex-col sm:flex-row sm:items-center gap-4 justify-between">
            <div className="flex items-center gap-3">
              <Cpu className="h-6 w-6 text-accent-500" />
              <div>
                <p className="font-semibold text-warmgray-900">Demo dataset included</p>
                <p className="text-sm text-warmgray-500">
                  Breast Cancer Wisconsin (Diagnostic) ships by default — swap in any tabular
                  biomedical CSV (hospital records, genomics panels) on the Upload page.
                </p>
              </div>
            </div>
            <Link to="/upload">
              <Button variant="subtle">Upload Dataset</Button>
            </Link>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
