import React, { useEffect, useRef, useState } from "react";
import { NavLink } from "react-router-dom";
import { gsap } from "gsap";
import {
  LayoutDashboard,
  UploadCloud,
  Cpu,
  Stethoscope,
  BarChart3,
  ShieldAlert,
  Atom,
  Menu,
  X,
  ScanLine,
  Image,
} from "lucide-react";
import { cn } from "@/lib/utils";

const TABULAR_NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/upload", label: "Dataset Upload", icon: UploadCloud },
  { to: "/training", label: "Model Training", icon: Cpu },
  { to: "/prediction", label: "Prediction", icon: Stethoscope },
  { to: "/comparison", label: "Comparison", icon: BarChart3 },
  { to: "/results", label: "Risk Results", icon: ShieldAlert },
];

const IMAGING_NAV = [
  { to: "/imaging", label: "CT Lung Diagnostics", icon: ScanLine },
];

export default function Sidebar() {
  const navRef = useRef(null);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    if (!navRef.current) return;
    const items = navRef.current.querySelectorAll(".nav-item");
    gsap.fromTo(
      items,
      { opacity: 0, x: -16 },
      { opacity: 1, x: 0, duration: 0.4, stagger: 0.06, ease: "power2.out" }
    );
  }, []);

  return (
    <>
      {/* Mobile top bar */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-40 flex items-center justify-between bg-white/90 backdrop-blur-sm border-b border-border px-4 h-16">
        <div className="flex items-center gap-2">
          <Atom className="h-6 w-6 text-accent-500" />
          <span className="font-semibold text-warmgray-900">QuanDetect</span>
        </div>
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="p-2 rounded-lg hover:bg-primary-50"
          aria-label="Toggle navigation"
        >
          {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      <aside
        className={cn(
          "fixed lg:sticky top-0 left-0 z-30 h-screen w-72 shrink-0 bg-white border-r border-border flex flex-col transition-transform duration-300 lg:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="hidden lg:flex items-center gap-3 px-6 h-20 border-b border-border">
          <div className="relative">
            <Atom className="h-8 w-8 text-accent-500 animate-float" />
            <span className="absolute inset-0 blur-lg bg-accent-400/30 rounded-full" />
          </div>
          <div>
            <p className="font-bold text-lg text-warmgray-900 leading-none">QuanDetect</p>
            <p className="text-[11px] text-warmgray-400 tracking-wide mt-1">Quantum-Classical ML Platform</p>
          </div>
        </div>

        <nav ref={navRef} className="flex-1 overflow-y-auto scrollbar-thin px-4 py-6 space-y-5 mt-16 lg:mt-0">
          <div>
            <p className="px-4 text-[10px] font-bold uppercase tracking-wider text-warmgray-400 mb-2">
              Tabular Pipeline
            </p>
            <div className="space-y-1">
              {TABULAR_NAV.map(({ to, label, icon: Icon, end }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={end}
                  onClick={() => setMobileOpen(false)}
                  className={({ isActive }) =>
                    cn(
                      "nav-item group flex items-center gap-3 rounded-xl px-4 py-2.5 text-sm font-medium transition-all duration-200",
                      isActive
                        ? "bg-primary-50 text-primary-700 shadow-soft"
                        : "text-warmgray-500 hover:bg-warmgray-50 hover:text-warmgray-900"
                    )
                  }
                >
                  <Icon className="h-[18px] w-[18px] shrink-0" />
                  <span>{label}</span>
                </NavLink>
              ))}
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between px-4 mb-2">
              <p className="text-[10px] font-bold uppercase tracking-wider text-accent-600">
                Medical Imaging (CT)
              </p>
              <span className="text-[9px] bg-accent-100 text-accent-700 font-semibold px-1.5 py-0.5 rounded-full">
                New
              </span>
            </div>
            <div className="space-y-1">
              {IMAGING_NAV.map(({ to, label, icon: Icon }) => (
                <NavLink
                  key={to}
                  to={to}
                  onClick={() => setMobileOpen(false)}
                  className={({ isActive }) =>
                    cn(
                      "nav-item group flex items-center gap-3 rounded-xl px-4 py-2.5 text-sm font-medium transition-all duration-200",
                      isActive
                        ? "bg-accent-50 text-accent-700 shadow-soft font-semibold"
                        : "text-warmgray-500 hover:bg-warmgray-50 hover:text-warmgray-900"
                    )
                  }
                >
                  <Icon className="h-[18px] w-[18px] shrink-0 text-accent-500" />
                  <span>{label}</span>
                </NavLink>
              ))}
            </div>
          </div>
        </nav>

        <div className="p-4 border-t border-border">
          <div className="rounded-xl bg-gradient-to-br from-primary-50 to-accent-50 p-4">
            <p className="text-xs font-semibold text-primary-700">SIH26139</p>
            <p className="text-[11px] text-warmgray-500 mt-1 leading-relaxed">
              Hybrid quantum-classical models for early, explainable disease detection.
            </p>
          </div>
        </div>
      </aside>

      {mobileOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/20 z-20"
          onClick={() => setMobileOpen(false)}
        />
      )}
    </>
  );
}
