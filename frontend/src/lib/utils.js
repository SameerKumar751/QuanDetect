import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export function formatPercent(value, digits = 1) {
  if (value === null || value === undefined) return "--";
  return `${(value * 100).toFixed(digits)}%`;
}

export function riskColor(level) {
  switch ((level || "").toLowerCase()) {
    case "low":
      return "text-risk-low bg-primary-50 border-primary-100";
    case "medium":
      return "text-risk-medium bg-amber-50 border-amber-100";
    case "high":
      return "text-risk-high bg-red-50 border-red-100";
    default:
      return "text-warmgray-500 bg-warmgray-50 border-border";
  }
}
