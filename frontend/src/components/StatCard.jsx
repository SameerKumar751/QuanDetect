import React from "react";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export default function StatCard({ icon: Icon, label, value, sublabel, accent = "primary", delay = 0 }) {
  const accentClasses = {
    primary: "bg-primary-50 text-primary-600",
    accent: "bg-accent-50 text-accent-600",
    amber: "bg-amber-50 text-amber-600",
    red: "bg-red-50 text-red-600",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
    >
      <Card className="hover:shadow-glow transition-shadow duration-300">
        <CardContent className="p-5 flex items-center gap-4">
          {Icon && (
            <div className={cn("h-11 w-11 rounded-xl flex items-center justify-center shrink-0", accentClasses[accent])}>
              <Icon className="h-5 w-5" />
            </div>
          )}
          <div className="min-w-0">
            <p className="text-2xl font-bold text-warmgray-900 leading-tight truncate">{value}</p>
            <p className="text-xs text-warmgray-500 mt-0.5">{label}</p>
            {sublabel && <p className="text-[11px] text-warmgray-400 mt-0.5">{sublabel}</p>}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
