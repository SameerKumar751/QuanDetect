import React from "react";
import { cn } from "@/lib/utils";

export function Skeleton({ className, ...props }) {
  return (
    <div
      className={cn("animate-pulse rounded-lg bg-gradient-to-r from-primary-50 via-warmgray-100 to-primary-50 bg-[length:200%_100%]", className)}
      {...props}
    />
  );
}
