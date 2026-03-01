"use client";

import { useEffect, useState } from "react";
import { checkBackendHealth } from "@/lib/api";

interface BackendStatusProps {
  onStatusChange?: (online: boolean) => void;
}

export default function BackendStatus({ onStatusChange }: BackendStatusProps) {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function check() {
      const isOnline = await checkBackendHealth();
      if (!cancelled) {
        setOnline(isOnline);
        onStatusChange?.(isOnline);
      }
    }

    check();
    const interval = setInterval(check, 30_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [onStatusChange]);

  if (online === null) {
    return (
      <div className="inline-flex items-center gap-1.5 text-xs text-gray-400 dark:text-slate-500">
        <span className="w-2 h-2 rounded-full bg-gray-300 dark:bg-slate-600 animate-pulse" />
        Checking backend…
      </div>
    );
  }

  return (
    <div
      className={`inline-flex items-center gap-1.5 text-xs ${
        online ? "text-green-600 dark:text-green-400" : "text-amber-600 dark:text-amber-400"
      }`}
    >
      <span
        className={`w-2 h-2 rounded-full ${
          online ? "bg-green-500" : "bg-amber-400"
        }`}
      />
      {online
        ? "Backend online — heatmaps enabled"
        : "Backend offline — heatmaps unavailable"}
    </div>
  );
}
