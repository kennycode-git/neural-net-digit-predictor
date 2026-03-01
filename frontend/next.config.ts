import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Empty turbopack config satisfies Next.js 16's Turbopack default
  // while keeping the webpack block for non-Turbopack builds
  turbopack: {},
  webpack: (config) => {
    // onnxruntime-web uses WASM; prevent node-specific modules from being bundled
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
    };
    return config;
  },
  async headers() {
    return [
      {
        // COOP + COEP are required for SharedArrayBuffer used by ort-web WASM threads
        source: "/(.*)",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
        ],
      },
    ];
  },
};

export default nextConfig;
