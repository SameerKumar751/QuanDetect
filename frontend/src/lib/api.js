import axios from "axios";

// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.js).
// In production, set VITE_API_BASE_URL to the deployed backend URL.
const baseURL = import.meta.env.VITE_API_BASE_URL || "/api";

export const api = axios.create({ baseURL, timeout: 120000 });

export const endpoints = {
  uploadDataset: (file) => {
    const form = new FormData();
    if (file) form.append("file", file);
    return api.post("/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
  useSampleDataset: () => api.post("/upload"),
  sampleDatasetInfo: () => api.get("/sample-dataset-info"),
  preprocess: (payload) => api.post("/preprocess", payload),
  trainClassical: (payload) => api.post("/train/classical", payload),
  trainQuantum: (payload) => api.post("/train/quantum", payload),
  predict: (payload) => api.post("/predict", payload),
  compare: (sessionId) => api.get(`/compare/${sessionId}`),
  explain: (payload) => api.post("/explain", payload),
};

export default api;
