import axios from "axios";

// In development, Vite proxies /api to http://localhost:8000
const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

const client = axios.create({
  baseURL: `${API_BASE}/api/imaging`,
  timeout: 45000,
});

export const imagingApi = {
  getSampleScans: async () => {
    const res = await client.get("/samples");
    return res.data;
  },

  uploadScan: async (formData) => {
    const res = await client.post("/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },

  selectSample: async (sampleId) => {
    const formData = new FormData();
    formData.append("sample_id", sampleId);
    const res = await client.post("/upload", formData);
    return res.data;
  },

  predict: async (sessionId, modelType = "all") => {
    const res = await client.post("/predict", {
      session_id: sessionId,
      model_type: modelType,
    });
    return res.data;
  },

  getGradCam: async (sessionId, targetClass = null) => {
    const res = await client.post("/explain/gradcam", {
      session_id: sessionId,
      target_class: targetClass,
    });
    return res.data;
  },

  getBenchmarks: async () => {
    const res = await client.get("/compare");
    return res.data;
  },
};
