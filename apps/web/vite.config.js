import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
    plugins: [react()],
    server: {
        proxy: {
            "/meta": "http://127.0.0.1:8000",
            "/tiles": "http://127.0.0.1:8000",
            "/point": "http://127.0.0.1:8000",
            "/timeseries": "http://127.0.0.1:8000"
        }
    }
});
