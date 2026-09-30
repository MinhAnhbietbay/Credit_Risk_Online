import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// strictPort: backend chỉ cho CORS từ đúng http://localhost:5173 — nếu port bận thì báo lỗi
// thay vì lặng lẽ chạy ở 5174 rồi mọi request bị chặn CORS.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
});
