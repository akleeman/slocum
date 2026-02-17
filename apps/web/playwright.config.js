import { defineConfig } from "@playwright/test";
export default defineConfig({
    testDir: "./e2e",
    use: {
        baseURL: "http://127.0.0.1:5173"
    },
    webServer: [
        {
            command: "bash -lc 'cd ../.. && pixi run api-dev'",
            port: 8000,
            reuseExistingServer: true,
            timeout: 120000
        },
        {
            command: "npm run dev -- --host 127.0.0.1 --port 5173",
            port: 5173,
            reuseExistingServer: true,
            timeout: 120000
        }
    ]
});
