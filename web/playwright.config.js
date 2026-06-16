import { defineConfig } from "@playwright/test";

// End-to-end regression against a real running server (the tiny dev model, served with the
// built UI). Playwright starts the server itself; history is written to a throwaway DB so the
// test never touches the real .crucible/history.db. Run with: npm run e2e (after build + e2e:install).
const PORT = 5199;
const HISTORY_DB = "/tmp/crucible-e2e-history.db";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  use: { baseURL: `http://127.0.0.1:${PORT}`, trace: "on-first-retry" },
  webServer: {
    command: `rm -f ${HISTORY_DB}; uv run mlxd serve -c config/dev.yaml --no-open --port ${PORT}`,
    cwd: "..",
    url: `http://127.0.0.1:${PORT}/healthz`,
    timeout: 180_000, // first run loads the model (and may download it)
    reuseExistingServer: false,
    env: { CRUCIBLE_HISTORY_DB: HISTORY_DB },
  },
});
