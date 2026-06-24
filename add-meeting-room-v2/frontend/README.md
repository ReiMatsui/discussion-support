# Experimental Viewer

This frontend is a replaceable experimental viewer for the initial Soniox room-mic runtime.

It is intentionally small and should not be treated as the production frontend architecture.

## Run

```bash
npm install
npm run dev
```

By default, it connects to:

```text
ws://localhost:8000/ws/viewer
```

Override with:

```bash
VITE_VIEWER_WS_URL=ws://localhost:8000/ws/viewer npm run dev
```
