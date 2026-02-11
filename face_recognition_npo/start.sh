#!/bin/bash
# Start script for Face Recognition API and Electron UI

echo "=========================================="
echo "Face Recognition System Startup"
echo "=========================================="

# Kill any existing API server
pkill -f "python api_server.py" 2>/dev/null
echo "[1/3] Stopped existing API server (if any)"

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
echo "[2/3] Cleared Python cache"

# Start API server in background
cd "$(dirname "$0")"
python api_server.py &
API_PID=$!

echo "[3/3] Started API server (PID: $API_PID)"
echo ""
echo "API Server: http://localhost:3000"
echo "Health Check: http://localhost:3000/api/health"
echo ""

# Start Electron UI
echo "Starting Electron UI..."
cd "$(dirname "$0")/electron-ui"
npm start &
ELECTRON_PID=$!

echo "Electron UI started (PID: $ELECTRON_PID)"
echo ""
echo "=========================================="
echo "Both servers are running!"
echo "Press Ctrl+C to stop"
echo "=========================================="

# Wait for both servers
wait $API_PID $ELECTRON_PID
