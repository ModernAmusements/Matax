#!/bin/bash
# Start script for Face Recognition API and Electron UI

echo "=========================================="
echo "Face Recognition System Startup"
echo "=========================================="

# Check for ArcFace flag
USE_ARCFACE=false
if [ "$1" = "--arcface" ] || [ "$1" = "-a" ]; then
    USE_ARCFACE=true
    echo "[0/4] ArcFace mode enabled"
fi

# Kill any existing API server
pkill -f "python api_server.py" 2>/dev/null
echo "[1/4] Stopped existing API server (if any)"

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
echo "[2/4] Cleared Python cache"

# Start API server in background
cd "$(dirname "$0")"
if [ "$USE_ARCFACE" = true ]; then
    echo "[3/4] Starting API server with ArcFace..."
    USE_ARCFACE=true python api_server.py &
else
    echo "[3/4] Starting API server with FaceNet..."
    python api_server.py &
fi
API_PID=$!

echo ""
echo "API Server: http://localhost:5000"
echo "Health Check: http://localhost:5000/api/health"
echo "Embedding Info: http://localhost:5000/api/embedding-info"
echo ""

# Start Electron UI
echo "[4/4] Starting Electron UI..."
cd "$(dirname "$0")/electron-ui"
npm start &
ELECTRON_PID=$!

echo ""
echo "=========================================="
echo "Both servers are running!"
echo ""
if [ "$USE_ARCFACE" = true ]; then
    echo "Mode: ArcFace (512-dim)"
else
    echo "Mode: FaceNet (128-dim)"
fi
echo ""
echo "To stop: Ctrl+C"
echo "=========================================="

# Wait for both servers
wait $API_PID $ELECTRON_PID
