#!/bin/bash
# Start script for Face Recognition API and Electron UI
# ArcFace enabled by default

echo "=========================================="
echo "Face Recognition System Startup"
echo "=========================================="

# Kill any existing servers (aggressive)
echo "Stopping existing servers..."
pkill -9 -f "python.*api_server" 2>/dev/null
pkill -9 -f "electron" 2>/dev/null
sleep 1
lsof -ti :3000 | xargs -r kill -9 2>/dev/null
sleep 1
echo "✓ Existing servers stopped"

# Clear Python cache
echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
echo "✓ Cache cleared"

# Start API server with ArcFace
echo "Starting API server with ArcFace..."
cd "$(dirname "$0")"
source venv/bin/activate
USE_ARCFACE=true python api_server.py &
API_PID=$!

echo ""
echo "=========================================="
echo "Face Recognition System Running"
echo "=========================================="
echo ""
echo "API Server: http://localhost:3000"
echo "ArcFace Mode: ENABLED (512-dim)"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="

# Wait for API server
wait $API_PID
