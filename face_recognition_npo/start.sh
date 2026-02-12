#!/bin/bash
# Best Practice: start.sh starts Flask API, Electron connects to existing server

echo "=========================================="
echo "Face Recognition System Startup"
echo "=========================================="

# Kill any existing servers (aggressive cleanup)
echo ""
echo "[1/3] Stopping existing servers..."
pkill -9 -f "python.*api_server" 2>/dev/null
pkill -9 -f "electron" 2>/dev/null
sleep 1
lsof -ti :3000 | xargs -r kill -9 2>/dev/null
sleep 1
echo "       ✓ Existing servers stopped"

# Clear Python cache
echo ""
echo "[2/3] Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
echo "       ✓ Cache cleared"

# Start Flask API server
echo ""
echo "[3/3] Starting Flask API server..."
cd "$(dirname "$0")"
source venv/bin/activate
python api_server.py &
API_PID=$!

echo ""
echo "Waiting for Flask API to be ready..."

# Wait for Flask to be ready (poll port 3000)
MAX_WAIT=30
WAIT_COUNT=0
while ! curl -s http://localhost:3000/api/health > /dev/null 2>&1; do
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        echo "ERROR: Flask API failed to start within ${MAX_WAIT}s"
        exit 1
    fi
    echo "       Waiting... ($WAIT_COUNT/${MAX_WAIT}s)"
done

echo ""
echo "       ✓ Flask API is ready!"

# Ask user how to open
echo ""
echo "=========================================="
echo "Choose how to open the application:"
echo "  [1] Electron Desktop App (recommended)"
echo "  [2] Browser"
echo "  [3] Both"
echo "=========================================="
echo ""
read -p "Enter choice [1]: " choice
echo ""

case "$choice" in
    2)
        echo "Opening in browser..."
        open http://localhost:3000
        ;;
    3)
        echo "Opening Electron Desktop App and browser..."
        open http://localhost:3000 &
        cd "$(dirname "$0")/electron-ui" && npm start &
        ;;
    *)
        echo "Starting Electron Desktop App..."
        cd "$(dirname "$0")/electron-ui" && npm start &
        ;;
esac

echo ""
echo "=========================================="
echo "Face Recognition System Running"
echo "=========================================="
echo ""
echo "Flask API: http://localhost:3000"
echo "ArcFace Mode: ENABLED (512-dim)"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo "Choose how to open the application:"
echo "  [1] Electron Desktop App (recommended)"
echo "  [2] Browser"
echo "  [3] Both"
echo "=========================================="
echo ""
read -p "Enter choice [1]: " choice
echo ""

case "$choice" in
    2)
        echo "Opening in browser..."
        open http://localhost:3000
        ;;
    3)
        echo "Opening Electron Desktop App and browser..."
        open http://localhost:3000 &
        cd "$(dirname "$0")/electron-ui" && npm start &
        ;;
    *)
        echo "Starting Electron Desktop App..."
        cd "$(dirname "$0")/electron-ui" && npm start &
        ;;
esac

echo ""
echo "=========================================="
echo "Face Recognition System Running"
echo "=========================================="
echo ""
echo "Flask API: http://localhost:3000"
echo "ArcFace Mode: ENABLED (512-dim)"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""
echo "Flask API: http://localhost:3000"
echo "ArcFace Mode: ENABLED (512-dim)"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
