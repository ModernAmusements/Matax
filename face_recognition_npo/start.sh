#!/bin/bash

echo "Face Recognition System"
echo "======================="
echo ""

# Kill existing servers
echo "==> Stopping existing servers..."
pkill -9 -f "python.*api_server" 2>/dev/null
pkill -9 -f "electron" 2>/dev/null
lsof -ti :3000 | xargs -r kill -9 2>/dev/null
sleep 1
echo "==> Stopped"
echo ""

# Clear cache
echo "==> Clearing cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "==> Cache cleared"
echo ""

# Start Flask API
echo "==> Starting Flask API server..."
cd "$(dirname "$0")"
source venv/bin/activate
python api_server.py > /tmp/api_server.log 2>&1 &

# Simple spinner while waiting
SPINNER='|/-\'
WAIT_COUNT=0
MAX_WAIT=30

while ! curl -s http://localhost:3000/api/health > /dev/null 2>&1; do
    sleep 0.5
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $WAIT_COUNT -ge $((MAX_WAIT * 2)) ]; then
        echo "==> ERROR: Flask API failed to start"
        echo "    Check log: /tmp/api_server.log"
        exit 1
    fi
    printf "\r==> Waiting for API server... ${SPINNER:((WAIT_COUNT % 4)):1}"
done
printf "\r==> Waiting for API server... done\n"

echo "==> API ready at http://localhost:3000"
echo ""

# Get API info
API_INFO=$(curl -s http://localhost:3000/api/health 2>/dev/null)
echo "==> System started successfully"
echo ""

echo "How would you like to open the application?"
echo ""
echo "  [1] Electron Desktop App"
echo "  [2] Browser"
echo "  [3] Both"
echo ""

read -p "  Enter choice [1]: " choice
echo ""

case "$choice" in
    2)
        echo "==> Opening in browser..."
        open http://localhost:3000
        ;;
    3)
        echo "==> Opening Electron and browser..."
        open http://localhost:3000 &
        cd "$(dirname "$0")/electron-ui" && npm start &
        ;;
    *)
        echo "==> Starting Electron Desktop App..."
        cd "$(dirname "$0")/electron-ui" && npm start &
        ;;
esac

echo ""
echo "==> Face Recognition System is running"
echo ""
echo "    API:        http://localhost:3000"
echo "    ArcFace:    ENABLED (512-dim)"
echo "    Preprocessing: AUTO (quality-based)"
echo "    Pose-Aware: ENABLED"
echo ""
echo "    Press Ctrl+C to stop"
echo ""
