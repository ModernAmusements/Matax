#!/bin/bash
# Start script for NGO Facial Image Analysis System

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  NGO Facial Image Analysis System"
echo "=============================================="
echo ""

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Darwin*)     PYENV="venv/bin/activate" ;;
    Linux*)      PYENV="venv/bin/activate" ;;
    MINGW*|MSYS*|CYGWIN*) PYENV="Scripts/activate" ;;
    *)           PYENV="venv/bin/activate" ;;
esac

# Check if virtual environment exists
if [ ! -f "$PYENV" ]; then
    echo "❌ Error: Virtual environment not found at $PYENV"
    echo "   Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source "$PYENV"
echo "✅ Virtual environment activated"
echo ""

show_menu() {
    echo "Choose an option:"
    echo ""
    echo "  1) Start Flask API Server + Open Browser"
    echo "  2) Start Electron Desktop App"
    echo "  3) Start Tkinter GUI (Standalone)"
    echo "  4) Run End-to-End Tests"
    echo "  5) Show API Server Logs"
    echo "  6) Exit"
    echo ""
    read -p "Enter option [1-6]: " choice
}

start_flask_browser() {
    echo ""
    echo "🌐 Starting Flask API Server..."
    echo "   API will be available at: http://localhost:3000"
    echo ""
    echo "   Press Ctrl+C to stop the server"
    echo ""
    python api_server.py
}

start_electron() {
    echo ""
    echo "🖥️  Starting Electron Desktop App..."
    echo ""

    if [ ! -d "electron-ui/node_modules" ]; then
        echo "📦 Installing Electron dependencies..."
        cd electron-ui
        npm install
        cd ..
        echo ""
    fi

    cd electron-ui
    npm start
}

start_tkinter() {
    echo ""
    echo "🪟 Starting Tkinter GUI..."
    echo ""
    echo "   Press Ctrl+C to stop"
    echo ""
    python gui/facial_analysis_gui.py
}

run_tests() {
    echo ""
    echo "🧪 Running End-to-End Pipeline Tests..."
    echo ""
    python test_e2e_pipeline.py
}

show_logs() {
    echo ""
    echo "📋 API Server logs (press Ctrl+C to exit):"
    echo ""
    tail -f /dev/null
}

# Main
while true; do
    show_menu

    case "$choice" in
        1) start_flask_browser; break ;;
        2) start_electron; break ;;
        3) start_tkinter; break ;;
        4) run_tests; break ;;
        5) show_logs; break ;;
        6) echo ""; echo "Goodbye!"; exit 0 ;;
        *) echo "Invalid option. Please try again." ;;
    esac
done
