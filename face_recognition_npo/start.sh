#!/bin/bash
# =============================================================================
# Face Recognition System - Rich Startup Script
# =============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color
BOLD='\033[1m'
DIM='\033[2m'

# Unicode symbols
CHECK="${GREEN}✓${NC}"
CROSS="${RED}✗${NC}"
ARROW="${CYAN}➜${NC}"
SPINNER="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
DOTS="..."

# Spinner animation
spin() {
    local pid=$1
    local delay=0.1
    local spinstr='\|/-'
    while kill -0 $pid 2>/dev/null; do
        local temp=${spinstr:0:1}
        spinstr=${spinstr:1}$temp
        printf "\r${YELLOW}  ${spinstr} ${1}${NC}"
        sleep $delay
    done
    printf "\r"
}

# Print step with status
print_step() {
    local num=$1
    local total=$2
    local message=$3
    local status=$4
    
    printf "${CYAN}[${num}/${total}]${NC} ${message} "
    if [ "$status" = "done" ]; then
        echo -e "${CHECK} ${GREEN}done${NC}"
    elif [ "$status" = "fail" ]; then
        echo -e "${CROSS} ${RED}failed${NC}"
    else
        echo ""
    fi
}

# Print header
print_header() {
    clear
    echo -e "${CYAN}"
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║     ███████╗ ██████╗██╗  ██╗ ██████╗     ██████╗ ███████╗███╗   ██╗ ║
║     ██╔════╝██╔════╝██║  ██║██╔═══██╗    ██╔══██╗██╔════╝████╗  ██║ ║
║     █████╗  ██║     ███████║██║   ██║    ██║  ██║█████╗  ██╔██╗ ██║ ║
║     ██╔══╝  ██║     ██╔══██║██║   ██║    ██║  ██║██╔══╝  ██║╚██╗██║ ║
║     ██║     ╚██████╗██║  ██║╚██████╔╝    ██████╔╝███████╗██║ ╚████║ ║
║     ╚═╝      ╚═════╝╚═╝  ╚═╝ ╚═════╝     ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ║
║                                                                       ║
║                      ═══ System Startup ═══                           ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Progress bar
progress_bar() {
    local current=$1
    local total=$2
    local width=40
    local percent=$((current * 100 / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))
    
    printf "${CYAN}Progress:${NC} ["
    for i in $(seq 1 $filled); do printf "${GREEN}▓${NC}"; done
    for i in $(seq 1 $empty); do printf "${DIM}░${NC}"; done
    printf "] ${percent}%%\r"
}

# Print info box
print_info() {
    local title=$1
    local content=$2
    local len=${#content}
    local width=$((len + 4))
    
    echo -e "${DIM}┌${NC}$(printf '─%.0s' $(seq 1 $width))${DIM}┐${NC}"
    echo -e "${DIM}│${NC} ${BOLD}${CYAN}${title}${NC}$(printf ' %.0s' $(seq 1 $((width - ${#title} - 1)))) ${DIM}│${NC}"
    echo -e "${DIM}│${NC} ${content}$(printf ' %.0s' $(seq 1 $((width - ${#content} - 1)))) ${DIM}│${NC}"
    echo -e "${DIM}└${NC}$(printf '─%.0s' $(seq 1 $width))${DIM}┘${NC}"
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

print_header

# Step 1: Kill existing servers
print_step "1" "4" "Stopping existing servers... " "running"
pkill -9 -f "python.*api_server" 2>/dev/null
pkill -9 -f "electron" 2>/dev/null
sleep 1
lsof -ti :3000 | xargs -r kill -9 2>/dev/null
sleep 1
print_step "1" "4" "Stopping existing servers... " "done"

# Step 2: Clear cache
print_step "2" "4" "Clearing Python cache... " "running"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
print_step "2" "4" "Clearing Python cache... " "done"

# Step 3: Start Flask API
print_step "3" "4" "Starting Flask API server... " "running"
cd "$(dirname "$0")"
source venv/bin/activate
python api_server.py > /tmp/api_server.log 2>&1 &
API_PID=$!

# Show spinner while waiting
WAIT_COUNT=0
MAX_WAIT=30
printf "${YELLOW}  Waiting for API${NC} "

while ! curl -s http://localhost:3000/api/health > /dev/null 2>&1; do
    sleep 0.5
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $WAIT_COUNT -ge $((MAX_WAIT * 2)) ]; then
        echo ""
        print_step "3" "4" "Starting Flask API server... " "fail"
        echo -e "${RED}ERROR: Flask API failed to start${NC}"
        echo -e "${RED}Check log: /tmp/api_server.log${NC}"
        exit 1
    fi
    # Animated dots
    case $((WAIT_COUNT % 4)) in
        0) printf "\r${YELLOW}  Waiting for API   ${DOTS}${NC}" ;;
        1) printf "\r${YELLOW}  Waiting for API  .${DOTS}${NC}" ;;
        2) printf "\r${YELLOW}  Waiting for API ..${DOTS}${NC}" ;;
        3) printf "\r${YELLOW}  Waiting for API...${DOTS}${NC}" ;;
    esac
done
printf "\r                                                                 \r"
print_step "3" "4" "Starting Flask API server... " "done"

# Get API info
API_INFO=$(curl -s http://localhost:3000/api/health 2>/dev/null)
echo -e "${GRAY}──────────────────────────────────────────────────────────────────${NC}"

# Step 4: Ready
print_step "4" "4" "System ready! " "done"

# Print info boxes
echo ""
echo -e "${GRAY}╭────────────────────────────────────────────────────────────────╮${NC}"
echo -e "${GRAY}│${NC}  ${BOLD}${GREEN}🚀 SYSTEM STARTUP COMPLETE${NC}                                    ${GRAY}│${NC}"
echo -e "${GRAY}├────────────────────────────────────────────────────────────────┤${NC}"
echo -e "${GRAY}│${NC}  ${WHITE}API URL:${NC}      ${CYAN}http://localhost:3000${NC}                          ${GRAY}│${NC}"
echo -e "${GRAY}│${NC}  ${WHITE}Mode:${NC}         ${GREEN}ArcFace (512-dim)${NC}                              ${GRAY}│${NC}"
echo -e "${GRAY}│${NC}  ${WHITE}Features:${NC}     ${YELLOW}Pose Detection • Multi-Ref • Preprocessing${NC}     ${GRAY}│${NC}"
echo -e "${GRAY}╰────────────────────────────────────────────────────────────────╯${NC}"

echo ""
echo -e "${BOLD}How would you like to open the application?${NC}"
echo ""
echo -e "  ${CYAN}[1]${NC} ${WHITE}Electron Desktop App${NC}  ${DIM}(recommended)${NC}"
echo -e "  ${CYAN}[2]${NC} ${WHITE}Browser${NC}"
echo -e "  ${CYAN}[3]${NC} ${WHITE}Both${NC}"
echo ""

read -p "  Enter choice ${CYAN}[1]${NC}: " choice
echo ""

case "$choice" in
    2)
        echo -e "${ARROW} ${WHITE}Opening in browser...${NC}"
        open http://localhost:3000
        ;;
    3)
        echo -e "${ARROW} ${WHITE}Opening Electron and browser...${NC}"
        open http://localhost:3000 &
        cd "$(dirname "$0")/electron-ui" && npm start &
        ;;
    *)
        echo -e "${ARROW} ${WHITE}Starting Electron Desktop App...${NC}"
        cd "$(dirname "$0")/electron-ui" && npm start &
        ;;
esac

echo ""
echo -e "${GRAY}════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${GREEN}✓${NC} ${WHITE}Face Recognition System is running!${NC}"
echo ""
echo -e "  ${DIM}API:${NC}        ${CYAN}http://localhost:3000${NC}"
echo -e "  ${DIM}ArcFace:${NC}    ${GREEN}ENABLED${NC} (512-dim)"
echo -e "  ${DIM}Preprocessing:${NC} ${YELLOW}AUTO${NC} (quality-based)"
echo -e "  ${DIM}Pose-Aware:${NC}  ${YELLOW}ENABLED${NC}"
echo ""
echo -e "  ${RED}Press Ctrl+C to stop${NC}"
echo ""
echo -e "${GRAY}════════════════════════════════════════════════════════════════════${NC}"
