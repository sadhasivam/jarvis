#!/bin/bash
# Check StockMind server status

PID_FILE="logs/stockmind.pid"
LOG_FILE="logs/stockmind.log"

echo "📊 StockMind Server Status"
echo "="$(printf '=%.0s' {1..50})

if [ ! -f "$PID_FILE" ]; then
    echo "Status: ❌ Not running (no PID file)"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ps -p $PID > /dev/null 2>&1; then
    echo "Status: ✅ Running"
    echo "PID: $PID"

    # Get port from process
    PORT=$(lsof -Pan -p $PID -i 2>/dev/null | grep LISTEN | awk '{print $9}' | cut -d: -f2 | head -1)
    if [ ! -z "$PORT" ]; then
        echo "Port: $PORT"
        echo "URL: http://localhost:$PORT"
    fi

    # Memory usage
    MEM=$(ps -o rss= -p $PID | awk '{printf "%.1f MB", $1/1024}')
    echo "Memory: $MEM"

    # Uptime
    UPTIME=$(ps -o etime= -p $PID | sed 's/^ *//')
    echo "Uptime: $UPTIME"

    echo ""
    echo "Recent logs (last 10 lines):"
    echo "---"
    if [ -f "$LOG_FILE" ]; then
        tail -10 "$LOG_FILE"
    else
        echo "No log file found"
    fi
else
    echo "Status: ❌ Not running (PID $PID not found)"
    rm "$PID_FILE"
    exit 1
fi
