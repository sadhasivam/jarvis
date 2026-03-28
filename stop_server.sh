#!/bin/bash
# Stop StockMind server

PID_FILE="logs/stockmind.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "⚠️  No PID file found. Server may not be running."
    echo "   Killing any running uvicorn processes..."
    pkill -f "uvicorn main:app"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ps -p $PID > /dev/null 2>&1; then
    echo "🛑 Stopping StockMind server (PID: $PID)..."
    kill $PID
    sleep 2

    # Force kill if still running
    if ps -p $PID > /dev/null 2>&1; then
        echo "   Force killing..."
        kill -9 $PID
    fi

    rm "$PID_FILE"
    echo "✅ Server stopped"
else
    echo "⚠️  Process $PID not found. Cleaning up..."
    rm "$PID_FILE"
fi
