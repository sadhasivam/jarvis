#!/bin/bash
# Production server startup script with nohup

PORT=${1:-9999}
LOG_FILE="logs/stockmind.log"
PID_FILE="logs/stockmind.pid"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "🚀 Starting StockMind server on port $PORT..."

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠️  Server already running with PID $OLD_PID"
        echo "   Run ./stop_server.sh to stop it first"
        exit 1
    else
        rm "$PID_FILE"
    fi
fi

# Activate virtual environment
source .venv/bin/activate

# Check if inventory exists
if ! python -c "import duckdb; conn = duckdb.connect('jarvis.db'); tables = [t[0] for t in conn.execute('SHOW TABLES').fetchall()]; exit(0 if 'inventory_snapshot' in tables else 1)" 2>/dev/null; then
    echo "⚠️  Inventory not found. Run: python generate_inventory.py"
    exit 1
fi

# Start FastAPI (uvicorn) in background with nohup
nohup uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info > "$LOG_FILE" 2>&1 &

# Save PID
echo $! > "$PID_FILE"

echo "✅ Server started successfully!"
echo "   PID: $(cat $PID_FILE)"
echo "   Port: $PORT"
echo "   URL: http://0.0.0.0:$PORT"
echo "   Logs: $LOG_FILE"
echo ""
echo "Commands:"
echo "   View logs: tail -f $LOG_FILE"
echo "   Stop server: ./stop_server.sh"
echo "   Check status: ./status_server.sh"
