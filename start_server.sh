#!/bin/bash
# Start the Translation API server

echo "========================================"
echo "Translation API Server"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo ""
    echo "Installing requirements..."
    pip install -r requirements.txt
    echo ""
fi

echo ""
echo "Starting Translation API server on http://0.0.0.0:8005"
echo ""
echo "Note: The first run will download models (~3GB total):"
echo "  - NLLB-200 model (translation)"
echo "  - Qwen2.5-1.5B model (refinement, optional)"
echo ""
echo "This may take several minutes depending on your internet speed."
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "========================================"
echo ""

python main.py


