@echo off
echo Starting Fall Detection Dashboard...
echo Open http://localhost:8000 in your browser
echo Press Ctrl+C to stop.
echo.
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
