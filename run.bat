@echo off
echo Starting CRM Email AI...
echo.

:: Load environment variables from .env
if not exist .env (
    echo ERROR: .env file not found.
    echo Please copy .env.example to .env and fill in your API keys.
    echo.
    pause
    exit /b 1
)

for /f "tokens=1,2 delims==" %%i in (.env) do set %%i=%%j

:: Launch Streamlit
streamlit run app.py
