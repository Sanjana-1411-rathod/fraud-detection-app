@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting FraudShield AI...
streamlit run app.py
pause
