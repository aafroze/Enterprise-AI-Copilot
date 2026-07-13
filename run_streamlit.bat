@echo off
REM Load environment variables from .env and launch Streamlit

REM Read .env file and set environment variables
for /f "tokens=*" %%A in (type .env ^| findstr /v "^#") do (
    for /f "tokens=1* delims==" %%B in ("%%A") do (
        set %%B=%%C
    )
)

REM Launch Streamlit
python -m streamlit run app.py
pause
