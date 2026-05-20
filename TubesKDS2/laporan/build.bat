@echo off
echo === Building Cell Cycle Paper (IEEE format) ===
echo.

REM Run twice for references/labels
pdflatex -interaction=nonstopmode main.tex >nul 2>&1
pdflatex -interaction=nonstopmode main.tex

echo.
if exist main.pdf (
    echo SUCCESS: main.pdf generated
) else (
    echo FAILED: Check main.log for errors
)
echo.
pause
