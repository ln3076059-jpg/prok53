@echo off
title Roadwatch V3 Master Pipeline
cd /d "%~dp0"
echo ======================================================================
echo Starting Roadwatch V3 Autonomous Master Pipeline
echo ======================================================================
py -3.11 training/dmd/run_v3_master_pipeline.py
echo ======================================================================
echo Execution complete. Press any key to exit.
echo ======================================================================
pause
