@REM Simple bat to profile scripts: profile file.py

@echo off
python -m cProfile -o dump.prof %1
snakeviz dump.prof
del dump.prof