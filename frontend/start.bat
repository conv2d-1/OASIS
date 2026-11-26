@echo off
setlocal
cd /d "%~dp0"
echo OASIS frontend is running at http://127.0.0.1:4173/
echo Keep this window open while using the page.
python -m http.server 4173 --bind 127.0.0.1
pause
