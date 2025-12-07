chcp 65001 > NUL
@echo off

pushd %~dp0
echo Running app.py...
.venv\Scripts\python scripts\app.py

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause