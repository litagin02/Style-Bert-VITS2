chcp 65001 > NUL
@echo off

pushd %~dp0
set PYTHONPATH=%~dp0
echo Running gradio_tabs/train.py...
.venv\Scripts\python -m scripts.gradio_tabs.train

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause