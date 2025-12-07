chcp 65001 > NUL
@echo off

pushd %~dp0
echo Running gradio_tabs/convert_onnx.py...
.venv\Scripts\python -m scripts.gradio_tabs.convert_onnx

if %errorlevel% neq 0 ( pause & popd & exit /b %errorlevel% )

popd
pause
