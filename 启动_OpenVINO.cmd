@echo off
chcp 65001 >nul
"%~dp0openvino\onnx-seg-server.exe" --root "%~dp0." --backend openvino %*
if errorlevel 1 pause
