param([string]$CMake = '', [string]$Generator = '')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
if (-not $CMake) {
    $taskCommand = Get-Command cmake -ErrorAction SilentlyContinue
    if ($taskCommand) { $CMake = $taskCommand.Source }
    else {
        $taskProgramFiles = [Environment]::GetFolderPath('ProgramFilesX86')
        $taskVswhere = Join-Path $taskProgramFiles 'Microsoft Visual Studio\Installer\vswhere.exe'
        if (Test-Path -LiteralPath $taskVswhere) {
            $taskVs = & $taskVswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            if ($taskVs) { $CMake = Join-Path $taskVs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' }
        }
    }
}
if (-not $CMake -or -not (Test-Path -LiteralPath $CMake)) {
    throw 'Install Visual Studio C++ desktop development and CMake, or pass -CMake <path>.'
}
if (-not (Test-Path -LiteralPath (Join-Path $taskRoot 'deps\onnxruntime-win-x64-1.22.0\include\onnxruntime_cxx_api.h'))) {
    throw 'Run scripts\bootstrap.ps1 first.'
}
$taskArgs = @('-S', $taskRoot, '-B', (Join-Path $taskRoot 'build'), '-A', 'x64')
if ($Generator) { $taskArgs += @('-G', $Generator) }
& $CMake @taskArgs
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
& $CMake --build (Join-Path $taskRoot 'build') --config Release --parallel 2
if ($LASTEXITCODE -ne 0) { throw 'C++ build failed.' }
& $CMake --install (Join-Path $taskRoot 'build') --config Release --prefix $taskRoot
if ($LASTEXITCODE -ne 0) { throw 'EXE deployment failed. Close the running server before rebuilding.' }
Write-Output (Join-Path $taskRoot 'onnx-seg-server.exe')
