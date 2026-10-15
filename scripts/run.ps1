param([int]$Port = 9600, [string]$HostAddress = '127.0.0.1',
      [int]$Threads = 0, [int]$MaxWorkMb = 1024)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskExe = Join-Path $taskRoot 'build\Release\onnx-seg-server.exe'
if (-not (Test-Path -LiteralPath $taskExe)) { throw 'Run scripts\build.ps1 first.' }
& $taskExe --root $taskRoot --host $HostAddress --port $Port --threads $Threads --max-work-mb $MaxWorkMb
if ($LASTEXITCODE -ne 0) { throw "Server exited with code $LASTEXITCODE." }
