param([string]$CMake = '', [string]$Generator = '')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskInstalled = Join-Path $taskRoot '.openvino-test-deps'
$taskOrt = Join-Path $taskInstalled 'onnxruntime'
$taskOv = Join-Path $taskInstalled 'openvino'
$taskOvNotices = Join-Path $taskInstalled 'openvino-2025.1.0.dist-info'
if (-not (Test-Path -LiteralPath (Join-Path $taskOrt 'capi\onnxruntime_providers_openvino.dll')) -or
    -not (Test-Path -LiteralPath (Join-Path $taskOv 'libs\openvino.dll'))) {
    # Official wheel archives also contain native Windows DLLs; Python is not needed.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $taskPackages = Get-Content -LiteralPath (Join-Path $taskRoot 'deps.lock.json') -Raw | ConvertFrom-Json |
        Where-Object { $_.name -in @('ONNX Runtime OpenVINO 1.22.0 Windows x64', 'OpenVINO 2025.1.0 Windows x64') }
    if (@($taskPackages).Count -ne 2) { throw 'Missing OpenVINO entries in deps.lock.json.' }
    foreach ($taskPackage in $taskPackages) {
        $taskArchive = Join-Path $taskRoot $taskPackage.path
        $taskExtract = Join-Path $taskRoot $taskPackage.extract
        if (-not (Test-Path -LiteralPath $taskExtract)) {
            New-Item -ItemType Directory -Path (Split-Path -Parent $taskArchive) -Force | Out-Null
            if (-not (Test-Path -LiteralPath $taskArchive)) {
                Invoke-WebRequest -Uri $taskPackage.url -OutFile $taskArchive
            }
            if ((Get-FileHash -LiteralPath $taskArchive -Algorithm SHA256).Hash -ne $taskPackage.sha256) {
                throw ('Dependency SHA256 mismatch: ' + $taskPackage.path)
            }
            [IO.Compression.ZipFile]::ExtractToDirectory($taskArchive, $taskExtract)
        }
    }
    $taskOrt = Join-Path $taskRoot 'deps\onnxruntime-openvino-win-x64-1.22.0\onnxruntime'
    $taskOv = Join-Path $taskRoot 'deps\openvino-win-x64-2025.1.0\openvino'
    $taskOvNotices = Join-Path $taskRoot 'deps\openvino-win-x64-2025.1.0\openvino-2025.1.0.dist-info'
}
& (Join-Path $PSScriptRoot 'build.ps1') -CMake $CMake -Generator $Generator
if ($LASTEXITCODE -ne 0) { throw 'C++ build failed.' }
$taskTarget = Join-Path $taskRoot 'openvino'
New-Item -ItemType Directory -Path $taskTarget -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $taskRoot 'build\Release\onnx-seg-server.exe') -Destination $taskTarget
foreach ($taskDllRoot in @($taskRoot, (Join-Path $taskOrt 'capi'), (Join-Path $taskOv 'libs'))) {
    Get-ChildItem -LiteralPath $taskDllRoot -Filter '*.dll' -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $taskTarget $_.Name) -Force
    }
}
$taskLicenseDir = Join-Path $taskRoot 'third_party\licenses\openvino'
New-Item -ItemType Directory -Path $taskLicenseDir -Force | Out-Null
foreach ($taskNotice in @('LICENSE', 'ThirdPartyNotices.txt')) {
    Copy-Item -LiteralPath (Join-Path $taskOrt $taskNotice) -Destination (Join-Path $taskLicenseDir ('onnxruntime-openvino-' + $taskNotice)) -Force
}
foreach ($taskNotice in @('LICENSE', 'onednn_third-party-programs.txt', 'onetbb_third-party-programs.txt', 'runtime-third-party-programs.txt')) {
    Copy-Item -LiteralPath (Join-Path $taskOvNotices $taskNotice) -Destination (Join-Path $taskLicenseDir ('openvino-' + $taskNotice)) -Force
}
Write-Output (Join-Path $taskRoot '启动_OpenVINO.cmd')
