# Download only official pinned packages; no global installation or PATH changes.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$taskRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$taskLock = Get-Content -LiteralPath (Join-Path $taskRoot 'deps.lock.json') -Raw | ConvertFrom-Json
foreach ($taskDependency in $taskLock) {
    $taskTarget = [IO.Path]::GetFullPath((Join-Path $taskRoot $taskDependency.path))
    if (-not $taskTarget.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar)) {
        throw 'Dependency path is outside project.'
    }
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($taskTarget)) | Out-Null
    if (-not (Test-Path -LiteralPath $taskTarget)) {
        Write-Output ("Downloading " + $taskDependency.name)
        Invoke-WebRequest -Uri $taskDependency.url -OutFile $taskTarget
    }
    if ((Get-FileHash -LiteralPath $taskTarget -Algorithm SHA256).Hash -ne $taskDependency.sha256) {
        throw ("SHA256 mismatch: " + $taskTarget + '. Remove this file and retry after verifying its source.')
    }
}
$taskOrt = Join-Path $taskRoot 'deps\onnxruntime-win-x64-1.22.0'
if (-not (Test-Path -LiteralPath (Join-Path $taskOrt 'include\onnxruntime_cxx_api.h'))) {
    Expand-Archive -LiteralPath (Join-Path $taskRoot 'deps\downloads\onnxruntime-win-x64-1.22.0.zip') -DestinationPath (Join-Path $taskRoot 'deps') -Force
}
$taskOpenCv = Join-Path $taskRoot 'deps\opencv'
if (-not (Test-Path -LiteralPath (Join-Path $taskOpenCv 'build\x64\vc16\lib\OpenCVConfig.cmake'))) {
    & (Join-Path $taskRoot 'deps\downloads\7zr.exe') x (Join-Path $taskRoot 'deps\downloads\opencv-4.12.0-windows.exe') ("-o" + (Join-Path $taskRoot 'deps')) -y
    if ($LASTEXITCODE -ne 0) { throw 'OpenCV archive extraction failed.' }
}
$taskLicenseDir = Join-Path $taskRoot 'third_party\licenses'
[IO.Directory]::CreateDirectory($taskLicenseDir) | Out-Null
Copy-Item -LiteralPath (Join-Path $taskOrt 'LICENSE') -Destination (Join-Path $taskLicenseDir 'onnxruntime-LICENSE.txt') -Force
Copy-Item -LiteralPath (Join-Path $taskOrt 'ThirdPartyNotices.txt') -Destination (Join-Path $taskLicenseDir 'onnxruntime-ThirdPartyNotices.txt') -Force
Copy-Item -LiteralPath (Join-Path $taskOpenCv 'sources\LICENSE') -Destination (Join-Path $taskLicenseDir 'opencv-LICENSE.txt') -Force
$taskThirdparty = Join-Path $taskOpenCv 'sources\3rdparty'
foreach ($taskNotice in Get-ChildItem -LiteralPath $taskThirdparty -Recurse -File) {
    if ($taskNotice.Name -notmatch '^(LICENSE(\..*)?|COPYING(\..*)?|COPYRIGHT(\..*)?|NOTICE(\..*)?)$') { continue }
    $taskRelative = $taskNotice.FullName.Substring($taskThirdparty.Length + 1)
    $taskNoticeTarget = [IO.Path]::GetFullPath((Join-Path (Join-Path $taskLicenseDir 'opencv-3rdparty') $taskRelative))
    if (-not $taskNoticeTarget.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar)) { throw 'Notice path is outside project.' }
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($taskNoticeTarget)) | Out-Null
    Copy-Item -LiteralPath $taskNotice.FullName -Destination $taskNoticeTarget -Force
}
Write-Output 'Dependencies verified and ready.'
