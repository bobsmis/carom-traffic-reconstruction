param([string]$Repo='\\wsl.localhost\Ubuntu-24.04\home\hcr\workspaces\carom-rebuild', [string]$Output="$PSScriptRoot\release")
$ErrorActionPreference='Stop'
$Python='C:\Users\hcrrc\Desktop\大创\CAROM_CLIENT_BUILD\venv\Scripts\python.exe'
$Stage=Join-Path $Output 'stage'
New-Item -ItemType Directory -Force $Stage | Out-Null
Copy-Item -LiteralPath (Join-Path $Repo 'desktop\client.py') -Destination $Stage
New-Item -ItemType Directory -Force (Join-Path $Stage 'web') | Out-Null
Get-ChildItem -LiteralPath (Join-Path $Repo 'frontend\dist') | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $Stage 'web') -Recurse -Force }
Copy-Item -LiteralPath (Join-Path $Repo 'desktop\examples') -Destination $Stage -Recurse -Force
& $Python -m PyInstaller --noconfirm --add-binary "C:\Users\hcrrc\miniconda3\envs\carla\Library\bin\ffi.dll;." --onedir --windowed --name CAROM-Studio --distpath (Join-Path $Output 'dist') --workpath (Join-Path $Output 'build') --specpath $Stage --add-data "${Stage}\web;web" --add-data "${Stage}\examples;examples" (Join-Path $Stage 'client.py')
if($LASTEXITCODE -ne 0){throw 'PyInstaller build failed'}
Copy-Item -LiteralPath (Join-Path $Repo 'desktop\README.md') -Destination (Join-Path $Output 'dist\CAROM-Studio\使用说明.md') -Force
Compress-Archive -LiteralPath (Join-Path $Output 'dist\CAROM-Studio') -DestinationPath (Join-Path $Output 'CAROM-Studio-Windows-x64.zip') -Force

