$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = 'C:\Users\hcrrc\miniconda3\envs\carla\python.exe'
$available = $false
try { $null = Invoke-RestMethod 'http://127.0.0.1:8766/sim-api/status'; $available = $true } catch { }
if (-not $available) {
    Start-Process -FilePath $python -ArgumentList 'control_server.py' -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput "$root\service.stdout.log" -RedirectStandardError "$root\service.stderr.log"
}
Start-Process 'http://127.0.0.1:8766/'
