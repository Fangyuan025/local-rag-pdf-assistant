# Hushdoc dev launcher (PowerShell).
#
# Starts the FastAPI backend on :8000 and the Vite frontend on :5173 in
# parallel. Vite proxies /api/* to the backend so the browser stays on
# localhost:5173. Ctrl+C stops both.
#
#   .\dev.ps1            # default: opens the browser when Vite is ready
#   .\dev.ps1 -NoOpen    # skip the auto-open

param([switch]$NoOpen)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition

$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "venv not found at $venvPython. Run: py -3.12 -m venv .venv && .venv\Scripts\pip install -r requirements.txt"
}
$webDir = Join-Path $root "web"
if (-not (Test-Path (Join-Path $webDir "node_modules"))) {
    Write-Host "[hushdoc] web/node_modules missing — running 'npm install'..." -ForegroundColor Yellow
    Push-Location $webDir; npm install; Pop-Location
}

Write-Host "[hushdoc] starting FastAPI backend on http://localhost:8000 ..." -ForegroundColor Cyan
$backend = Start-Process -FilePath $venvPython `
    -ArgumentList "-m","uvicorn","server.main:app","--port","8000" `
    -WorkingDirectory $root -PassThru -NoNewWindow

Write-Host "[hushdoc] starting Vite frontend on http://localhost:5173 ..." -ForegroundColor Cyan
$frontend = Start-Process -FilePath "npm" -ArgumentList "run","dev" `
    -WorkingDirectory $webDir -PassThru -NoNewWindow

if (-not $NoOpen) {
    Start-Sleep -Seconds 4
    Start-Process "http://localhost:5173/"
}

# Forward Ctrl+C to children, wait for either to exit.
try {
    Wait-Process -Id $backend.Id, $frontend.Id
} finally {
    foreach ($p in @($backend, $frontend)) {
        if ($p -and -not $p.HasExited) {
            Write-Host "[hushdoc] stopping pid $($p.Id)..." -ForegroundColor Yellow
            try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
        }
    }
}
