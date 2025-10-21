$UI = Join-Path (Resolve-Path "$PSScriptRoot\..").Path "ui"
Set-Location $UI
Write-Host "Serving UI from $UI at http://localhost:8080" -ForegroundColor Cyan
py -3 -m http.server 8080
