param(
  [string]$PbpRoot = "C:\Users\awolk\Documents\NFELO\Other Data\nflverse\pbp",
  [string]$DepthRoot = "C:\Users\awolk\Documents\NFELO\Other Data\nflverse\depth_charts",
  [string]$Years = "2020:2025",
  [int]$Seed = 42,
  [int]$BagsMake = 24,
  [int]$BagsAttempt = 14
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$Pipeline = Join-Path $RepoRoot "pipeline"
$UI = Join-Path $RepoRoot "ui"

Write-Host "Installing/validating Python deps..." -ForegroundColor Cyan
py -3 -m pip install --upgrade -r (Join-Path $RepoRoot "requirements.txt")

Write-Host "Section 1 → curate directly into UI (output-dir = UI)..." -ForegroundColor Cyan
py -3 (Join-Path $Pipeline "fg_pipeline_section1.py") `
  --pbp-root "$PbpRoot" `
  --depth-root "$DepthRoot" `
  --output-dir "$UI" `
  --years $Years

Write-Host "Section 2 → train & export directly into UI (output-dir = UI)..." -ForegroundColor Cyan
py -3 (Join-Path $Pipeline "fg_pipeline_section2.py") `
  --pbp-root "$PbpRoot" `
  --output-dir "$UI" `
  --seed $Seed `
  --bags-make $BagsMake `
  --bags-attempt $BagsAttempt

Write-Host "Done. JSONs written directly to $UI" -ForegroundColor Green
Get-ChildItem $UI *.json | Select-Object Name,Length
