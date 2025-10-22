param(
  [string]$PbpRoot = "C:\Users\awolk\Documents\NFELO\Other Data\nflverse\pbp",
  [string]$DepthRoot = "C:\Users\awolk\Documents\NFELO\Other Data\nflverse\depth_charts",
  [string]$Years = "2020:2025",
  [string]$OutDir = ".\artifacts",
  [string]$UiDir = ".\ui",
  [string]$StadiumsCsv = "C:\Users\awolk\Documents\NFELO\Other Data\nflverse\pbp\team_stadiums.csv",
  [int]$BagsMake = 24,
  [int]$BagsAttempt = 14,
  [int]$Seed = 42,
  [switch]$SkipCalibration
)

$py = "py -3"
$base = "pipeline"

function Run([string]$cmd) {
  Write-Host ">> $cmd"
  iex $cmd
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $cmd" }
}

Run "$py $base\B_ingest.py --pbp-root `"$PbpRoot`" --depth-root `"$DepthRoot`" --output-dir `"$OutDir`" --ui-dir `"$UiDir`" --years $Years"
Run "$py $base\C_features.py --output-dir `"$OutDir`" --assets-dir .\assets --stadiums-csv `"$StadiumsCsv`""
Run "$py $base\D_make_physics_gam.py --output-dir `"$OutDir`" --log-level INFO"
Run "$py $base\E_make_gbm_residual.py --output-dir `"$OutDir`" --ui-dir `"$UiDir`" --bags $BagsMake --seed $Seed --log-level INFO"
Run "$py $base\F_kicker_hier_bayes.py --output-dir `"$OutDir`" --ui-dir `"$UiDir`""
Run "$py $base\G_attempt_model.py --output-dir `"$OutDir`" --ui-dir `"$UiDir`" --bags $BagsAttempt --seed $Seed"
Run "$py $base\H_coupling_and_wp.py --output-dir `"$OutDir`" --ui-dir `"$UiDir`""

if (-not $SkipCalibration) {
  Run "$py $base\I_calibration_eval.py --output-dir `"$OutDir`""
}

Run "$py $base\J_export_grids.py --output-dir `"$OutDir`" --ui-dir `"$UiDir`" --rebuild all"

Write-Host "All done ✅"
