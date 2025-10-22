@echo off
setlocal
set PBP=C:\Users\awolk\Documents\NFELO\Other Data\nflverse\pbp
set DEPTH=C:\Users\awolk\Documents\NFELO\Other Data\nflverse\depth_charts
set YEARS=2020:2025
set OUT=.\artifacts
set UI=.\ui
set STAD=C:\Users\awolk\Documents\NFELO\Other Data\nflverse\pbp\team_stadiums.csv
set BMAKE=24
set BATT=14
set SEED=42

py -3 pipeline\B_ingest.py --pbp-root "%PBP%" --depth-root "%DEPTH%" --output-dir "%OUT%" --ui-dir "%UI%" --years %YEARS% || goto :err
py -3 pipeline\C_features.py --output-dir "%OUT%" --assets-dir ".\assets" --stadiums-csv "%STAD%" || goto :err
py -3 pipeline\D_make_physics_gam.py --output-dir "%OUT%" --log-level INFO || goto :err
py -3 pipeline\E_make_gbm_residual.py --output-dir "%OUT%" --ui-dir "%UI%" --bags %BMAKE% --seed %SEED% --log-level INFO || goto :err
py -3 pipeline\F_kicker_hier_bayes.py --output-dir "%OUT%" --ui-dir "%UI%" || goto :err
py -3 pipeline\G_attempt_model.py --output-dir "%OUT%" --ui-dir "%UI%" --bags %BATT% --seed %SEED% || goto :err
py -3 pipeline\H_coupling_and_wp.py --output-dir "%OUT%" --ui-dir "%UI%" || goto :err
py -3 pipeline\I_calibration_eval.py --output-dir "%OUT%" || goto :err
py -3 pipeline\J_export_grids.py --output-dir "%OUT%" --ui-dir "%UI%" --rebuild all || goto :err

echo All done ✅
exit /b 0
:err
echo FAILED with error %ERRORLEVEL%
exit /b %ERRORLEVEL%
