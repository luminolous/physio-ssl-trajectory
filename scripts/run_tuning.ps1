# Run the deep baseline tuning grid: seed 1 searches lr per fold, seeds 2 and 3 reuse its choices.
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_tuning.ps1 [-Models eegnet,cnn_lstm]
# Stops at the first failed run.
param([string[]]$Models = @("eegnet", "shallowconvnet", "cnn_lstm"))

$ErrorActionPreference = "Stop"
$python = ".venv\Scripts\python"

foreach ($model in $Models) {
    $config = "configs\diagnostics\tuning\$model.yaml"
    & $python scripts\run_baseline.py --config $config --seed 1 --tag seed1
    if ($LASTEXITCODE -ne 0) { throw "$model seed 1 failed" }

    $gridRun = Get-ChildItem "output\diagnostics\tuning\$model" -Directory -Filter "*_seed1" |
        Sort-Object Name | Select-Object -Last 1
    $selectionRun = "output/diagnostics/tuning/$model/$($gridRun.Name)"

    foreach ($seed in 2, 3) {
        & $python scripts\run_baseline.py --config $config --seed $seed --tag "seed$seed" --selection-run $selectionRun
        if ($LASTEXITCODE -ne 0) { throw "$model seed $seed failed" }
    }
}
