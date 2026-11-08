# Electrode reduction curve: run every config of configs\diagnostics\electrode_reduction for one seed, in order.
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_electrode_reduction.ps1 -Seed 1 [-Configs ica__cz__car__direct,ica__cz__car__residual]
# Stops at the first failed run. A config that already has a run for this seed is skipped, so a stopped grid can resume.
param([int]$Seed = 1, [string[]]$Configs = @())

$ErrorActionPreference = "Stop"
$python = ".venv\Scripts\python"

$Configs = $Configs -split ","  # powershell -File passes a,b as one string
if (-not $Configs) {
    $Configs = Get-ChildItem "configs\diagnostics\electrode_reduction\*.yaml" | ForEach-Object { $_.BaseName }
}
foreach ($config in $Configs) {
    $done = Get-ChildItem "output\diagnostics\final-checks\electrode-reduction\$config\*_seed$Seed\metrics.json" -ErrorAction SilentlyContinue
    if ($done) {
        Write-Host "$config seed $Seed already done, skipped"
        continue
    }
    & $python scripts\run_baseline.py --config "configs\diagnostics\electrode_reduction\$config.yaml" --seed $Seed --tag "seed$Seed"
    if ($LASTEXITCODE -ne 0) { throw "$config seed $Seed failed" }
}
