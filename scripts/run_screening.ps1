# Band and channel screening: run every config of configs\diagnostics\screening for one seed.
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_screening.ps1 -Seed 1 [-Configs mrcp__all,env__all]
# Stops at the first failed run. A config that already has a run for this seed is skipped, so a stopped grid can resume.
param([int]$Seed = 1, [string[]]$Configs = @())

$ErrorActionPreference = "Stop"
$python = ".venv\Scripts\python"

if (-not $Configs) {
    $Configs = Get-ChildItem "configs\diagnostics\screening\*.yaml" | ForEach-Object { $_.BaseName }
}
foreach ($config in $Configs) {
    $done = Get-ChildItem "output\diagnostics\pre-design\screening\$config\*_seed$Seed\metrics.json" -ErrorAction SilentlyContinue
    if ($done) {
        Write-Host "$config seed $Seed already done, skipped"
        continue
    }
    & $python scripts\run_baseline.py --config "configs\diagnostics\screening\$config.yaml" --seed $Seed --tag "seed$Seed"
    if ($LASTEXITCODE -ne 0) { throw "$config seed $Seed failed" }
}
