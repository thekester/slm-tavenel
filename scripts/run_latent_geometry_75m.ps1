param([Parameter(Mandatory = $true)][string]$ProjectRoot)

# CPU-only latent-geometry analysis of the 75M proxies (AdamW CE, distillation, Muon) on corpus v1.
$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = 'src'
$env:CUDA_VISIBLE_DEVICES = '-1'
$python = 'C:\Program Files\Python311\python.exe'
$out = 'runs/GEOMETRY'
New-Item -ItemType Directory -Force -Path $out | Out-Null
foreach ($run in @(
    @{ slug = '75m-adamw-ce'; ckpt = 'runs/EXP-0034a-75m-ce/checkpoints/step_00001500.pt' },
    @{ slug = '75m-adamw-distill'; ckpt = 'runs/EXP-0034a-75m-distill/checkpoints/step_00001500.pt' },
    @{ slug = '75m-muon'; ckpt = 'runs/EXP-0036a-75m-muon/checkpoints/step_00001500.pt' }
)) {
    if (Test-Path -LiteralPath "$out/$($run.slug).json") { continue }
    & $python scripts/latent_geometry.py --checkpoint $run.ckpt --data data/tokens/slm-tavenel-edu-corpus-v1-smollm2 `
        --out "$out/$($run.slug).json" *> "$out/$($run.slug).log"
}
"done $(Get-Date -Format o)" | Set-Content -LiteralPath "$out/75m-complete.txt" -Encoding utf8
