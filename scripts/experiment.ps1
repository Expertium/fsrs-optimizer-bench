[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('prepare-baseline', 'new')]
    [string]$Action,
    [ValidateRange(1, 20)]
    [int]$Experiment,
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path $Root).Path
$BaselineRevision = '0485116d4d04e06feb0713d7f6a3b154df540e11'
$BaselineDir = Join-Path $Root 'baseline'
$BaselineBinary = Join-Path $BaselineDir 'fsrs_rs_python.abi3.pyd'
$BaselineManifest = Join-Path $BaselineDir 'manifest.json'

function Get-Sha256([string]$Path) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

if ($Action -eq 'prepare-baseline') {
    if (Test-Path $BaselineBinary) { throw "Baseline already exists at $BaselineBinary; refusing to overwrite it." }
    if (Test-Path (Join-Path $Root 'baseline')) {
        $baselineContents = Get-ChildItem (Join-Path $Root 'baseline') -Force
        if ($baselineContents) { throw 'Baseline directory exists and is not empty.' }
    }

    $buildRoot = Join-Path $env:TEMP ("fsrs-baseline-" + [guid]::NewGuid().ToString('N'))
    try {
        git -C $Root worktree add --detach $buildRoot $BaselineRevision
        if ($LASTEXITCODE -ne 0) { throw 'Unable to create pinned baseline worktree.' }
        Push-Location $buildRoot
        try {
            $env:CARGO_TARGET_DIR = 'fsrs_rs_python/target/baseline'
            cargo +1.85 build --manifest-path fsrs_rs_python/Cargo.toml --release --features pyo3/extension-module,pyo3/abi3-py312
            if ($LASTEXITCODE -ne 0) { throw 'Pinned baseline build failed.' }
        }
        finally {
            Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue
            Pop-Location
        }
        $built = Join-Path $buildRoot 'fsrs_rs_python/target/baseline/release/fsrs_rs_python.dll'
        if (!(Test-Path $built)) { throw "Expected baseline extension not found at $built." }
        New-Item -ItemType Directory -Force $BaselineDir | Out-Null
        Copy-Item -LiteralPath $built -Destination $BaselineBinary
        $manifest = [ordered]@{
            source_commit = $BaselineRevision
            rust_toolchain = '1.85'
            python_abi = 'abi3-py312'
            sha256 = Get-Sha256 $BaselineBinary
        }
        $manifest | ConvertTo-Json | Set-Content -LiteralPath $BaselineManifest -Encoding utf8
        (Get-Item -LiteralPath $BaselineBinary).IsReadOnly = $true
        Write-Output "Protected baseline created: $BaselineBinary"
        Write-Output "SHA256: $($manifest.sha256)"
    }
    finally {
        git -C $Root worktree remove --force $buildRoot 2>$null
        git -C $Root worktree prune
    }
    exit 0
}

if ($null -eq $Experiment) { throw 'Specify -Experiment 1..20 when creating an experiment.' }
if (!(Test-Path $BaselineBinary) -or !(Test-Path $BaselineManifest)) {
    throw 'Protected baseline is missing. Run scripts/experiment.ps1 prepare-baseline first.'
}
$manifest = Get-Content -Raw $BaselineManifest | ConvertFrom-Json
$actualHash = Get-Sha256 $BaselineBinary
if ($actualHash -ne $manifest.sha256) { throw 'Protected baseline hash mismatch; do not use this artifact.' }

$WorktreeRoot = Join-Path (Split-Path -Parent $Root) 'fsrs-optimizer-bench-experiments'
$Worktree = Join-Path $WorktreeRoot ("run-{0:D2}" -f $Experiment)
if (Test-Path $Worktree) { throw "Experiment directory already exists: $Worktree. Use a fresh experiment number/directory." }
New-Item -ItemType Directory -Force (Split-Path -Parent $Worktree) | Out-Null
git -C $Root worktree add --detach $Worktree HEAD
if ($LASTEXITCODE -ne 0) { throw 'Unable to create isolated experiment worktree.' }

$ExperimentBaselineDir = Join-Path $Worktree 'baseline'
New-Item -ItemType Directory -Force $ExperimentBaselineDir | Out-Null
$ExperimentBaseline = Join-Path $ExperimentBaselineDir 'fsrs_rs_python.abi3.pyd'
Copy-Item -LiteralPath $BaselineBinary -Destination $ExperimentBaseline
$metadata = [ordered]@{
    experiment = $Experiment
    worktree_commit = (git -C $Worktree rev-parse HEAD).Trim()
    baseline_source_commit = $manifest.source_commit
    baseline_sha256 = $actualHash
    created_utc = [DateTime]::UtcNow.ToString('o')
}
$metadata | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Worktree 'baseline/manifest.json') -Encoding utf8
Write-Output "Isolated experiment worktree: $Worktree"
Write-Output "Protected baseline copy: $ExperimentBaseline"
Write-Output "Build candidate with CARGO_TARGET_DIR=fsrs_rs_python/target/experiment-$Experiment; never overwrite baseline/*.pyd."
