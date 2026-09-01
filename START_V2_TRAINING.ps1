[CmdletBinding()]
param(
    [ValidateSet("train", "smoke", "preflight")]
    [string]$Mode = "train",
    [ValidateSet("auto", "rtx5060ti_8gb", "rtx5060ti_16gb")]
    [string]$Profile = "auto",
    [string]$Working = "",
    [string]$BackupDirectory = "",
    [ValidateSet("phone_detector", "seatbelt_detector")]
    [string[]]$Only = @(),
    [switch]$InstallDependencies,
    [switch]$SkipSmoke,
    [switch]$AllowProposalTraining,
    [switch]$NoResume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$bundleRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
if (-not $Working) {
    $Working = Join-Path $bundleRoot "runs\v2_portable"
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    $pythonExecutable = $pyLauncher.Source
    $pythonPrefix = @("-3")
} else {
    $pythonLauncher = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonLauncher) {
        throw "Python 3 was not found. Install Python 3.11+ and run this script again."
    }
    $pythonExecutable = $pythonLauncher.Source
    $pythonPrefix = @()
}

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $pythonExecutable @pythonPrefix @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

Push-Location $bundleRoot
try {
    if ($InstallDependencies) {
        Invoke-CheckedPython -Arguments @("-m", "pip", "install", "-r", "requirements-windows-cu128.txt")
        Invoke-CheckedPython -Arguments @("-m", "pip", "install", "-r", "requirements-kaggle.txt")
    }

    $baseArguments = @(
        "-m", "training.v2_portable_runner",
        "--bundle", $bundleRoot,
        "--profile", $Profile
    )
    if ($AllowProposalTraining) {
        $baseArguments += @("--readiness-level", "proposal")
    } else {
        $baseArguments += @("--readiness-level", "governed")
    }
    if ($BackupDirectory) {
        $baseArguments += @("--backup-dir", $BackupDirectory)
    }
    $arguments = $baseArguments + @("--working", $Working)
    if ($Mode -eq "preflight") {
        $arguments += "--preflight-only"
    } elseif ($Mode -eq "smoke") {
        $arguments += "--smoke"
    }
    if ($Only.Count -gt 0) {
        $arguments += "--only"
        $arguments += $Only
    }
    if ($NoResume) {
        $arguments += "--no-resume"
    }
    if ($Mode -eq "train" -and -not $SkipSmoke) {
        $smokeWorking = "$Working-smoke"
        $smokeArguments = $baseArguments + @("--working", $smokeWorking, "--smoke")
        if ($Only.Count -gt 0) {
            $smokeArguments += "--only"
            $smokeArguments += $Only
        }
        if ($NoResume) {
            $smokeArguments += "--no-resume"
        }
        Invoke-CheckedPython -Arguments $smokeArguments
    }
    Invoke-CheckedPython -Arguments $arguments
} finally {
    Pop-Location
}
