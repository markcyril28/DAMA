# HOW TO RUN (PowerShell, from the DAMA repository)
#   Validate without training: .\local_train.ps1 -ValidateOnly
#   Policy recovery baseline: .\local_train.ps1
#   Enhanced stage after promotion:
#     .\local_train.ps1 -EnhancedStage -Resume models\checkpoints_policy_distillation_recovery_wd1e4\model_step_NNNNNN.pt
#   Other config, latest:     .\local_train.ps1 -Config config\training_config.yaml
#   Other config, fresh:      .\local_train.ps1 -Config config\training_config.yaml -FreshStart
#   One-process policy bypass:
#     powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\local_train.ps1 -ValidateOnly
#
# Run .\setup_conda.ps1 first if the Windows "dama" environment is not ready.
#
#Requires -Version 5.1

<#
.SYNOPSIS
Runs DAMA training natively on Windows with a CUDA-enabled Conda environment.

.DESCRIPTION
This launcher bypasses WSL and its GPU bridge. It performs read-only preflight
checks before starting training and does not install packages, compile Cython
extensions, edit configuration files, or modify Git state.

Training itself still writes the logs, replay data, and checkpoints configured
by DAMA.

.PARAMETER CondaEnvironment
Name of the Windows Conda environment. The default is "dama".

.PARAMETER Config
Training configuration path relative to the DAMA repository. The default
matches local_train.sh.

.PARAMETER Resume
Resume from a specific checkpoint. This cannot be combined with FreshStart.
The policy-distillation recovery config accepts only its preserved step-134000
baseline.

.PARAMETER FreshStart
Start without a checkpoint. Without this option or Resume, the launcher uses
the preserved step-134000 baseline for the policy-distillation recovery config
and the latest checkpoint for other configs. FreshStart is rejected for the
policy-distillation recovery experiment.

.PARAMETER EnhancedStage
Allow the canonical policy-distillation config to resume from a checkpoint
recorded as a promoted policy-only model. The trainer verifies the promotion
record, checkpoint hash, and frozen-suite fingerprint before training starts.

.PARAMETER TrainDuration
Optional duration such as 2d, 4h, 30m, or 1d12h.

.PARAMETER ValidateOnly
Run all environment, dependency, project-import, and CUDA checks without
starting training.

.EXAMPLE
.\local_train.ps1 -ValidateOnly

.EXAMPLE
.\local_train.ps1

.EXAMPLE
.\local_train.ps1 -Config config\training_config.yaml -FreshStart -TrainDuration 4h

.EXAMPLE
.\local_train.ps1 -Resume .\models\checkpoints_policy_distillation\model_step_134000.pt
#>

[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string] $CondaEnvironment = 'dama',

    [ValidateNotNullOrEmpty()]
    [string] $Config = 'config\training_config_policy_distillation.yaml',

    [string] $Resume = '',

    [switch] $FreshStart,

    [switch] $EnhancedStage,

    [ValidateSet(2, 3)]
    [int] $EnhancedInferenceDepth = 2,

    [ValidatePattern('^(?=.+)(?:\d+d)?(?:\d+h)?(?:\d+m)?$')]
    [string] $TrainDuration = '',

    [switch] $ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    throw 'This launcher must run in Windows PowerShell or PowerShell on Windows.'
}

if ($FreshStart -and -not [string]::IsNullOrWhiteSpace($Resume)) {
    throw 'Use either -FreshStart or -Resume, not both.'
}

$ProjectDirectory = [System.IO.Path]::GetFullPath($PSScriptRoot)
$ProjectPrefix = $ProjectDirectory.TrimEnd('\') + '\'
$ConfigCandidate = if ([System.IO.Path]::IsPathRooted($Config)) {
    $Config
} else {
    Join-Path $ProjectDirectory $Config
}
$ConfigPath = [System.IO.Path]::GetFullPath($ConfigCandidate)

if (-not $ConfigPath.StartsWith($ProjectPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "The training config must be inside the DAMA repository: $ProjectDirectory"
}
if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "Training config not found: $ConfigPath"
}

$PolicyRecoveryConfigPath = [System.IO.Path]::GetFullPath(
    (Join-Path $ProjectDirectory 'config\training_config_policy_distillation.yaml')
)
$PolicyRecoveryBaselinePath = [System.IO.Path]::GetFullPath(
    (Join-Path $ProjectDirectory 'models\checkpoints_policy_distillation\model_step_134000.pt')
)
$PolicyRecoveryBaselineSha256 = '7238CD80F2EF6DC9D8487D2579DE4BDF35AF4B85DCB2B3BD271659E795B14D27'
$PolicyRecoveryLegacyStatsPath = [System.IO.Path]::GetFullPath(
    (Join-Path $ProjectDirectory 'models\training_stats_policy_distillation.json')
)
$PolicyRecoveryStatsPath = [System.IO.Path]::GetFullPath(
    (Join-Path $ProjectDirectory 'models\training_stats_policy_distillation_recovery_wd1e4.json')
)
$IsPolicyRecovery = $ConfigPath.Equals(
    $PolicyRecoveryConfigPath,
    [System.StringComparison]::OrdinalIgnoreCase
)
if ($EnhancedStage -and -not $IsPolicyRecovery) {
    throw '-EnhancedStage is available only with training_config_policy_distillation.yaml.'
}

$ResumePath = ''
if (-not [string]::IsNullOrWhiteSpace($Resume)) {
    $ResumeCandidate = if ([System.IO.Path]::IsPathRooted($Resume)) {
        $Resume
    } else {
        Join-Path $ProjectDirectory $Resume
    }
    $ResumePath = [System.IO.Path]::GetFullPath($ResumeCandidate)
}

if ($IsPolicyRecovery) {
    if ($FreshStart) {
        throw 'FreshStart is disabled for the policy-distillation recovery experiment. Resume only from model_step_134000.pt.'
    }
    if (-not (Test-Path -LiteralPath $PolicyRecoveryBaselinePath -PathType Leaf)) {
        throw "Policy recovery baseline not found: $PolicyRecoveryBaselinePath"
    }

    if ($EnhancedStage) {
        if ([string]::IsNullOrWhiteSpace($ResumePath)) {
            throw '-EnhancedStage requires -Resume with the recorded promoted policy-only checkpoint.'
        }
        if (-not (Test-Path -LiteralPath $ResumePath -PathType Leaf)) {
            throw "Promoted policy checkpoint not found: $ResumePath"
        }
    } else {
        if (-not [string]::IsNullOrWhiteSpace($ResumePath) -and
            -not $ResumePath.Equals(
                $PolicyRecoveryBaselinePath,
                [System.StringComparison]::OrdinalIgnoreCase
            )) {
            throw "The policy-distillation recovery experiment must resume from: $PolicyRecoveryBaselinePath"
        }
        $ResumePath = $PolicyRecoveryBaselinePath
    }

    $ActualPolicyRecoverySha256 = (Get-FileHash -LiteralPath $PolicyRecoveryBaselinePath `
        -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($ActualPolicyRecoverySha256 -ne $PolicyRecoveryBaselineSha256) {
        throw (
            "Policy recovery baseline SHA-256 mismatch. Expected " +
            "$PolicyRecoveryBaselineSha256 but found $ActualPolicyRecoverySha256. " +
            'Training was not started.'
        )
    }
} elseif (-not [string]::IsNullOrWhiteSpace($ResumePath) -and
          -not (Test-Path -LiteralPath $ResumePath -PathType Leaf)) {
    throw "Checkpoint not found: $ResumePath"
}

$CondaExecutable = $null
if (-not [string]::IsNullOrWhiteSpace($env:CONDA_EXE) -and
    (Test-Path -LiteralPath $env:CONDA_EXE -PathType Leaf)) {
    $CondaExecutable = [System.IO.Path]::GetFullPath($env:CONDA_EXE)
} else {
    # An activated Conda PowerShell session defines a function named "conda".
    # Resolve only native commands so arguments such as "env list --json" are
    # passed to Conda instead of PowerShell parameter binding.
    $CondaCommand = Get-Command -Name conda.exe, conda.bat `
        -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $CondaCommand) {
        $CondaExecutable = $CondaCommand.Source
    }
}
if ([string]::IsNullOrWhiteSpace($CondaExecutable)) {
    throw 'A native Conda executable was not found. Open an Anaconda or Miniconda PowerShell prompt and try again.'
}

$LogDirectory = Join-Path $ProjectDirectory 'logs\local\console'
[void] [System.IO.Directory]::CreateDirectory($LogDirectory)
$LogTimestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogFile = Join-Path $LogDirectory "console_windows_$LogTimestamp.txt"

$Mutex = New-Object System.Threading.Mutex($false, 'Local\DAMA_Local_Train_Windows')
$MutexAcquired = $false
$TranscriptStarted = $false
$LocationPushed = $false

$EnvironmentNames = @(
    'PYTHONPATH',
    'PROCESS_TITLE',
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_MAX_THREADS'
)
$SavedEnvironment = @{}
foreach ($Name in $EnvironmentNames) {
    $SavedEnvironment[$Name] = [System.Environment]::GetEnvironmentVariable($Name, 'Process')
}

function Invoke-CondaCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    $SavedPreference = $ErrorActionPreference
    try {
        # Windows PowerShell can convert native stderr into PowerShell errors.
        # Let the native process exit code determine success instead.
        $ErrorActionPreference = 'Continue'
        & $script:CondaExecutable @Arguments
        $script:NativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $SavedPreference
    }
}

try {
    try {
        $MutexAcquired = $Mutex.WaitOne(0)
    } catch [System.Threading.AbandonedMutexException] {
        $MutexAcquired = $true
    }
    if (-not $MutexAcquired) {
        throw 'Another native Windows DAMA training launcher is already running.'
    }

    Start-Transcript -Path $LogFile -Append | Out-Null
    $TranscriptStarted = $true

    Write-Host '=== Filipino Dama - ML Training (Native Windows) ==='
    Write-Host "Project:   $ProjectDirectory"
    Write-Host "Config:    $ConfigPath"
    Write-Host "Conda env: $CondaEnvironment"
    Write-Host "Log:       $LogFile"
    if ($IsPolicyRecovery) {
        Write-Host "Recovery:  verified step-134000 baseline ($PolicyRecoveryBaselineSha256)"
    }
    Write-Host ''
    Write-Host 'Source safeguard: no package installation, source build, config edit, or Git mutation is performed.'
    Write-Host 'Expected training outputs may still be written according to the selected config.'
    Write-Host ''

    $GitCommand = Get-Command git -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $GitCommand) {
        $GitStatus = @(& $GitCommand.Source -C $ProjectDirectory status --porcelain 2>$null)
        if ($GitStatus.Count -gt 0) {
            Write-Warning 'The DAMA working tree already contains local changes. This launcher will leave them untouched.'
            Write-Host ''
        }
    }

    $PathSeparator = [System.IO.Path]::PathSeparator
    $SourceDirectory = Join-Path $ProjectDirectory 'src'
    $ExistingPythonPath = $SavedEnvironment['PYTHONPATH']
    if ([string]::IsNullOrWhiteSpace($ExistingPythonPath)) {
        $env:PYTHONPATH = $SourceDirectory
    } else {
        $env:PYTHONPATH = "$SourceDirectory$PathSeparator$ExistingPythonPath"
    }
    $env:PROCESS_TITLE = 'micro-trainer'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:NUMEXPR_MAX_THREADS = '1'

    Push-Location -LiteralPath $ProjectDirectory
    $LocationPushed = $true

    Write-Host 'Running Windows Python and CUDA preflight checks...'
    $PreflightCode = @'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"ERROR: PyTorch import failed: {exc}")

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: CUDA is unavailable in this Windows environment. "
        f"PyTorch={torch.__version__}, torch CUDA={torch.version.cuda}"
    )

import numpy
import psutil
import yaml
import dama.ai.ml.trainer

device = torch.cuda.current_device()
properties = torch.cuda.get_device_properties(device)
print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"GPU: {properties.name}")
print(f"VRAM: {properties.total_memory / (1024 ** 3):.1f} GB")
print("DAMA trainer import: OK")
'@
    # conda.bat can truncate literal multiline arguments on Windows. Encode the
    # preflight so Python receives it through one ASCII command-line argument.
    $PreflightPayload = [System.Convert]::ToBase64String(
        [System.Text.Encoding]::UTF8.GetBytes($PreflightCode)
    )
    $PreflightBootstrap = "import base64;exec(base64.b64decode('$PreflightPayload'))"

    $PreflightArguments = @(
        'run', '--no-capture-output', '-n', $CondaEnvironment,
        'python', '-W', 'ignore::FutureWarning', '-c', $PreflightBootstrap
    )
    Invoke-CondaCommand -Arguments $PreflightArguments
    if ($NativeExitCode -ne 0) {
        Write-Host ''
        Write-Host 'Preflight failed. No training was started.' -ForegroundColor Red
        Write-Host "Install a CUDA-enabled Windows PyTorch build in Conda environment '$CondaEnvironment'."
        Write-Host 'Official selector: https://pytorch.org/get-started/locally/'
        throw "Windows CUDA preflight exited with code $NativeExitCode."
    }

    Write-Host ''
    Write-Host 'Preflight passed.' -ForegroundColor Green
    if ($ValidateOnly) {
        Write-Host 'Validation-only mode requested. Training was not started.'
        return
    }

    if ($IsPolicyRecovery -and -not $EnhancedStage -and
        -not (Test-Path -LiteralPath $PolicyRecoveryStatsPath) -and
        (Test-Path -LiteralPath $PolicyRecoveryLegacyStatsPath -PathType Leaf)) {
        Copy-Item -LiteralPath $PolicyRecoveryLegacyStatsPath `
            -Destination $PolicyRecoveryStatsPath
        Write-Host (
            'Recovery stats seeded without modifying the legacy stats file: ' +
            $PolicyRecoveryStatsPath
        )
    }

    $TrainerArguments = @(
        'run', '--no-capture-output', '-n', $CondaEnvironment,
        'python', '-W', 'ignore::FutureWarning',
        '-m', 'dama.ai.ml.trainer',
        '--config', $ConfigPath
    )

    if ($IsPolicyRecovery) {
        $TrainerArguments += @('--resume', $ResumePath)
        if ($EnhancedStage) {
            $TrainerArguments += @(
                '--enhanced-stage',
                '--inference-depth',
                $EnhancedInferenceDepth.ToString([System.Globalization.CultureInfo]::InvariantCulture)
            )
            Write-Host "Resume policy: gated promoted policy checkpoint $ResumePath"
        } else {
            Write-Host "Resume policy: locked recovery baseline $PolicyRecoveryBaselinePath"
        }
    } elseif (-not [string]::IsNullOrWhiteSpace($ResumePath)) {
        $TrainerArguments += @('--resume', $ResumePath)
        Write-Host "Resume policy: checkpoint $ResumePath"
    } elseif ($FreshStart) {
        Write-Host 'Resume policy: fresh start'
    } else {
        $TrainerArguments += '--resume-latest'
        Write-Host 'Resume policy: latest checkpoint'
    }

    if (-not [string]::IsNullOrWhiteSpace($TrainDuration)) {
        $TrainerArguments += @('--train-duration', $TrainDuration)
        Write-Host "Training duration override: $TrainDuration"
    }

    Write-Host ''
    Write-Host 'CUDA verified. Starting native Windows training...'
    Invoke-CondaCommand -Arguments $TrainerArguments
    if ($NativeExitCode -ne 0) {
        throw "DAMA training exited with code $NativeExitCode. See log: $LogFile"
    }

    Write-Host 'DAMA training completed successfully.' -ForegroundColor Green
} finally {
    if ($LocationPushed) {
        Pop-Location
    }

    foreach ($Name in $EnvironmentNames) {
        $PreviousValue = $SavedEnvironment[$Name]
        if ($null -eq $PreviousValue) {
            Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
        } else {
            [System.Environment]::SetEnvironmentVariable($Name, $PreviousValue, 'Process')
        }
    }

    if ($TranscriptStarted) {
        Stop-Transcript | Out-Null
    }
    if ($MutexAcquired) {
        [void] $Mutex.ReleaseMutex()
    }
    if ($null -ne $Mutex) {
        $Mutex.Dispose()
    }
}
