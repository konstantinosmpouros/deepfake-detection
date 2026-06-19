# Babysitter: run the project notebooks headlessly, one at a time, GPU-serialized.
# Default: the FULL pipeline 00 -> 13 then the eval notebooks. Use -Only to run a
# single notebook (by filename or pipeline name, partial match) and bypass the list.
#
# A run is launched as a DETACHED process (Start-Process -PassThru); the loop polls
# until it exits, then classifies success by EXIT CODE (nbconvert returns non-zero on
# any cell error) -> works for data/eval notebooks too, not just the Optuna ones.
# At the time deadline the loop breaks WITHOUT killing the in-progress run.
#
# Examples:
#   powershell -File _babysit_runs.ps1                       # full sequence, 18h window
#   powershell -File _babysit_runs.ps1 -Hours 48             # full sequence, 48h window
#   powershell -File _babysit_runs.ps1 -Only patch-ensemble  # just that one notebook
#   powershell -File _babysit_runs.ps1 -Only eval-robustness # just one eval
#   powershell -File _babysit_runs.ps1 -Only 07              # the 07_* notebook
#
# WARNING: do NOT start this while another babysitter is still running - two instances
# would launch notebooks concurrently and fight over the GPU.

param(
  [string]$Only  = "",     # run only notebooks whose filename or pipe matches (wildcard, case-insensitive); "" = full sequence
  [double]$Hours = 18      # time window in hours before the loop stops launching
)

$ErrorActionPreference = "Continue"
$py       = "C:\Program Files\Python312\python.exe"
$kernel   = "df312"                                   # Python312 / CUDA kernel
$nbDir    = $PSScriptRoot                             # ...\notebooks
$repo     = Split-Path $nbDir                         # repo root
$START    = Get-Date
$deadline = $START.AddHours($Hours)
$pollSec  = 60                                        # check the child every 60s...
$logEvery = 15                                        # ...but log "running" only every ~15 min
$log      = Join-Path $nbDir "_babysit.log"

Set-Location $repo

# Full sequence: data prep (00-03) -> 6 core pipelines + 4 extra archs (04-13) -> evals.
$plan = @(
  @{ nb = "00_data_collection.ipynb";         pipe = "data-collection" },
  @{ nb = "01_eda.ipynb";                      pipe = "eda" },
  @{ nb = "02_cleaning.ipynb";                 pipe = "cleaning" },
  @{ nb = "03_split_and_preprocessing.ipynb";  pipe = "split-preprocessing" },
  @{ nb = "04_cnn-scratch.ipynb";              pipe = "cnn-scratch" },
  @{ nb = "05_cnn-residual.ipynb";             pipe = "cnn-residual" },
  @{ nb = "06_cnn-finetune.ipynb";             pipe = "cnn-finetune" },
  @{ nb = "07_vit-lora.ipynb";                 pipe = "vit-lora" },
  @{ nb = "08_clip-probe.ipynb";               pipe = "clip-probe" },
  @{ nb = "09_two-stream.ipynb";               pipe = "two-stream" },
  @{ nb = "10_freqcross.ipynb";                pipe = "freqcross" },
  @{ nb = "11_srm-noise.ipynb";                pipe = "srm-noise" },
  @{ nb = "12_patch-ensemble.ipynb";           pipe = "patch-ensemble" },
  @{ nb = "13_dire-recon.ipynb";               pipe = "dire-recon" },
  @{ nb = "eval-comparison.ipynb";             pipe = "eval-comparison" },
  @{ nb = "eval-generalization.ipynb";         pipe = "eval-generalization" },
  @{ nb = "eval-robustness.ipynb";             pipe = "eval-robustness" },
  @{ nb = "eval-optuna.ipynb";                 pipe = "eval-optuna" },
  @{ nb = "eval-explainability.ipynb";         pipe = "eval-explainability" }
)

function Log($m) {
  $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Output $line
}

function FreshMetrics($pipe) {
  # Supplementary signal for the TRAINING notebooks: metrics.json written THIS session
  # (mtime > start) AND best_params.json present. Always false for data/eval notebooks
  # (they have no such files) - those rely on the exit code alone.
  $m  = Join-Path $repo "notebooks\artifacts\$pipe\metrics\metrics.json"
  $bp = Join-Path $repo "notebooks\artifacts\$pipe\metrics\best_params.json"
  if (-not (Test-Path $m)) { return $false }
  return (((Get-Item $m).LastWriteTime -gt $START) -and (Test-Path $bp))
}

# -Only filter: keep plan items whose filename or pipe name matches the pattern.
if ($Only -ne "") {
  $sel = @($plan | Where-Object { ($_.nb -like "*$Only*") -or ($_.pipe -like "*$Only*") })
  if ($sel.Count -eq 0) {
    Log ("no notebook matches -Only '{0}'. Available: {1}" -f $Only, ($plan.pipe -join ', '))
    exit 1
  }
  Log ("-Only '{0}' -> {1} notebook(s): {2}" -f $Only, $sel.Count, ($sel.pipe -join ', '))
  $plan = $sel
}

Log ("babysitter started | {0}h window -> {1} | poll {2}s | sequence: {3}" -f $Hours, $deadline, $pollSec, ($plan.pipe -join ' -> '))

$idx = 0                 # next plan item to launch
$child = $null
$current = $null
$ticks = 0               # poll ticks since the current run started (for throttled logging)
$ok = 0; $failed = 0

while ($true) {
  if ((Get-Date) -ge $deadline) { Log ("{0}h deadline reached - stopping (leaving any in-progress run alive)" -f $Hours); break }

  # Something running right now? Poll fast, log slow.
  if ($null -ne $child -and -not $child.HasExited) {
    $ticks++
    if ($ticks % $logEvery -eq 0) { Log ("running: {0} (pid {1})" -f $current, $child.Id) }
    Start-Sleep -Seconds $pollSec; continue
  }

  # A launched run just finished -> classify by EXIT CODE (universal success signal).
  if ($null -ne $child -and $child.HasExited) {
    $code = $child.ExitCode
    if ($code -eq 0) {
      $ok++
      $extra = if (FreshMetrics $current) { " (fresh metrics.json)" } else { "" }
      Log ("FINISHED ok: {0} (exit 0){1}" -f $current, $extra)
    } else {
      $failed++
      Log ("WARN: {0} exited with code {1} - see _run_{0}.err.log" -f $current, $code)
    }
    $child = $null; $current = $null; $ticks = 0
  }

  if ($idx -ge $plan.Count) { Log ("sequence complete - ok={0} failed={1}" -f $ok, $failed); break }

  # Launch the next notebook headlessly (executes in place, saving outputs into the .ipynb).
  $target = $plan[$idx]; $idx++
  $current = $target.pipe
  $nbpath = Join-Path $nbDir $target.nb
  $out = Join-Path $nbDir ("_run_{0}.log" -f $current)
  $err = Join-Path $nbDir ("_run_{0}.err.log" -f $current)
  Log ("LAUNCHING [{0}/{1}] {2}: {3}" -f $idx, $plan.Count, $current, $target.nb)
  $a = @("-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
         "--ExecutePreprocessor.kernel_name=$kernel", "--ExecutePreprocessor.timeout=-1",
         ('"{0}"' -f $nbpath))
  $child = Start-Process -FilePath $py -ArgumentList $a -PassThru -NoNewWindow `
             -RedirectStandardOutput $out -RedirectStandardError $err
  $ticks = 0
  Start-Sleep -Seconds $pollSec
}
Log "babysitter exiting"
