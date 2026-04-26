param([string]$Repo = "c:/Users/nehem/OneDrive/Documents/GitHub/Final_Project_MPHY_6120")
$ErrorActionPreference = "Stop"
$outDir = Join-Path $Repo "outputs_full"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$logPath = Join-Path $outDir "run_full.log"
$statusPath = Join-Path $outDir "run_full.status.txt"
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class NativeMethods {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1
$ES_AWAYMODE_REQUIRED = [uint32]64
function Write-Status([string]$msg) {
  $ts = Get-Date -Format o
  "$ts`t$msg" | Out-File -FilePath $statusPath -Encoding utf8
}
$pythonExe = Join-Path $Repo ".venv/Scripts/python.exe"
$entry = Join-Path $Repo "brain_tumor_prediction_full.py"
Write-Status "RUNNING"
"`n==== Detached run started: $(Get-Date -Format o) ====" | Out-File -FilePath $logPath -Append -Encoding utf8
try {
  $state = [uint32]($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED)
  [void][NativeMethods]::SetThreadExecutionState($state)
  & $pythonExe $entry *>> $logPath
  Write-Status "COMPLETED"
  "==== Detached run finished: $(Get-Date -Format o) ====\n" | Out-File -FilePath $logPath -Append -Encoding utf8
}
catch {
  Write-Status ("FAILED: " + $_.Exception.Message)
  "==== Detached run failed: $(Get-Date -Format o) ====\n$($_ | Out-String)" | Out-File -FilePath $logPath -Append -Encoding utf8
}
finally {
  [void][NativeMethods]::SetThreadExecutionState($ES_CONTINUOUS)
}
