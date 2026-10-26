param([Parameter(Mandatory=$true)][string]$LaunchDirectory)
$ErrorActionPreference = 'Stop'

# Open live telemetry with delete sharing so the writer can atomically replace it.
# Close the handle before parsing JSON or inspecting other processes.
function Read-SharedJson([string]$Path) {
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read,
        [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete)
    $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8)
    try { $contents = $reader.ReadToEnd() } finally { $reader.Dispose() }
    return ConvertFrom-Json -InputObject $contents -AsHashtable -DateKind String
}

$launch = (Resolve-Path -LiteralPath $LaunchDirectory).Path
$dispatch = Read-SharedJson (Join-Path $launch 'process.json')
$campaign = Read-SharedJson (Join-Path $dispatch.output 'campaign.json')
$all = @(Get-CimInstance Win32_Process)
$checks = @()
$known = @($campaign.supervisor_processes)
foreach ($stage in @($campaign.stages)) { $known += @($stage.known_processes) }
foreach ($expected in $known) {
    $actual = $all | Where-Object ProcessId -eq $expected.pid
    $same = $actual -and ([DateTimeOffset]$actual.CreationDate) -eq ([DateTimeOffset]$expected.created_at_utc)
    $checks += [ordered]@{expected=$expected;present=[bool]$actual;same_instance=[bool]$same;
        full_identity_match=[bool]($same -and $actual.CommandLine -ceq $expected.command_line -and
            [int]$actual.ParentProcessId -eq $expected.parent_pid)}
}
$stages = @()
foreach ($stage in @($campaign.stages)) {
    $stageOutput = Join-Path $dispatch.output $stage.name
    $progressPath = Join-Path $stageOutput 'progress.json'
    $progress = if (Test-Path -LiteralPath $progressPath) { Read-SharedJson $progressPath } else { @{} }
    $cachePath = Join-Path $stageOutput 'cache-progress.json'
    $cache = if (Test-Path -LiteralPath $cachePath) { Read-SharedJson $cachePath } else { $progress.cache_scoring_progress }
    $cpu = if ($progress.pid) { Get-Process -Id $progress.pid -ErrorAction SilentlyContinue } else { $null }
    $stages += [ordered]@{name=$stage.name;status=$progress.status;process_id=$progress.pid;
        actual_optimizer_updates=$progress.actual_optimizer_updates;cache=$cache;
        training_started=$progress.training_started;model_device=$progress.model.device;
        migration_report=$progress.cache_prefix_recovery_report;process_guard=$progress.process_guard.last.reason;
        total_wait_seconds=$progress.process_guard.total_waited_seconds;error=$progress.error;
        monitoring_error=$stage.monitoring_error;exit_code=$stage.exit_code;
        cpu_seconds=$(if($cpu){$cpu.CPU}else{$null});stop_present=(Test-Path -LiteralPath (Join-Path $stageOutput 'STOP'))}
}
$record = [ordered]@{kind='alignlab-shared-reader-monitor-v1';checked_at_utc=[DateTime]::UtcNow.ToString('o');
    reader_file_share='ReadWrite|Delete';campaign_status=$campaign.status;plan=$dispatch.plan;output=$dispatch.output;
    started_at_epoch=$campaign.started_at_epoch;deadline_epoch=$campaign.deadline_epoch;
    original_budget_unchanged=($campaign.started_at_epoch -eq 1792634845.7777236 -and $campaign.deadline_epoch -eq 1792894045.7777236);
    process_checks=$checks;stages=$stages;campaign_stop_present=(Test-Path -LiteralPath (Join-Path $dispatch.output 'STOP'));
    gpu=@(& nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader);
    available_ram_gb=[Math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB,3);
    python_processes=@($all | Where-Object {$_.Name -match '^python(w)?\.exe$'} | ForEach-Object {
        [ordered]@{pid=[int]$_.ProcessId;parent_pid=[int]$_.ParentProcessId;
            created_at_utc=([DateTimeOffset]$_.CreationDate).UtcDateTime.ToString('o');command_line=$_.CommandLine}})}
$path = Join-Path $launch ('monitor-' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ') + '.json')
[IO.File]::WriteAllText($path, ($record | ConvertTo-Json -Depth 35), [Text.UTF8Encoding]::new($false))
[ordered]@{report=$path;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant();
    checked_at_utc=$record.checked_at_utc;campaign_status=$record.campaign_status;
    all_expected_processes_present_and_matching=($checks.Count -gt 0 -and @($checks | Where-Object { !$_.present -or !$_.full_identity_match }).Count -eq 0);
    live_owned_pids=@($checks | Where-Object full_identity_match | ForEach-Object {$_.expected.pid});
    original_budget_unchanged=$record.original_budget_unchanged;stages=$stages;gpu=$record.gpu;
    available_ram_gb=$record.available_ram_gb} | ConvertTo-Json -Depth 12
