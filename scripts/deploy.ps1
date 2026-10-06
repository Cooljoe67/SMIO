[CmdletBinding()]
param(
    [string]$ProjectId = "",
    [string]$Region = "",
    [string]$ServiceName = "smio",
    [int]$CommandTimeoutMinutes = 0,
    [string]$Repository = "smio",
    [string]$Cpu = "2",
    [string]$Memory = "8Gi",
    [int]$SummaryStartHour = 8,
    [string]$SummaryTimezone = "Europe/Berlin",
    [string]$GcsBucket = "",
    [string]$GcsModelPrefix = "models/distilbert_deployed",
    [string]$GcsNerModelPrefix = "",
    [string]$GcsDatabaseObject = "databases/smio.db",
    [string]$SchedulerLocation = "",
    [string]$FiveMinuteSchedulerJob = "smio-five-minute",
    [string]$DailySchedulerJob = "smio-daily",
    [string]$RuntimeServiceAccount = "",
    [string]$ImapHostSecret = "imap-host",
    [string]$ImapUserSecret = "imap-user",
    [string]$ImapPasswordSecret = "imap-password",
    [switch]$UpdateNerModel,
    [string]$NerModelDirectory = ".\models\ner-smio"
)

$ErrorActionPreference = "Stop"

$DotEnvValues = @{}
$DotEnvPath = Join-Path $PSScriptRoot "..\.env"
if (Test-Path $DotEnvPath) {
    foreach ($line in Get-Content $DotEnvPath) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
            $key = $Matches[1]
            $value = $Matches[2].Trim()
            if ($value.Length -ge 2 -and
                (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                    ($value.StartsWith("'") -and $value.EndsWith("'")))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            $DotEnvValues[$key] = $value
        }
    }
}

function Resolve-ConfigValue {
    param(
        [string]$ParameterValue,
        [string]$Name,
        [hashtable]$FileValues
    )

    if (-not [string]::IsNullOrWhiteSpace($ParameterValue)) {
        return $ParameterValue
    }
    $environmentValue = [Environment]::GetEnvironmentVariable($Name)
    if (-not [string]::IsNullOrWhiteSpace($environmentValue)) {
        return $environmentValue
    }
    if ($FileValues.ContainsKey($Name)) {
        return [string]$FileValues[$Name]
    }
    return ""
}

$ProjectId = Resolve-ConfigValue $ProjectId "GCP_PROJECT_ID" $DotEnvValues
$Region = Resolve-ConfigValue $Region "GCP_REGION" $DotEnvValues
$GcsBucket = Resolve-ConfigValue $GcsBucket "GCS_BUCKET" $DotEnvValues
$GcsNerModelPrefix = Resolve-ConfigValue $GcsNerModelPrefix "GCS_NER_MODEL_PREFIX" $DotEnvValues
$ConfiguredCommandTimeout = Resolve-ConfigValue "" "SMIO_COMMAND_TIMEOUT_MINUTES" $DotEnvValues
if ($CommandTimeoutMinutes -le 0) {
    if ([string]::IsNullOrWhiteSpace($ConfiguredCommandTimeout)) {
        $CommandTimeoutMinutes = 15
    }
    elseif (-not [int]::TryParse($ConfiguredCommandTimeout, [ref]$CommandTimeoutMinutes)) {
        throw "SMIO_COMMAND_TIMEOUT_MINUTES must be a positive integer."
    }
}
if ($CommandTimeoutMinutes -lt 1) {
    throw "CommandTimeoutMinutes must be a positive integer."
}
$SchedulerLocation = Resolve-ConfigValue $SchedulerLocation "GCP_SCHEDULER_LOCATION" $DotEnvValues
if ([string]::IsNullOrWhiteSpace($SchedulerLocation)) {
    $SchedulerLocation = $Region
}
if ([string]::IsNullOrWhiteSpace($GcsNerModelPrefix)) {
    $GcsNerModelPrefix = "models/ner-smio"
}

if ([string]::IsNullOrWhiteSpace($ProjectId) -or
    [string]::IsNullOrWhiteSpace($Region) -or
    [string]::IsNullOrWhiteSpace($GcsBucket)) {
    throw "Set GCP_PROJECT_ID, GCP_REGION, and GCS_BUCKET in .env or pass the matching script parameters."
}

function Invoke-Gcloud {
    param([string[]]$Arguments)

    & gcloud @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud failed with exit code ${LASTEXITCODE}: gcloud $($Arguments -join ' ')"
    }
}

function Pause-SchedulerJob {
    param(
        [string]$JobName,
        [System.Collections.Generic.List[string]]$PausedJobs
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $description = & gcloud scheduler jobs describe $JobName `
            "--location=$SchedulerLocation" "--project=$ProjectId" `
            "--format=value(state)" 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    $state = ($description | Out-String).Trim().ToUpperInvariant()
    if ($exitCode -ne 0) {
        if ($state -match "NOT_FOUND|matched no jobs|was not found") {
            Write-Host "Scheduler job '$JobName' not found; skipping it."
            return
        }
        throw "Could not inspect scheduler job '$JobName': $state"
    }

    if ($state -eq "PAUSED") {
        Write-Host "Scheduler job '$JobName' is already paused; it will remain paused."
        return
    }
    if ($state -ne "ENABLED") {
        throw "Scheduler job '$JobName' has unexpected state '$state'; deployment stopped."
    }

    Write-Host "Pausing scheduler job '$JobName'..."
    Invoke-Gcloud @("scheduler", "jobs", "pause", $JobName, "--location=$SchedulerLocation", "--project=$ProjectId")
    $null = $PausedJobs.Add($JobName)
}

function Resume-SchedulerJobs {
    param([System.Collections.Generic.List[string]]$PausedJobs)

    foreach ($jobName in $PausedJobs) {
        Write-Host "Resuming scheduler job '$jobName'..."
        try {
            Invoke-Gcloud @("scheduler", "jobs", "resume", $jobName, "--location=$SchedulerLocation", "--project=$ProjectId")
        }
        catch {
            Write-Warning "Could not resume '$jobName'. Run: gcloud scheduler jobs resume $jobName --location=$SchedulerLocation --project=$ProjectId"
        }
    }
}

function Backup-GcsDatabase {
    $sourceUri = "gs://$GcsBucket/$GcsDatabaseObject"
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $listing = & gcloud storage ls $sourceUri 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -ne 0) {
        $details = ($listing | Out-String).Trim()
        if ($details -match "matched no objects|No URLs matched") {
            Write-Host "No existing GCS database found; skipping the backup for first deployment."
            return
        }
        throw "Could not verify the GCS database before deployment: $details"
    }

    $timestamp = [DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss")
    $backupUri = "gs://$GcsBucket/backups/smio.db.before-deploy-$timestamp"
    Write-Host "Backing up GCS database to $backupUri..."
    Invoke-Gcloud @("storage", "cp", $sourceUri, $backupUri)
}

function Assert-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

function Wait-ForDocker {
    $startupTimeoutSeconds = 180
    $timer = [System.Diagnostics.Stopwatch]::StartNew()

    while ($true) {
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            docker info *> $null
            $dockerExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }

        if ($dockerExitCode -eq 0) {
            return
        }

        if ($timer.Elapsed.TotalSeconds -ge $startupTimeoutSeconds) {
            throw "Docker Desktop did not become ready within $startupTimeoutSeconds seconds. Start it manually and rerun deploy.ps1."
        }

        Write-Host "Waiting for Docker Desktop to finish starting..."
        Start-Sleep -Seconds 5
    }
}

function Start-DockerDesktop {
    if (Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue) {
        return
    }

    $desktopPath = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $desktopPath)) {
        $desktopPath = Join-Path $env:LOCALAPPDATA "Programs\Docker\Docker\Docker Desktop.exe"
    }
    if (-not (Test-Path $desktopPath)) {
        throw "Docker Desktop was not found in the standard install locations. Start it manually and rerun deploy.ps1."
    }

    Write-Host "Docker Desktop is not running. Starting it..."
    Start-Process -FilePath $desktopPath
}

Assert-Command "docker"
Assert-Command "gcloud"

$ActiveGcloudAccount = (& gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not check Google Cloud authentication status."
}
if ([string]::IsNullOrWhiteSpace($ActiveGcloudAccount)) {
    Write-Host "No active Google Cloud login found. Starting Google Cloud login..."
    Invoke-Gcloud @("auth", "login")
}
else {
    Write-Host "Google Cloud SDK is already authenticated. Skipping login."
}

Invoke-Gcloud @("config", "set", "project", $ProjectId)

Start-DockerDesktop
Wait-ForDocker

if ([string]::IsNullOrWhiteSpace($GcsBucket) -or $GcsBucket -eq "YOUR_BUCKET") {
    throw "A real GCS bucket is required. Current value: '$GcsBucket'"
}

$Image = "$Region-docker.pkg.dev/$ProjectId/$Repository/$ServiceName`:latest"
$ArtifactRegistryHost = "$Region-docker.pkg.dev"

Write-Host "Project:    $ProjectId"
Write-Host "Region:     $Region"
Write-Host "Image:      $Image"
Write-Host "GCS bucket: $GcsBucket"

Invoke-Gcloud @("services", "enable", "artifactregistry.googleapis.com", "run.googleapis.com", "secretmanager.googleapis.com")

$repositoryCheckErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$repositoryOutput = & gcloud artifacts repositories describe $Repository "--location=$Region" "--project=$ProjectId" 2>&1
$repositoryExitCode = $LASTEXITCODE
$ErrorActionPreference = $repositoryCheckErrorAction

$repositoryExists = $false
$repositoryExists = ($repositoryExitCode -eq 0)

if (-not $repositoryExists) {
    Invoke-Gcloud @(
        "artifacts", "repositories", "create", $Repository,
        "--repository-format=docker", "--location=$Region", "--project=$ProjectId"
    )
}

Invoke-Gcloud @("auth", "configure-docker", $ArtifactRegistryHost, "--quiet")

Write-Host "Building Docker image..."
docker build --target cloud-run -t $Image .
if ($LASTEXITCODE -ne 0) {
    throw "Docker build failed with exit code $LASTEXITCODE"
}

Write-Host "Pushing Docker image..."
docker push $Image
if ($LASTEXITCODE -ne 0) {
    throw "Docker push failed with exit code $LASTEXITCODE"
}

if ([string]::IsNullOrWhiteSpace($RuntimeServiceAccount)) {
    $projectNumber = (& gcloud projects describe $ProjectId --format="value(projectNumber)").Trim()
    $RuntimeServiceAccount = "$projectNumber-compute@developer.gserviceaccount.com"
}

Write-Host "Granting runtime access to Secret Manager and GCS..."
Invoke-Gcloud @(
    "projects", "add-iam-policy-binding", $ProjectId,
    "--member=serviceAccount:$RuntimeServiceAccount",
    "--role=roles/secretmanager.secretAccessor",
    "--condition=None"
)
Invoke-Gcloud @(
    "projects", "add-iam-policy-binding", $ProjectId,
    "--member=serviceAccount:$RuntimeServiceAccount",
    "--role=roles/logging.viewer",
    "--condition=None"
)
Invoke-Gcloud @(
    "storage", "buckets", "add-iam-policy-binding", "gs://$GcsBucket",
    "--member=serviceAccount:$RuntimeServiceAccount",
    "--role=roles/storage.objectAdmin"
)

$secretBindings = @(
    "IMAP_HOST=$ImapHostSecret`:latest",
    "IMAP_USER=$ImapUserSecret`:latest",
    "IMAP_PASSWORD=$ImapPasswordSecret`:latest"
)

Write-Host "Deploying Cloud Run service..."
$NerModelRevision = ""
if ($UpdateNerModel) {
    $pythonExecutable = Join-Path $PSScriptRoot "..\smio\Scripts\python.exe"
    if (-not (Test-Path $pythonExecutable)) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCommand) {
            throw "Python was not found. Activate the project environment or install Python."
        }
        $pythonExecutable = $pythonCommand.Source
    }

    Write-Host "Uploading NER model release from '$NerModelDirectory'..."
    $revisionOutput = & $pythonExecutable (Join-Path $PSScriptRoot "update_ner_model.py") `
        --model-dir $NerModelDirectory `
        --bucket $GcsBucket `
        --prefix $GcsNerModelPrefix `
        --upload-only
    if ($LASTEXITCODE -ne 0) {
        throw "NER model upload failed with exit code $LASTEXITCODE"
    }
    $NerModelRevision = ($revisionOutput | Out-String).Trim()
    if ($NerModelRevision -notmatch '^\d{8}T\d{12}Z$') {
        throw "NER model upload returned an invalid revision identifier."
    }
}

$deployEnvVars = "GCS_BUCKET=$GcsBucket,GCS_CLASSIFIER_MODEL_PREFIX=$GcsModelPrefix,GCS_NER_MODEL_PREFIX=$GcsNerModelPrefix,GCS_DATABASE_OBJECT=$GcsDatabaseObject,SMIO_SUMMARY_START_HOUR=$SummaryStartHour,SMIO_SUMMARY_TIMEZONE=$SummaryTimezone,SMIO_COMMAND_TIMEOUT_MINUTES=$CommandTimeoutMinutes"
if ($NerModelRevision) {
    $deployEnvVars += ",SMIO_NER_MODEL_REVISION=$NerModelRevision"
}
$deployArguments = @(
    "run", "deploy", $ServiceName,
    "--image=$Image",
    "--project=$ProjectId",
    "--region=$Region",
    "--port=8080",
    "--no-allow-unauthenticated",
    "--max-instances=1",
    "--concurrency=1",
    "--timeout=3600",
    "--cpu=$Cpu",
    "--memory=$Memory",
    "--service-account=$RuntimeServiceAccount",
    "--set-env-vars=$deployEnvVars",
    "--set-secrets=$($secretBindings -join ',')"
)
$pausedSchedulerJobs = [System.Collections.Generic.List[string]]::new()
try {
    Pause-SchedulerJob -JobName $FiveMinuteSchedulerJob -PausedJobs $pausedSchedulerJobs
    Pause-SchedulerJob -JobName $DailySchedulerJob -PausedJobs $pausedSchedulerJobs

    if ($pausedSchedulerJobs.Count -gt 0) {
        Write-Host "Scheduled triggers are paused. Check that any in-flight workflow has finished before proceeding."
        Read-Host "Press Enter to continue with the database backup and deployment"
    }

    Backup-GcsDatabase
    Invoke-Gcloud $deployArguments
}
finally {
    Resume-SchedulerJobs -PausedJobs $pausedSchedulerJobs
}

Write-Host "Deployment completed successfully."
$ServiceUrl = (& gcloud run services describe $ServiceName --project=$ProjectId --region=$Region --format="value(status.url)").Trim()
Write-Host "Service URL: $ServiceUrl"
Write-Host "Smoke test: `$token = gcloud auth print-identity-token; Invoke-RestMethod '$ServiceUrl/' -Headers @{ Authorization = ``"Bearer `$token``" }"