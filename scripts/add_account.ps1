[CmdletBinding()]
param(
    [string]$ProjectId = "",
    [string]$GcsBucket = "",
    [string]$GcsConfigObject = "",
    [string]$AccountId = "",
    [string]$ImapHost = "",
    [string]$ImapUser = "",
    [string]$Email = "",
    [string]$FolderList = "default",
    [ValidateSet("imap", "gmail")]
    [string]$Provider = "imap",
    [string]$ClientId = ""
)

# Stores a new account's credentials in Secret Manager and prints the entry to
# add to the service's config in GCS. The service reads them at login, so the
# account needs no redeploy.
#   IMAP:  the password, as imap-<id>-password.
#   Gmail (-Provider gmail): opens the browser for the Google login and stores
#          the refresh token as gmail-<id>-refresh-token, and once for all Gmail
#          accounts the OAuth client secret as gmail-client-secret. Needs an
#          OAuth client of type "Desktop app" (Google Auth Platform > Clients,
#          app published "In production"). Run it again after revoking access
#          or changing the Google password.

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
$GcsBucket = Resolve-ConfigValue $GcsBucket "GCS_BUCKET" $DotEnvValues
$GcsConfigObject = Resolve-ConfigValue $GcsConfigObject "GCS_CONFIG_OBJECT" $DotEnvValues
if ([string]::IsNullOrWhiteSpace($GcsConfigObject)) {
    $GcsConfigObject = "config/smio.yaml"
}

if ([string]::IsNullOrWhiteSpace($ProjectId) -or
    [string]::IsNullOrWhiteSpace($GcsBucket)) {
    throw "Set GCP_PROJECT_ID and GCS_BUCKET in .env or pass -ProjectId and -GcsBucket."
}

function Assert-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

function Invoke-Gcloud {
    param([string[]]$Arguments)

    & gcloud @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud failed with exit code ${LASTEXITCODE}: gcloud $($Arguments -join ' ')"
    }
}

function Test-GcloudSecret {
    param([string]$Name)

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & gcloud secrets describe $Name "--project=$ProjectId" "--format=value(name)" 2>&1 | Out-Null
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
}

function Set-GcloudSecret {
    # Creates the secret, or adds a version: the service reads the latest one.
    param([string]$Name, [string]$Value)

    # Read from a file: piping the value to gcloud would add a newline to the secret.
    $valueFile = [System.IO.Path]::GetTempFileName()
    try {
        [System.IO.File]::WriteAllText($valueFile, $Value)
        if (Test-GcloudSecret $Name) {
            Invoke-Gcloud @(
                "secrets", "versions", "add", $Name,
                "--data-file=$valueFile",
                "--project=$ProjectId"
            ) | Out-Null
            Write-Host "Added a new version of secret '$Name'."
        }
        else {
            Invoke-Gcloud @(
                "secrets", "create", $Name,
                "--replication-policy=automatic",
                "--data-file=$valueFile",
                "--project=$ProjectId"
            ) | Out-Null
            Write-Host "Created secret '$Name'."
        }
    }
    finally {
        Remove-Item -LiteralPath $valueFile -Force -ErrorAction SilentlyContinue
    }
}

function Read-HiddenValue {
    param([string]$Prompt)

    $secureValue = Read-Host -Prompt $Prompt -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureValue)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

function Read-RequiredValue {
    param([string]$Value, [string]$Prompt)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        $Value = Read-Host -Prompt $Prompt
    }
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "A value for '$Prompt' is required."
    }
    return $Value.Trim()
}

function ConvertTo-YamlString {
    param([string]$Value)

    return "'" + $Value.Replace("'", "''") + "'"
}

function Get-GcloudSecretValue {
    param([string]$Name)

    $value = & gcloud secrets versions access latest "--secret=$Name" "--project=$ProjectId"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read secret '$Name'."
    }
    return ($value -join "`n")
}

function Get-GmailAddressKey {
    # Gmail ignores case and dots in the name, and googlemail.com is gmail.com.
    param([string]$Address)

    $name, $domain = $Address.Trim().ToLowerInvariant() -split '@', 2
    if ($domain -in @("gmail.com", "googlemail.com")) {
        return $name.Replace('.', '') + "@gmail.com"
    }
    return "$name@$domain"
}

function ConvertTo-Base64Url {
    param([byte[]]$Bytes)

    return [Convert]::ToBase64String($Bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
}

function New-RandomBase64Url {
    param([int]$ByteCount)

    $bytes = New-Object byte[] $ByteCount
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    return ConvertTo-Base64Url $bytes
}

function Get-GmailToken {
    # Google's OAuth flow for installed apps (with PKCE): the browser hands the
    # code to a one-time listener on 127.0.0.1, which is exchanged for tokens.
    param([string]$ClientId, [string]$ClientSecret, [string]$LoginHint)

    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    try {
        $redirectUri = "http://127.0.0.1:$($listener.LocalEndpoint.Port)"
        $verifier = New-RandomBase64Url 48
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        $challenge = ConvertTo-Base64Url ($sha256.ComputeHash([System.Text.Encoding]::ASCII.GetBytes($verifier)))
        $state = New-RandomBase64Url 16
        $query = [ordered]@{
            client_id             = $ClientId
            redirect_uri          = $redirectUri
            response_type         = "code"
            scope                 = "https://www.googleapis.com/auth/gmail.modify"
            code_challenge        = $challenge
            code_challenge_method = "S256"
            state                 = $state
            # A refresh token, also when this client was approved before.
            access_type           = "offline"
            prompt                = "consent"
            login_hint            = $LoginHint
        }
        $pairs = foreach ($item in $query.GetEnumerator()) {
            if (-not [string]::IsNullOrWhiteSpace($item.Value)) {
                "$($item.Key)=$([Uri]::EscapeDataString($item.Value))"
            }
        }
        $authUrl = "https://accounts.google.com/o/oauth2/v2/auth?" + ($pairs -join "&")
        Write-Host "Opening the browser for the Google login. Google warns that it has not"
        Write-Host "verified the app: choose 'Advanced', then 'Go to ... (unsafe)'. If no"
        Write-Host "browser opens, visit:"
        Write-Host $authUrl
        Start-Process $authUrl

        $answer = $null
        while ($null -eq $answer) {
            $connection = $listener.AcceptTcpClient()
            try {
                $stream = $connection.GetStream()
                $requestLine = [System.IO.StreamReader]::new($stream).ReadLine()
                $page = "SMIO"
                if ($requestLine -match '^GET /\?(\S+) HTTP') {
                    $answer = @{}
                    foreach ($pair in $Matches[1] -split '&') {
                        $name, $value = $pair -split '=', 2
                        $answer[$name] = [Uri]::UnescapeDataString(([string]$value).Replace('+', ' '))
                    }
                    $page = "SMIO: login received. You can close this window."
                }
                $writer = [System.IO.StreamWriter]::new($stream)
                $writer.Write("HTTP/1.1 200 OK`r`nContent-Type: text/plain; charset=utf-8`r`nConnection: close`r`n`r`n$page")
                $writer.Flush()
            }
            catch {
                # A browser's speculative connection that closed without a request.
            }
            finally {
                $connection.Close()
            }
        }
    }
    finally {
        $listener.Stop()
    }

    if ($answer["error"]) {
        throw "Google login failed: $($answer['error'])"
    }
    if ($answer["state"] -ne $state) {
        throw "The login answer does not belong to this request; run the script again."
    }
    $token = Invoke-RestMethod -Method Post -Uri "https://oauth2.googleapis.com/token" -Body @{
        code          = $answer["code"]
        client_id     = $ClientId
        client_secret = $ClientSecret
        redirect_uri  = $redirectUri
        grant_type    = "authorization_code"
        code_verifier = $verifier
    }
    if (-not $token.refresh_token) {
        throw "Google returned no refresh token."
    }
    return $token
}

Assert-Command "gcloud"

$AccountId = Read-RequiredValue $AccountId "Account id (lowercase letters, digits, hyphens; starts with a letter)"
# Part of the secret name, and the key of the account's stored mails. A leading
# letter keeps YAML from reading ids like 2, no or 2024-01-15 as other types.
if ($AccountId -cnotmatch '^[a-z][a-z0-9-]*$') {
    throw "Account id '$AccountId' must start with a lowercase letter and contain only lowercase letters, digits and hyphens."
}
$GcsUri = "gs://$GcsBucket/$GcsConfigObject"
$LocalCopy = Join-Path ([System.IO.Path]::GetTempPath()) "smio.yaml"
$YamlId = ConvertTo-YamlString $AccountId
$YamlFolderList = ConvertTo-YamlString $FolderList

if ($Provider -eq "imap") {
    $ImapHost = Read-RequiredValue $ImapHost "IMAP host"
    $ImapUser = Read-RequiredValue $ImapUser "IMAP username"
    $password = Read-HiddenValue "IMAP password"
    if ([string]::IsNullOrEmpty($password)) {
        throw "An IMAP password is required."
    }

    $SecretName = "imap-$AccountId-password"
    Set-GcloudSecret $SecretName $password

    if ([string]::IsNullOrWhiteSpace($Email)) {
        $Email = $ImapUser
    }
    $YamlHost = ConvertTo-YamlString $ImapHost
    $YamlUser = ConvertTo-YamlString $ImapUser
    $YamlEmail = ConvertTo-YamlString $Email
    $Entry = @"
  - id: $YamlId
    provider: imap
    email: $YamlEmail
    folder_list: $YamlFolderList
    mailbox:
      inbox: INBOX
      folder_root: INBOX
      trash: Papierkorb
    imap:
      host: $YamlHost
      username: $YamlUser
      password_secret: $SecretName
"@
    $EntryHint = "adjust the mailbox folder names to this server"
}
else {
    $ClientId = Read-RequiredValue (Resolve-ConfigValue $ClientId "GMAIL_CLIENT_ID" $DotEnvValues) "OAuth client ID (Google Auth Platform > Clients, type Desktop app)"
    $ClientSecretName = "gmail-client-secret"
    $storeClientSecret = -not (Test-GcloudSecret $ClientSecretName)
    if ($storeClientSecret) {
        $clientSecret = Read-HiddenValue "OAuth client secret of that client"
        if ([string]::IsNullOrEmpty($clientSecret)) {
            throw "The OAuth client secret is required."
        }
    }
    else {
        $clientSecret = Get-GcloudSecretValue $ClientSecretName
    }

    # The API the service and the check below call; nothing happens if it is on.
    Invoke-Gcloud @("services", "enable", "gmail.googleapis.com", "--project=$ProjectId")
    $token = Get-GmailToken $ClientId $clientSecret $Email
    $gmailProfile = Invoke-RestMethod -Uri "https://gmail.googleapis.com/gmail/v1/users/me/profile" `
        -Headers @{ Authorization = "Bearer $($token.access_token)" }
    if (-not [string]::IsNullOrWhiteSpace($Email) -and
        (Get-GmailAddressKey $gmailProfile.emailAddress) -ne (Get-GmailAddressKey $Email)) {
        throw "Logged in as $($gmailProfile.emailAddress), not $Email; run again and choose that account."
    }
    $Email = $gmailProfile.emailAddress
    Write-Host "Logged in as $Email."

    # Stored only after a working login, so a mistyped client secret is never kept.
    if ($storeClientSecret) {
        Set-GcloudSecret $ClientSecretName $clientSecret
    }
    $SecretName = "gmail-$AccountId-refresh-token"
    Set-GcloudSecret $SecretName $token.refresh_token

    $YamlEmail = ConvertTo-YamlString $Email
    $YamlClientId = ConvertTo-YamlString $ClientId
    $Entry = @"
  - id: $YamlId
    provider: gmail
    email: $YamlEmail
    folder_list: $YamlFolderList
    mailbox:
      inbox: INBOX
      folder_root: ''
      trash: TRASH
    gmail:
      client_id: $YamlClientId
      client_secret_secret: $ClientSecretName
      refresh_token_secret: $SecretName
"@
    $EntryHint = "the folder list's folders become Gmail labels"
}

Write-Host ""
Write-Host "Add this entry under 'accounts:' in the service's config ($EntryHint):"
Write-Host ""
Write-Host $Entry
Write-Host ""
Write-Host "Download the config, add the entry, and upload it again:"
Write-Host "  gcloud storage cp `"$GcsUri`" `"$LocalCopy`""
Write-Host "  gcloud storage cp `"$LocalCopy`" `"$GcsUri`""
Write-Host ""
Write-Host "The next workflow run picks it up. Never change the id afterwards: the"
Write-Host "account's stored mails are keyed by it."
