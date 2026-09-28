# SMIO – Smart Mail Inbox Organizer

An AI-powered system that automatically analyzes, categorizes, and declutters your email inbox.

SMIO connects to your mailbox via IMAP, classifies incoming mails with a fine-tuned DistilBERT model, extracts delivery-related details (tracking numbers, carriers, dates) with a custom NER model, moves mails into category folders according to configurable rules, sends you a daily plain-language summary, and periodically retrains the classifier on your manual corrections to improve its F1 score over time.

---

## 📌 Features

- **AI-based email classification**
  Classifies each mail into `delivery`, `commercial`, `social`, `tech`, or `other` using a fine-tuned DistilBERT model ([src/ai/classifier.py](src/ai/classifier.py)).

- **Named Entity Recognition (NER) for deliveries**
  For mails classified as `delivery`, extracts order IDs, tracking numbers, carrier, item name, and delivery status/date using a custom token-classification model plus regex/rule fallbacks ([src/ai/ner.py](src/ai/ner.py), [src/ai/email_metadata.py](src/ai/email_metadata.py)). Includes special-cased handling for Amazon sender addresses.

- **Automated inbox actions**
  - Moves classified mails into per-category IMAP folders. Read mail follows each folder's configured retention; unread mail is moved to `Papierkorb` after 20 days. Read delivery mail has no expiry by default — see [src/api/utils/folder_rules.py](src/api/utils/folder_rules.py).
  - Detects manual corrections: if you move a mail to a different folder yourself, SMIO records that as the ground-truth label (`true_label`) for future retraining.
  - Leaves SMIO-generated mail in `INBOX` without classification or folder routing.
  - `undo_last_processing` reverts the most recent processing batch and moves mails back to `INBOX`.

- **Persistent daily summary snapshots**
  Builds a logical-day digest from `08:00` to `08:00` in the configured timezone, stores the message and queryable metrics in the `summary_snapshots` table, and can send the stored snapshot by email. Human-readable dates use `DD/MM/YYYY`; tagged seed training/evaluation records are excluded from mailbox totals while remaining available for retraining. Retraining history includes unpromoted candidate metrics — see [src/ai/daily_summary.py](src/ai/daily_summary.py).

- **Email commands**
  Reply from the configured mailbox to a summary using its reply subject (for example, `AW: SMIO Daily Summary`) or use the fallback subject `SMIO Command`. Put `UNDO`, `RETRAIN`, `RELOAD MODEL`, or `SUMMARY` on the first non-empty body line; optionally prefix it with `SMIO:`. The five-minute workflow handles these commands and moves handled command messages to `Papierkorb`.

- **Self-improving classifier (replay-buffer retraining)**
  Once enough manual corrections accumulate (default: 20, configurable), SMIO fine-tunes the deployed model on a mix of the new corrections and a random sample of existing labeled data per class (90/10 replay ratio), evaluates the candidate against the currently deployed model on a fixed, continuously-growing hold-out set, and only promotes the new model if its macro-F1 is at least as good — see [src/ai/retrain.py](src/ai/retrain.py).

- **FastAPI backend**
  Endpoints for IMAP sync/fetch, inbox processing, and the daily summary (see below).

- **Sequential cron workflow**
  The workflow can run as one daily operation or as separate gather, send, and retrain jobs. Each scheduled operation is guarded by a process lock so overlapping work is skipped safely — see [src/scheduler/workflows.py](src/scheduler/workflows.py).

---

## 🧠 Core technologies

- Python 3.x
- HuggingFace Transformers (DistilBERT for classification, token classification for NER)
- PyTorch, scikit-learn (F1 evaluation)
- FastAPI + SQLAlchemy (SQLite, optionally synchronized to Google Cloud Storage)
- imap-tools (IMAP access)
- pydantic-settings (`.env` configuration)

---

## 📂 Project structure

```
SMIO/
  src/
    ai/
      classifier.py        # DistilBERT classification + reload_model()
      ner.py                # NER model + rule-based extraction
      email_metadata.py     # Delivery field extraction from sender/subject/body
      labels.py             # Classification label <-> id mapping
      daily_summary.py       # Builds the daily digest text
      retrain.py             # Replay-buffer retraining pipeline
    api/
      main.py               # FastAPI app, router registration
      routers/               # inbox, imap, summary endpoints
      utils/
        imap_client.py       # IMAP fetch/sync/move/cleanup logic
        inbox_processor.py    # Classification + NER processing pipeline
        folder_rules.py       # Category -> IMAP folder + retention rules
        settings.py           # IMAP/SMTP/app settings (.env)
        mailer.py             # SMTP summary email sender
    db/
      database.py           # SQLite engine, session, schema migration helper
      models.py              # Email SQLAlchemy model
  scripts/
    deploy.ps1               # Cloud Run deployment
    run_workflow.py          # Cron entry point: fetch -> process -> summary -> retrain
    backfill_summary_snapshots.py # Backfill snapshots from existing email history
    bootstrap_eval_holdout.py # One-time: carve the initial retrain eval hold-out
    sync_database_with_gcs.py # Download/edit/upload the GCS-backed SQLite database
    create_gmail_refresh_token.py # Create Gmail API OAuth credentials
    train_distilbert_optuna.py, train_ner.py, ...  # Standalone training/eval scripts
  models/
    distilbert_deployed/     # Currently deployed classifier
    distilbert_optuna/        # Hyperparameter search checkpoints
    ner-smio/                 # Deployed NER model
  data/
    raw/, labeled/
  tests/
```

---

# 🚀 Getting started

### 1. Clone the repository
```bash
git clone https://github.com/yourname/SMIO.git
cd SMIO
```

### 2. Create a virtual environment
```bash
python -m venv smio
smio\Scripts\activate      # Windows
source smio/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI server
```bash
uvicorn src.api.main:app --reload
```
API available at http://localhost:8000

### Docker
Build the API image (the `.env` file is deliberately excluded; pass secrets as environment variables):

```bash
docker build -t smio:local .
docker run --rm -p 8080:8080 --env-file .env smio:local
```

The container serves the API at `http://localhost:8080`. For Cloud Run, configure
the same environment variables through Secret Manager or the Cloud Run service
configuration; do not add `.env` to the image.

### Local operation without Google Cloud

Google Cloud Storage is optional. For a local PC or Raspberry Pi, leave
`GCS_BUCKET` unset. SMIO then uses the checked-out files directly:

- classifier: `models/distilbert_deployed`
- NER model: `models/ner-smio`
- SQLite database: `./smio.db`
- retraining state and candidates: `models/`

Before the first run on a new machine, install Git LFS and hydrate the model
files. Otherwise Git may leave large model files as small LFS pointer files:

```bash
git lfs install
git lfs pull
```

Then create a local `.env` containing the IMAP settings and start the API from
the repository root:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

Do not set `GCS_BUCKET` in local mode. After local retraining promotes a model,
the changed Git LFS file can be reviewed and versioned normally:

```bash
git add models/distilbert_deployed
git commit -m "Update deployed classifier model"
git push
```

### Cloud Run and Cloud Scheduler
The summary uses a logical day from `08:00` to the next day `08:00` in
`Europe/Berlin`. The default start hour and timezone can be changed with
`SMIO_SUMMARY_START_HOUR` and `SMIO_SUMMARY_TIMEZONE`. The scheduled daily job
runs at the same configured start time and stores the gathered data in the
`summary_snapshots` database table before sending and retraining.

The API exposes these scheduler/action endpoints:

| Method | Path | Schedule | Work |
|---|---|---|---|
| POST | `/jobs/five-minute` | Every 5 minutes | Fetch, process instruction mails, classify and file new mail |
| POST | `/jobs/daily` | Daily | Gather, send, then retrain |
| POST | `/jobs/daily/gather` | Manual | Gather and store one summary date (`force=true` can overwrite an older snapshot) |
| POST | `/jobs/daily/send` | Manual | Send the stored summary for one date |
| POST | `/jobs/daily/retrain` | Manual | Run retraining if the correction threshold is met |

For a manual source deployment, keep both the Cloud Run instance count and request
concurrency at one while SQLite is stored in GCS:

```bash
gcloud run deploy smio \
  --source . \
  --region=europe-west3 \
  --no-allow-unauthenticated \
  --max-instances=1 \
  --concurrency=1 \
  --timeout=3600 \
  --set-env-vars=GCS_BUCKET=your-smio-bucket,GCS_CLASSIFIER_MODEL_PREFIX=models/distilbert_deployed,GCS_DATABASE_OBJECT=databases/smio.db
```

Set IMAP and SMTP values with Secret Manager rather than command-line environment
variables. Give the Cloud Run service account read/write access to the GCS bucket.
Create a dedicated scheduler service account, grant it `roles/run.invoker` on the
`smio` service, then create the schedules with an OIDC token:

```bash
gcloud scheduler jobs create http smio-five-minute \
  --location=europe-west3 \
  --schedule="*/5 * * * *" \
  --uri="https://YOUR_CLOUD_RUN_URL/jobs/five-minute" \
  --http-method=POST \
  --oidc-service-account-email=YOUR_SCHEDULER_SERVICE_ACCOUNT

gcloud scheduler jobs create http smio-daily \
  --location=europe-west3 \
  --schedule="0 8 * * *" \
  --time-zone="Europe/Berlin" \
  --uri="https://YOUR_CLOUD_RUN_URL/jobs/daily" \
  --http-method=POST \
  --attempt-deadline=30m \
  --oidc-service-account-email=YOUR_SCHEDULER_SERVICE_ACCOUNT
```

Cloud Scheduler's HTTP deadline is limited to 30 minutes. If retraining can exceed
that, move the retraining part of the daily routine to a Cloud Run Job; do not let a
long training request be retried while it is still running.

## Scripts and Operations

Run these commands from the repository root.

### Deploy to Cloud Run

[scripts/deploy.ps1](scripts/deploy.ps1) reuses an active Google Cloud SDK login
(or prompts for login if needed), starts Docker Desktop if it is not running, and
waits for the Docker engine before building and pushing the image to Artifact
Registry. It grants the runtime service account access to Secret Manager and GCS,
then deploys Cloud Run with the required single-instance SQLite settings.

```powershell
.\scripts\deploy.ps1
```

Use `-IncludeGmailSecrets` when the daily summary should be sent through Gmail.
The script defaults to project `smio-509409`, region `europe-west3`, bucket
`smio-marcus-my-gcp-project-artifacts`, and the `Europe/Berlin` summary timezone.
Override its parameters when deploying to another environment.

### Edit the GCS-backed SQLite database

Use the guarded sync utility to download the current database, edit the local
copy, and optionally write changes back:

```powershell
.\smio\Scripts\python.exe .\scripts\sync_database_with_gcs.py
```

Full mode pauses the `smio-five-minute` Cloud Scheduler job, downloads and
integrity-checks the database, then waits while you edit `smio.db`. After editing,
press Enter and type uppercase `Y` to upload. If the database is unchanged, the
upload is skipped. The job resumes automatically when the script exits if this
run paused it. `--download-only` downloads and exits without pausing the job or
offering write-back. Keep other SMIO processes that could write to the database
stopped while editing.

### Backfill existing summaries

After enabling persistent snapshots on an existing database, create snapshots from
historical processed mail with:

```powershell
.\smio\Scripts\python.exe .\scripts\backfill_summary_snapshots.py `
  --db-path .\smio.db `
  --start 2026-09-01
```

The script uses the configured logical-day boundary and never enables GCS, so the
local database copy cannot upload changes to the production bucket. Use `--end`
to limit the range and `--force` to overwrite existing historical snapshots.

### Run the full workflow once

Fetch, process, summarize, and retrain in one local run:

```bash
python -m scripts.run_workflow
```
For unattended operation, schedule this command (for example, via Windows Task
Scheduler). A lockfile (`tmp/workflow.lock`) prevents overlapping runs.

### Bootstrap the retraining hold-out

Run once before the first retraining attempt:

```bash
python -m scripts.bootstrap_eval_holdout
```

### Configure Gmail summary delivery

Create a Gmail OAuth client and refresh token for the configured sender address.
The helper script opens browser authorization and prints the values to store in
the local environment or Secret Manager:

```powershell
.\smio\Scripts\python.exe .\scripts\create_gmail_refresh_token.py `
  --client-secrets path/to/client_secret.json
```

---

## 🔧 Configuration

Create a `.env` file in the project root:

```bash
# IMAP (required)
IMAP_HOST=imap.yourprovider.com
IMAP_USER=your_email@example.com
IMAP_PASSWORD=your_password
# Summary emails use IMAP_USER as their Reply-To address.

# Gmail API (optional — only needed to receive the daily summary email)
GMAIL_CLIENT_ID=your-oauth-client-id
GMAIL_CLIENT_SECRET=your-oauth-client-secret
GMAIL_REFRESH_TOKEN=your-oauth-refresh-token
GMAIL_TO_ADDRESS=your_email@example.com
# Gmail must authorize the configured sender address.
GMAIL_FROM_ADDRESS=sender@example.com
GMAIL_FROM_NAME=Smio, der Mail Organizer

# Personalization
USER_FIRST_NAME=Example User

# Optional: persist the deployed classifier and SQLite database in GCS.
# Cloud Run uses its service account through Application Default Credentials.
GCS_BUCKET=your-smio-bucket
GCS_CLASSIFIER_MODEL_PREFIX=models/distilbert_deployed
GCS_DATABASE_OBJECT=databases/smio.db
```

When the Gmail OAuth settings are absent, summary delivery is skipped. The
summary data is still stored in `summary_snapshots`, so sending can be retried
later without rebuilding the period.

To configure Gmail delivery, enable the Gmail API in the Google Cloud project,
then create an OAuth consent screen and a **Desktop app** OAuth client for
`sender@example.com`. Download that client's JSON file locally. Follow the
**Configure Gmail summary delivery** instructions in Scripts and Operations. The
browser authorization must use `sender@example.com`. Store the resulting refresh
token, client ID, and client secret in Secret Manager as `gmail-refresh-token`,
`gmail-client-id`, and `gmail-client-secret`. Configure `GMAIL_TO_ADDRESS` to the
1&1 recipient address. Do not add the downloaded OAuth client JSON or refresh token
to the repository.

When `GCS_BUCKET` is configured, SMIO downloads the deployed model and database at
startup. The first GCS-enabled run uploads its existing local artifacts if the bucket
does not yet contain them. Promoted classifier models and committed SQLite changes are
uploaded automatically. The Cloud Run service account needs `Storage Object Admin` (or
equivalent read/write access) for the configured bucket.

GCS-backed SQLite must run with a single active writer/instance. For concurrent Cloud
Run instances, set `DATABASE_URL` to a managed database such as Cloud SQL instead;
GCS continues to store the deployed classifier model.

---

## 📬 How it works

1. **Sync & fetch** — reconcile known messages across IMAP folders, fetch new INBOX mails ([src/api/utils/imap_client.py](src/api/utils/imap_client.py)).
2. **Process** — classify each new mail, run NER for `delivery` mails, move it into its category folder, apply retention-based cleanup ([src/api/utils/inbox_processor.py](src/api/utils/inbox_processor.py)).
3. **Summarize** — gather and persist a logical-day snapshot, then send its plain-language digest.
4. **Retrain (if due)** — once enough manual corrections have accumulated, fine-tune the classifier using a replay buffer (90% old data / 10% new corrections per class) and promote it only if it beats the deployed model's F1 on a fixed hold-out set.

---

## 🛠 API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/imap/fetch` | Sync folders and fetch new INBOX mails |
| GET | `/imap/sync` | Reconcile known messages across all classification folders |
| POST | `/inbox/process_unprocessed` | Classify + extract entities + move all unprocessed mails |
| POST | `/inbox/undo_last_processing` | Revert the last processing batch, restore mails to INBOX |
| GET | `/summary/daily?summary_date=YYYY-MM-DD` | Return a stored snapshot or generate a preview |
| POST | `/jobs/five-minute` | Cloud Scheduler: fetch and process new mail |
| POST | `/jobs/daily` | Cloud Scheduler: gather, send, and retrain |
| POST | `/jobs/daily/gather?summary_date=YYYY-MM-DD` | Store one fixed-period summary |
| POST | `/jobs/daily/send?summary_date=YYYY-MM-DD` | Send a stored summary |
| POST | `/jobs/daily/retrain?summary_date=YYYY-MM-DD` | Run retraining for the current DB state |

---

## 📈 Roadmap

**Done**
- IMAP ingestion, classification, NER extraction
- Folder-rule-based automated actions with retention cleanup
- Manual-correction tracking + replay-buffer retraining with a fixed eval hold-out
- Daily summary email
- Persistent summary snapshots with gather/send/retrain job controls
- Cloud Run deployment script with Artifact Registry and Secret Manager setup

**Next**
- Web dashboard

**Later**
- Multi-account support
- CI/CD pipeline

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

MIT License


