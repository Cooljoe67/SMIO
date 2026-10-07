# SMIO – Smart Mail Inbox Organizer

An AI-powered system that automatically analyzes, categorizes, and declutters your email inbox.

SMIO connects to your mailbox via IMAP, classifies incoming mails with a fine-tuned DistilBERT model, extracts delivery-related details (tracking numbers, carriers, dates) with a custom NER model, moves mails into category folders according to configurable rules, sends you a daily plain-language summary, and periodically retrains the classifier on your manual corrections to improve its F1 score over time.

---

## 📌 Features

- **AI-based email classification** — categorizes mail as `delivery`, `commercial`, `social`, `tech`, or `other` with a fine-tuned DistilBERT model.
- **Delivery extraction** — identifies tracking and order details with a token-classification NER model and rule-based fallbacks.
- **Inbox automation** — files messages by category, applies read/unread retention rules, records manual folder moves as corrections, and supports undo.
- **Email commands** — accepts `UNDO [n]`, `RETRAIN`, `RELOAD MODEL`, `RESTORE MODEL`, `LOGS <1-2000>`, and `SUMMARY` from the configured mailbox; commands are moved to Trash before execution, with failure notices for errors and timed-out attempts.
- **Daily summaries** — stores logical-day snapshots, delivery details, and retraining status. Display dates use `DD/MM/YYYY`; seed training data is excluded from mailbox counts.
- **Replay-buffer retraining** — learns from manual corrections mixed with earlier labeled mail, discards rejected candidates, and promotes only when macro-F1 is no worse, the paired-bootstrap 95% lower bound is within a 2-point non-inferiority margin, and no class with at least 10 holdout examples loses more than 10 points of recall. Replaced models are archived in GCS for seven days and can be restored by email command.
- **Local and cloud operation** — run locally or on Cloud Run with Cloud Scheduler; Cloud Run loads the classifier from GCS, while local images include a classifier fallback.

## 🧠 Core technologies

- Python 3.x, PyTorch, Hugging Face Transformers, scikit-learn
- FastAPI and SQLAlchemy with SQLite
- `imap-tools` for IMAP access
- Google Cloud Run, Cloud Scheduler, and Cloud Storage (optional)

## 📂 Project structure

```text
SMIO/
  src/
    ai/                 # classification, NER, summaries, retraining
    api/
      routers/          # inbox, IMAP, summary, scheduled jobs
      utils/            # mailbox, commands, folder rules, email delivery
    db/                 # database engine, migrations, ORM models
    scheduler/          # scheduled workflow orchestration
    storage/            # Google Cloud Storage integration
  scripts/              # deployment, data import/export, training, maintenance
    deploy.ps1
    update_ner_model.py
    sync_database_with_gcs.py
    run_workflow.py
    backfill_summary_snapshots.py
    bootstrap_eval_holdout.py
    evaluate_classifier.py
    export_ner_from_db.py
    export_training_data.py
    import_training_data.py
    test_classifier.py
    train_bert_classifier.py
    train_distilbert_hparam_cv.py
    train_distilbert_optuna.py
    train_ner.py
  models/
    distilbert_deployed/
    distilbert_optuna/
    distilbert_candidate_*/
    ner-smio/
  data/
    raw/
    labeled/
  training_data.jsonl
  ner_training.jsonl
  Dockerfile
  requirements.txt
  requirements-runtime.txt
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

The default `local` image includes the checked-out classifier and NER models. The
Cloud Run target omits both models and loads them from GCS. Docker installs the
CPU-only `requirements-runtime.txt`; the full `requirements.txt` remains for local
training and evaluation. Ensure the classifier prefix exists and upload the NER
model with `-UpdateNerModel` on the first migration. The container serves the API
at `http://localhost:8080`. Configure
Cloud Run environment variables through Secret Manager or service configuration; do
not add `.env` to the image.

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

For manual `gcloud` commands, set `GCP_PROJECT_ID`, `GCP_REGION`, and optionally
`GCP_SCHEDULER_LOCATION` in your shell from `.env` first. For a manual source
deployment, keep both the Cloud Run instance count and request concurrency at one
while SQLite is stored in GCS:

```bash
gcloud run deploy smio \
  --source . \
  --project="$GCP_PROJECT_ID" \
  --region="$GCP_REGION" \
  --no-allow-unauthenticated \
  --max-instances=1 \
  --concurrency=1 \
  --timeout=3600 \
  --set-env-vars=GCS_BUCKET="$GCS_BUCKET",GCS_CLASSIFIER_MODEL_PREFIX=models/distilbert_deployed,GCS_DATABASE_OBJECT=databases/smio.db
```

Set IMAP and SMTP values with Secret Manager rather than command-line environment
variables. Give the Cloud Run service account read/write access to the GCS bucket.
Create a dedicated scheduler service account, grant it `roles/run.invoker` on the
`smio` service, then create the schedules with an OIDC token:

```bash
SCHEDULER_LOCATION="${GCP_SCHEDULER_LOCATION:-$GCP_REGION}"

gcloud scheduler jobs create http smio-five-minute \
  --location="$SCHEDULER_LOCATION" \
  --project="$GCP_PROJECT_ID" \
  --schedule="*/5 * * * *" \
  --uri="https://YOUR_CLOUD_RUN_URL/jobs/five-minute" \
  --http-method=POST \
  --oidc-service-account-email=YOUR_SCHEDULER_SERVICE_ACCOUNT

gcloud scheduler jobs create http smio-daily \
  --location="$SCHEDULER_LOCATION" \
  --project="$GCP_PROJECT_ID" \
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
Registry. It applies the cleanup policy in [scripts/artifact-cleanup-policy.json](scripts/artifact-cleanup-policy.json)
(keep the newest two images, delete older ones after 1 day). It grants the runtime service account access to Secret Manager and GCS,
and read-only Cloud Logging access for the `LOGS` email command. It then deploys
Cloud Run with the required single-instance SQLite settings.
Before deploying, it pauses the configured daily and five-minute Scheduler jobs,
asks you to confirm in-flight workflows have finished, and backs up the existing
GCS database to a timestamped object under `backups/`. It resumes only the jobs it
paused, including when deployment fails. Jobs already paused remain paused.
The deploy defaults are 2 vCPU and 8 GiB memory to support CPU retraining.

```powershell
.\scripts\deploy.ps1
```

Add `-UpdateNerModel` to upload `models/ner-smio` as a new immutable GCS release
and deploy a Cloud Run revision pinned to it. Use `-NerModelDirectory` to choose a
different trained model directory.

Daily summaries are appended directly to the configured IMAP inbox. The deploy
script reads `GCP_PROJECT_ID`, `GCP_REGION`, and `GCS_BUCKET` from `.env`;
`GCP_SCHEDULER_LOCATION` is optional and defaults to `GCP_REGION`. Explicit
script parameters override `.env` values.

### Email commands

Reply to a daily summary or use the subject `SMIO Command`, with one command on
the first line. For example, `LOGS 100` appends up to 100 recent Cloud Run API
log entries as `smio-api-logs.txt`; the allowed count is 1–2000. Commands are
moved to Trash before execution. If no completion is recorded within
`SMIO_COMMAND_TIMEOUT_MINUTES` (15 minutes by default), the next five-minute job
reports the timeout by email.

### Replace only the NER model

After training NER locally, upload it and update the Cloud Run service without
rebuilding or pushing the Docker image:

```powershell
.\smio\Scripts\python.exe .\scripts\update_ner_model.py --model-dir .\models\ner-smio
```

The script validates the model files, uploads a timestamped release under
`GCS_NER_MODEL_PREFIX`, updates `current.json`, and creates a new Cloud Run
revision pinned to that release. If the service update fails, it restores the
previous pointer.

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

---

## 🔧 Configuration

Copy `.env.example` to `.env` in the project root, then fill in your values.
The deploy and database-sync scripts load `.env` automatically; keep `.env`
local and do not commit it.

```bash
# Google Cloud deployment
GCP_PROJECT_ID=your-google-cloud-project-id
GCP_REGION=your-cloud-run-and-artifact-region
# Optional; defaults to GCP_REGION.
GCP_SCHEDULER_LOCATION=your-cloud-scheduler-region
GCS_BUCKET=your-gcs-bucket

# IMAP (required)
IMAP_HOST=imap.yourprovider.com
IMAP_USER=your_email@example.com
IMAP_PASSWORD=your_password
# Daily summaries are appended directly to IMAP_USER's INBOX.

# Personalization
USER_FIRST_NAME=Example User

# Optional GCS object locations
# Cloud Run uses its service account through Application Default Credentials.
GCS_CLASSIFIER_MODEL_PREFIX=models/distilbert_deployed
GCS_NER_MODEL_PREFIX=models/ner-smio
GCS_DATABASE_OBJECT=databases/smio.db
SMIO_COMMAND_TIMEOUT_MINUTES=15
```

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
3. **Summarize** — gather and persist a logical-day snapshot, then append its plain-language digest to the IMAP inbox.
4. **Retrain (if due)** — once enough manual corrections have accumulated, fine-tune the classifier using a replay buffer (90% old data / 10% new corrections per class) and apply the macro-F1, paired-bootstrap, and per-class recall promotion gates on a fixed hold-out set. After a completed run, SMIO emails the baseline and candidate metrics, confidence interval, per-class recall, training duration, and promotion outcome.

---

## 🛠 API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/imap/fetch` | Sync folders and fetch new INBOX mails |
| GET | `/imap/sync` | Reconcile known messages across all classification folders |
| POST | `/inbox/process_unprocessed` | Classify + extract entities + move all unprocessed mails |
| POST | `/inbox/undo_last_processing?count=n` | Restore the mails of the last `n` processing batches (default 1, max 20) to INBOX |
| GET | `/summary/daily?summary_date=YYYY-MM-DD` | Return a stored snapshot or generate a preview |
| POST | `/jobs/five-minute` | Cloud Scheduler: fetch and process new mail |
| POST | `/jobs/daily` | Cloud Scheduler: gather, append to inbox, and retrain |
| POST | `/jobs/daily/gather?summary_date=YYYY-MM-DD` | Store one fixed-period summary |
| POST | `/jobs/daily/send?summary_date=YYYY-MM-DD` | Append a stored summary to the IMAP inbox |
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


