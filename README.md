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
  - Moves classified mails into per-category IMAP folders with configurable retention (auto-cleanup of read mails after N days) — see [src/api/utils/folder_rules.py](src/api/utils/folder_rules.py).
  - Detects manual corrections: if you move a mail to a different folder yourself, SMIO records that as the ground-truth label (`true_label`) for future retraining.
  - `undo_last_processing` reverts the most recent processing batch and moves mails back to `INBOX`.

- **Daily summary email**
  Generates a plain-language digest of everything processed since the last summary (new `other` senders, counts of mails moved per category) and emails it to you via SMTP before the retrain step runs — see [src/ai/daily_summary.py](src/ai/daily_summary.py) and [src/api/utils/mailer.py](src/api/utils/mailer.py).

- **Self-improving classifier (replay-buffer retraining)**
  Once enough manual corrections accumulate (default: 20, configurable), SMIO fine-tunes the deployed model on a mix of the new corrections and a random sample of existing labeled data per class (90/10 replay ratio), evaluates the candidate against the currently deployed model on a fixed, continuously-growing hold-out set, and only promotes the new model if its macro-F1 is at least as good — see [src/ai/retrain.py](src/ai/retrain.py).

- **FastAPI backend**
  Endpoints for IMAP sync/fetch, inbox processing, and the daily summary (see below).

- **Sequential cron workflow**
  A single script ([scripts/run_workflow.py](scripts/run_workflow.py)) runs the whole pipeline end to end — intended to be scheduled (e.g. Windows Task Scheduler) rather than run as a background API task.

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
    run_workflow.py          # Cron entry point: fetch -> process -> summary -> retrain
    bootstrap_eval_holdout.py # One-time: carve the initial retrain eval hold-out
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

### Cloud Run and Cloud Scheduler
The API exposes two scheduler endpoints:

| Method | Path | Schedule | Work |
|---|---|---|---|
| POST | `/jobs/five-minute` | Every 5 minutes | Fetch, process instruction mails, classify and file new mail |
| POST | `/jobs/daily` | Daily | Send the daily summary and retrain if enough corrections exist |

Deploy the image from source. Keep both the Cloud Run instance count and request
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

### 5. Run the full workflow once (fetch, process, summarize, retrain)
```bash
python -m scripts.run_workflow
```
For unattended operation, schedule this command (e.g. via Windows Task Scheduler). A lockfile (`tmp/workflow.lock`) prevents overlapping runs.

### 6. Bootstrap the retraining hold-out (once, before the first retrain)
```bash
python -m scripts.bootstrap_eval_holdout
```

---

## 🔧 Configuration

Create a `.env` file in the project root:

```bash
# IMAP (required)
IMAP_HOST=imap.yourprovider.com
IMAP_USER=your_email@example.com
IMAP_PASSWORD=your_password

# Gmail API (optional — only needed to receive the daily summary email)
GMAIL_CLIENT_ID=your-oauth-client-id
GMAIL_CLIENT_SECRET=your-oauth-client-secret
GMAIL_REFRESH_TOKEN=your-oauth-refresh-token
GMAIL_TO_ADDRESS=your_email@example.com
# Gmail must authorize the configured sender address.
GMAIL_FROM_ADDRESS=cooljoe67@gmail.com
GMAIL_FROM_NAME=Smio, der Mail Organizer

# Personalization
USER_FIRST_NAME=Marcus

# Optional: persist the deployed classifier and SQLite database in GCS.
# Cloud Run uses its service account through Application Default Credentials.
GCS_BUCKET=your-smio-bucket
GCS_CLASSIFIER_MODEL_PREFIX=models/distilbert_deployed
GCS_DATABASE_OBJECT=databases/smio.db
```

When the Gmail OAuth settings are absent, summary delivery is skipped. The daily
summary period is only advanced after Gmail accepts the message.

To configure Gmail delivery, enable the Gmail API in the Google Cloud project, then
create an OAuth consent screen and a **Desktop app** OAuth client for
`cooljoe67@gmail.com`. Download that client's JSON file locally and run:

```bash
python -m pip install -r requirements.txt
python scripts/create_gmail_refresh_token.py --client-secrets path/to/client_secret.json
```

The browser authorization must use `cooljoe67@gmail.com`. Store the resulting refresh
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
3. **Summarize** — build and email a plain-language digest of what happened since the last summary.
4. **Retrain (if due)** — once enough manual corrections have accumulated, fine-tune the classifier using a replay buffer (90% old data / 10% new corrections per class) and promote it only if it beats the deployed model's F1 on a fixed hold-out set.

---

## 🛠 API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/imap/fetch` | Sync folders and fetch new INBOX mails |
| GET | `/imap/sync` | Reconcile known messages across all classification folders |
| POST | `/inbox/process_unprocessed` | Classify + extract entities + move all unprocessed mails |
| POST | `/inbox/undo_last_processing` | Revert the last processing batch, restore mails to INBOX |
| GET | `/summary/daily` | Generate (and return) the daily digest text |
| POST | `/jobs/five-minute` | Cloud Scheduler: fetch and process new mail |
| POST | `/jobs/daily` | Cloud Scheduler: send daily summary and retrain |

---

## 📈 Roadmap

**Done**
- IMAP ingestion, classification, NER extraction
- Folder-rule-based automated actions with retention cleanup
- Manual-correction tracking + replay-buffer retraining with a fixed eval hold-out
- Daily summary email

**Next**
- Deployment target (Raspberry Pi / Oracle Cloud Free Tier under evaluation)
- Dockerized deployment
- Web dashboard

**Later**
- Multi-account support
- CI/CD pipeline

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

MIT License


