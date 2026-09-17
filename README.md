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
- FastAPI + SQLAlchemy (SQLite)
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

# SMTP (optional — only needed to receive the daily summary email)
SMTP_HOST=smtp.yourprovider.com
SMTP_PORT=465
SMTP_USER=your_email@example.com
SMTP_PASSWORD=your_password
SMTP_TO_ADDRESS=your_email@example.com
SMTP_FROM_NAME=SMIO, your mail organizer

# Personalization
USER_FIRST_NAME=Marcus
```

`SMTP_HOST` left empty disables the summary email (it's simply skipped, no error).

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


