# SMIO – Smart Mail Inbox Organizer
An AI‑powered system that automatically analyzes, categorizes, and declutters your email inbox.

SMIO uses modern NLP models (HuggingFace Transformers, BERT), spaCy NER pipelines, and a FastAPI backend to intelligently process incoming emails. It removes expired commercial offers, tracks delivery notifications until completion, preserves personal messages, and generates a daily summary of all relevant inbox activity.

---

## 📌 Features

- **AI‑based Email Classification**  
  Categorizes emails into *commercial*, *delivery*, *personal*, and *other* using transformer models.

- **Named Entity Recognition (NER)**  
  Extracts dates, companies, tracking numbers, and offer validity information using spaCy.

- **Automated Inbox Actions**  
  - Delete expired commercial offers  
  - Track delivery notifications until the package arrives  
  - Archive completed deliveries  
  - Leave personal messages untouched  

- **Daily Summary Email**  
  A single digest summarizing:
  - New messages  
  - Delivery updates  
  - Completed deliveries  
  - Deleted/archived commercial offers  

- **FastAPI Backend**  
  Provides endpoints for classification, inbox processing, and summary generation.

---

## 🧠 Core Technologies

- Python 3.x  
- HuggingFace Transformers  
- BERT / DistilBERT  
- spaCy NER  
- Pandas  
- FastAPI  
- imaplib  
- SQLite / JSON state tracking


## 📂 Project Structure
```bash
SMIO/
  src/
    classifiers/
    ner/
    actions/
    api/
    utils/
    scheduler/
  models/
    bert_email_classifier/
  data/
    raw/
    labeled/
  tests/
  README.md
  requirements.txt
```
# 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourname/SMIO.git
cd SMIO
```

### 2. Create a virtual environment
```bash
   python -m venv venv
   source venv/bin/activate   (macOS/Linux)
   venv\Scripts\activate      (Windows)
```

### 3. Install dependencies
```bash
   pip install -r requirements.txt
```


### 4. Start the FastAPI server
```bash
   uvicorn src.api.main:app --reload
```

API will be available at:
http://localhost:8000

## 🔧 Configuration

Create a .env file with your email credentials:

IMAP_HOST=imap.yourprovider.com
EMAIL_USER=your_email@example.com
EMAIL_PASS=your_password

## 📬 How It Works

1. Fetch emails via IMAP
2. Classify using a fine-tuned BERT model
3. Extract entities (dates, tracking numbers, companies)
4. Apply rules
5. Update state in SQLite/JSON
6. Generate daily summary

## 🛠 API Endpoints

/classify_email   - Classifies a single email
/process_inbox    - Processes all new inbox messages
/daily_summary    - Generates daily digest
/status           - Shows current delivery & inbox state

📈 Roadmap

Phase 1 – MVP
- Email ingestion
- Classification model
- Basic NER
- Rule engine
- Daily summary

Phase 2 – Advanced Features
- Web dashboard (Streamlit)
- Smart reply suggestions
- Multi-account support
- Fine-grained user preferences

Phase 3 – Production
- Docker deployment
- OAuth email access
- Cloud hosting
- CI/CD pipeline

🤝 Contributing

Pull requests are welcome.
For major changes, please open an issue first to discuss what you would like to change.

📄 License

MIT License 

