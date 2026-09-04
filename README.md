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



