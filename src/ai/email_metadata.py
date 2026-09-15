import json
import re

from src.ai.ner import amazon_sender_status, entities_to_fields, extract_entities


CARRIER_DOMAINS = {
    "amazon.de": "Amazon",
    "amazon.com": "Amazon",
    "dhl.de": "DHL",
    "dpd.de": "DPD",
    "hermes.de": "Hermes",
    "gls-group.com": "GLS",
    "ups.com": "UPS",
    "fedex.com": "FedEx",
}

STATUS_PATTERNS = (
    ("FAILED", r"konnte\s+nicht\s+zugestellt\s+werden|fehlgeschlagen"),
    ("DELIVERED", r"(?:wurde|ist)\s+zugestellt|geliefert"),
    ("IN_TRANSIT", r"in\s+zustellung|wird\s+zugestellt"),
    ("DELAYED", r"verzögert|verspätet"),
    ("READY_FOR_PICKUP", r"abholbereit|abholung"),
    ("IN_WAREHOUSE", r"paketzentrum|im\s+lager"),
    ("SENT", r"versendet|verschickt"),
)

DATE_PATTERN = r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"


def extract_email_metadata(sender, subject, body):
    sender = sender or ""
    subject = subject or ""
    body = body or ""
    entities = extract_entities(body)
    fields = entities_to_fields(entities)
    sources = {}

    company = _company_from_sender(sender)
    if company:
        fields["delivery_company"] = company
        sources["delivery_company"] = "sender"
    elif fields.get("entity_company"):
        fields["delivery_company"] = fields["entity_company"]
        sources["delivery_company"] = "ner"

    item_name = _item_from_subject(subject)
    if item_name:
        fields["item_name"] = item_name
        sources["item_name"] = "subject"

    status = amazon_sender_status(sender, subject, body)
    if status:
        fields["delivery_status"] = status
        sources["delivery_status"] = "sender_or_subject"
    else:
        status = _status_from_text(subject)
        if status:
            fields["delivery_status"] = status
            sources["delivery_status"] = "subject"
        else:
            status = _status_from_text(body)
            if status:
                fields["delivery_status"] = status
                sources["delivery_status"] = "body"

    delivery_date = _delivery_date(subject, body)
    if delivery_date:
        fields["delivery_date"] = delivery_date
        sources["delivery_date"] = "subject_or_body"

    for field_name in ("order_id", "tracking_id"):
        if fields.get(field_name):
            sources[field_name] = "body_ner_or_rules"

    return fields, entities, sources


def _company_from_sender(sender):
    match = re.search(r"@([^>\s]+)", sender.lower())
    if not match:
        return None
    domain = match.group(1).strip().rstrip(".")
    for known_domain, company in CARRIER_DOMAINS.items():
        if domain == known_domain or domain.endswith("." + known_domain):
            return company
    return None


def _item_from_subject(subject):
    match = re.search(
        r":\s*[\"“„']?([^\"”’']+?)[\"”’']?(?:\s*$|\.{3})",
        subject,
    )
    if not match:
        return None
    item = re.sub(r"\s+", " ", match.group(1)).strip(" .:-")
    return item or None


def _status_from_text(text):
    for status, pattern in STATUS_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return status
    return None


def _delivery_date(subject, body):
    combined = f"{subject} {body}"
    contextual_pattern = (
        r"(?:zugestellt|geliefert|lieferung|zustellung|erwartet|ankunft)"
        r"[^.\n]{0,80}?([^\d]|)(" + DATE_PATTERN + r")"
    )
    match = re.search(contextual_pattern, combined, flags=re.IGNORECASE)
    if match:
        return match.group(2)
    match = re.search(DATE_PATTERN, combined)
    return match.group(0) if match else None


def serialize_sources(sources):
    return json.dumps(sources, ensure_ascii=False)
