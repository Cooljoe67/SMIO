"""Builds the text a classifier sees, the same way for training and sorting.

The format is stored in the model's config.json (key smio_input), so a model
is always fed what it was trained on. Models without the key, such as the
original deployed one, see the body only.
"""

from email.utils import parseaddr

CONFIG_KEY = "smio_input"
BODY = "body"
SUBJECT_SENDER_BODY = "subject_sender_body"
INPUT_FORMATS = (BODY, SUBJECT_SENDER_BODY)


def sender_domain(sender):
    return parseaddr(sender or "")[1].rpartition("@")[2].lower()


def build_input(input_format, subject, sender, text):
    if input_format == BODY:
        return text or ""
    if input_format == SUBJECT_SENDER_BODY:
        # Subject and sender domain first, so truncation cuts the body, not them.
        return f"{subject or ''}\n{sender_domain(sender)}\n{text or ''}"
    raise ValueError(f"Unknown input format: {input_format!r}")


def model_input_format(model_config):
    return getattr(model_config, CONFIG_KEY, None) or BODY
