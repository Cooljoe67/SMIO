from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base
from datetime import datetime, timezone

Base = declarative_base()


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Email(Base):
    __tablename__ = "emails"

    id = Column(Integer, primary_key=True, index=True)
    # Id of the configured account (config/smio.yaml) the mail belongs to.
    account = Column(String, index=True)
    # IMAP UID: only valid in the current folder and changes on every move.
    uid = Column(String, index=True)
    message_id = Column(String, index=True)
    # Provider's stable id (Gmail message id, Exchange ImmutableId); none for IMAP.
    provider_message_id = Column(String, nullable=True, index=True)
    # Hash of the headers a move leaves unchanged (src/api/utils/mail_identity.py).
    fingerprint = Column(String, nullable=True, index=True)
    folder = Column(String, default="INBOX")
    removed_at = Column(DateTime, nullable=True, index=True)
    removal_reason = Column(String, nullable=True)
    last_seen_at = Column(DateTime, nullable=True, index=True)

    subject = Column(String)
    sender = Column(String)
    sender_name = Column(String, nullable=True)
    recipients = Column(Text, nullable=True)
    attachment_names = Column(Text, nullable=True)
    # INBOUND (external sender) or INTERNAL (sender in the account's own domain).
    direction = Column(String, nullable=True)
    is_own_domain_sender = Column(Boolean, nullable=True)
    is_reply = Column(Boolean, nullable=True)
    is_forward = Column(Boolean, nullable=True)
    # Message-IDs this mail answers (In-Reply-To, References), to follow threads.
    in_reply_to = Column(String, nullable=True, index=True)
    reference_ids = Column(Text, nullable=True)
    # Mail whose status this reply was checked against, and when.
    reply_to_id = Column(Integer, nullable=True, index=True)
    reply_checked_at = Column(DateTime, nullable=True)
    date = Column(DateTime, default=_utcnow)
    read_at = Column(DateTime, nullable=True, index=True)

    text = Column(Text)
    html = Column(Text)


    # AI fields
    # Folder list (config/smio.yaml) that classification and true_label refer to.
    folder_list = Column(String, nullable=True, index=True)
    classification = Column(String, nullable=True)
    classification_source = Column(String, nullable=True)
    confidence = Column(String, nullable=True)
    embedding = Column(Text, nullable=True)

    # Predicted mail handling ("VORSCHLAG_*" in the labelled dataset).
    status = Column(String, nullable=True)
    priority = Column(String, nullable=True)
    action = Column(String, nullable=True)
    mail_type_1 = Column(String, nullable=True)
    mail_type_2 = Column(String, nullable=True)
    classification_reason = Column(Text, nullable=True)
    # Where the status came from (model, reply, command, dataset) and why;
    # a reply below the confidence threshold only leaves a suggestion.
    status_source = Column(String, nullable=True)
    status_reason = Column(Text, nullable=True)
    status_suggestion = Column(String, nullable=True)
    status_updated_at = Column(DateTime, nullable=True)
    # Business case the mail belongs to, e.g. a shipment or order reference.
    case_ref = Column(String, nullable=True, index=True)

    order_id = Column(String, nullable=True)
    tracking_id = Column(String, nullable=True)
    entity_date = Column(String, nullable=True)
    entity_name = Column(String, nullable=True)
    entity_company = Column(String, nullable=True)
    delivery_status = Column(String, nullable=True)
    item_name = Column(String, nullable=True)
    delivery_date = Column(String, nullable=True)
    delivery_company = Column(String, nullable=True)
    ner_entities = Column(Text, nullable=True)
    extraction_sources = Column(Text, nullable=True)

    # Number of times this email has been processed.
    processed = Column(Integer, default=0, nullable=False)
    processing_batch = Column(Integer, nullable=True, index=True)
    processed_at = Column(DateTime, nullable=True, index=True)

    true_label = Column(String, nullable=True)
    # Confirmed handling values, the counterparts of the predictions above.
    true_status = Column(String, nullable=True)
    true_priority = Column(String, nullable=True)
    true_action = Column(String, nullable=True)
    true_mail_type_1 = Column(String, nullable=True)
    true_mail_type_2 = Column(String, nullable=True)
    # A person checked the confirmed values ("GEPRUEFT").
    reviewed = Column(Boolean, nullable=True)
    note = Column(Text, nullable=True)
    traits = Column(Text, nullable=True)

    # Origin of imported labelled data: dataset row key, TRAIN/TEST split, difficulty.
    dataset_key = Column(String, nullable=True, index=True)
    dataset_split = Column(String, nullable=True)
    difficulty = Column(String, nullable=True)

    # Set to the retrain run id once this correction has been used for fine-tuning.
    retrain_batch = Column(Integer, nullable=True, index=True)

    # Permanently reserved for evaluation; never used for training.
    is_eval_holdout = Column(Boolean, default=False, nullable=False)
    is_training_data = Column(Boolean, default=False, nullable=False)


class MailSyncState(Base):
    """Where the last sync of an account's folder stopped (src/mail/base.py)."""

    __tablename__ = "mail_sync_state"
    __table_args__ = (UniqueConstraint("account", "folder"),)

    id = Column(Integer, primary_key=True)
    account = Column(String, nullable=False, index=True)
    folder = Column(String, nullable=False)
    # Opaque to everything but the account's adapter (IMAP: UIDVALIDITY:UIDNEXT).
    cursor = Column(String, nullable=True)
    # Last complete read: only that can tell that a mail has left the folder.
    full_sync_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=_utcnow, nullable=False)


class MailWatch(Base):
    """An account's push notifications from its provider (Gmail: users.watch)."""

    __tablename__ = "mail_watches"

    account = Column(String, primary_key=True)
    # The watched folders, sorted and one per line: a new folder needs a new watch.
    folders = Column(Text, nullable=False, default="")
    expires_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=_utcnow, nullable=False)


class MailEvent(Base):
    """A change of a mail's state, in the provider-neutral format of
    src/mail/events.py; also the outbox its actions are run from."""

    __tablename__ = "mail_events"
    __table_args__ = (UniqueConstraint("email_id", "seq"),)

    id = Column(Integer, primary_key=True)
    # "<account>:<email id>:<seq>"; identifies the event to actions and consumers.
    event_key = Column(String, nullable=False, unique=True)
    account = Column(String, nullable=False, index=True)
    provider = Column(String, nullable=False)
    email_id = Column(Integer, nullable=False, index=True)
    # Numbers the events of one mail; a second detector recording the same
    # change collides on (email_id, seq) instead of recording it twice.
    seq = Column(Integer, nullable=False)
    event_type = Column(String, nullable=False)
    message_id = Column(String, nullable=True)
    from_folder = Column(String, nullable=True)
    to_folder = Column(String, nullable=True)
    # "user": found on the server; "system": done by SMIO.
    source = Column(String, nullable=False)
    occurred_at = Column(DateTime, default=_utcnow, nullable=False)
    # From a folder's first read, or SMIO sorting mail of one: recorded, no actions.
    initial = Column(Boolean, default=False, nullable=False)
    # Set once every action that applies has run or given up.
    dispatched_at = Column(DateTime, nullable=True, index=True)


class MailEventAction(Base):
    """One action's run for one event; at most one row per pair, so an action
    never runs twice for the same event."""

    __tablename__ = "mail_event_actions"
    __table_args__ = (UniqueConstraint("event_id", "action"),)

    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, nullable=False, index=True)
    # An action runs at most once per mail, whichever of its events triggers it.
    email_id = Column(Integer, nullable=False, index=True)
    action = Column(String, nullable=False)
    # running (claimed; stays so if the process died during the action), done, failed.
    status = Column(String, nullable=False)
    attempts = Column(Integer, default=0, nullable=False)
    error = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=_utcnow, nullable=False)


class SummarySnapshot(Base):
    __tablename__ = "summary_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    summary_date = Column(String, nullable=False, unique=True, index=True)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    # When this snapshot's data was collected, not when the row was last touched.
    gathered_at = Column(DateTime, default=_utcnow, nullable=False)
    data_json = Column(Text, nullable=False)
    message = Column(Text, nullable=False)
    html_message = Column(Text, nullable=False)
    sent_at = Column(DateTime, nullable=True)
    send_status = Column(String, nullable=True)
    retrain_due = Column(Boolean, default=False, nullable=False)
    retrain_run_id = Column(Integer, nullable=True)
    retrain_result_json = Column(Text, nullable=True)

    # Flattened metrics, queryable directly without parsing data_json or joining emails.
    total_entries = Column(Integer, nullable=True)
    processed_entries = Column(Integer, nullable=True)
    correction_count = Column(Integer, nullable=True)
    pending_corrections = Column(Integer, nullable=True)
    correction_threshold = Column(Integer, nullable=True)
    new_count_total = Column(Integer, nullable=True)
    unread_count_total = Column(Integer, nullable=True)
    removed_count_total = Column(Integer, nullable=True)
    retrain_last_run_at = Column(String, nullable=True)
    retrain_last_promoted_at = Column(String, nullable=True)
    retrain_accuracy = Column(String, nullable=True)
    retrain_recall = Column(String, nullable=True)
    retrain_f1 = Column(String, nullable=True)


class RetrainRun(Base):
    __tablename__ = "retrain_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_key = Column(String, nullable=False, unique=True, index=True)
    run_id = Column(Integer, nullable=True, index=True)
    run_at = Column(String, nullable=False, index=True)
    promoted = Column(Boolean, nullable=False, default=False)
    eval_holdout_size = Column(Integer, nullable=True)
    baseline_accuracy = Column(String, nullable=True)
    baseline_recall_macro = Column(String, nullable=True)
    baseline_f1 = Column(String, nullable=True)
    new_accuracy = Column(String, nullable=True)
    new_recall_macro = Column(String, nullable=True)
    new_f1 = Column(String, nullable=True)
    training_duration = Column(Float, nullable=True)
    result_json = Column(Text, nullable=False)

