"""Gmail adapter: the Gmail API over REST, authorised with google-auth.

Folders are labels. A folder name is a label name; system labels are named by
their ids (INBOX, SENT, TRASH, DRAFT). A message can carry several labels: the
sync scans the folder list's labels before the inbox, so a message counts as
being in the first one it finds (the copy rule in src/mail/sync.py). Moving
replaces one label with another, like "Move to" in Gmail.

Reading content costs one request per message, so reads return stubs (id and
read flag, from cheap id listings) and the sync fetches content only for mail it
has not stored. A complete read imports only mail of the last IMPORT_DAYS; older
mail in the inbox is left alone. Changes are read from Gmail's history: the
cursor is a historyId, which expires after about a week.

Push notifications: users.watch makes Gmail publish a message to a Pub/Sub topic
on every change in the watched labels, for seven days; the sync renews it. The
topic's push subscription calls /events/gmail (src/api/routers/events.py).

Authorisation (personal Gmail account): an OAuth client of type "Desktop app"
in the project's Google Auth Platform (External, In production; the
gmail.modify scope may stay unverified for personal use) and a refresh token
per mailbox, both stored in Secret Manager by scripts/add_account.ps1.
Google Workspace would differ: no refresh tokens, but a service account with
domain-wide delegation (Admin console > Security > API controls: the service
account's client id with the gmail.modify scope), used as
service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
.with_subject(account.email); an "Internal" app needs no publishing. An
organisation with the policy "domain restricted sharing" must also allow the
Google-owned publisher gmail-api-push@system.gserviceaccount.com on the topic.
"""

import base64
import logging
import os
import time as clock
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from email import message_from_bytes

from imap_tools import MailMessage as ParsedMessage

from src.storage import secret_store

from .base import CursorExpired, FolderState, Mailbox, MailMessage
from .imap import to_mail_message

logger = logging.getLogger(__name__)

API = "https://gmail.googleapis.com/gmail/v1/users/me"
TOKEN_URI = "https://oauth2.googleapis.com/token"
# Read, label, trash (not delete for good), drafts and sending.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
# A complete read stores new mail of these last days only.
IMPORT_DAYS = int(os.getenv("SMIO_GMAIL_IMPORT_DAYS", "30"))
TIMEOUT_SECONDS = 30
# Waits after Gmail's per-user rate limit (about 3000 reads a minute): 1, 2, ... 32 s.
RATE_LIMIT_RETRIES = 6
_sleep = clock.sleep


class GmailApiError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f"Gmail API {status}: {message}")
        self.status = status


def _rate_limited(response):
    if response.status_code == 429:
        return True
    if response.status_code != 403:
        return False
    try:
        error = response.json()["error"]
    except Exception:
        return False
    reasons = {item.get("reason") for item in error.get("errors", [])}
    return bool(reasons & {"rateLimitExceeded", "userRateLimitExceeded"}) or (
        "per minute" in error.get("message", "")
    )


def _credential(value, secret_name):
    return value if secret_name is None else secret_store.read_secret(secret_name)


def connect(account):
    """An authorised HTTP session for the account's mailbox."""
    if account.unavailable:
        raise RuntimeError(f"Account {account.id!r} is unavailable: {account.unavailable}")
    if account.provider != "gmail":
        raise NotImplementedError(f"Account {account.id!r} is not a Gmail account")
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2.credentials import Credentials

    gmail = account.gmail
    credentials = Credentials(
        token=None,
        refresh_token=_credential(gmail.refresh_token, gmail.refresh_token_secret),
        client_id=gmail.client_id,
        client_secret=_credential(gmail.client_secret, gmail.client_secret_secret),
        token_uri=TOKEN_URI,
        scopes=SCOPES,
    )
    return AuthorizedSession(credentials)


def address_key(address):
    """An address in the form Gmail compares it: case and dots in the name do
    not matter, and googlemail.com is gmail.com."""
    name, _, domain = (address or "").strip().lower().partition("@")
    if domain in ("gmail.com", "googlemail.com"):
        return name.replace(".", "") + "@gmail.com"
    return f"{name}@{domain}"


def _b64encode(raw):
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _b64decode(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _epoch(day):
    return int(datetime.combine(day, time.min, tzinfo=timezone.utc).timestamp())


class GmailMailbox(Mailbox):
    def __init__(self, account, session=None):
        super().__init__(account)
        self._reconnect = session is None
        self.session = session if session is not None else connect(account)
        self._labels = None
        # Per session: history pages by start cursor, label ids by message id.
        self._history = {}
        self._labels_of = {}

    def close(self):
        self.session.close()

    def _request(self, method, path, params=None, json=None, retried=False):
        from google.auth.exceptions import RefreshError

        try:
            for attempt in range(RATE_LIMIT_RETRIES + 1):
                response = self.session.request(
                    method, f"{API}/{path}", params=params, json=json, timeout=TIMEOUT_SECONDS
                )
                if not _rate_limited(response) or attempt == RATE_LIMIT_RETRIES:
                    break
                wait = 2 ** attempt
                logger.info("Gmail rate limit for account %s; retrying in %d s", self.account.id, wait)
                _sleep(wait)
        except RefreshError as error:
            gmail = self.account.gmail
            if self._reconnect and not retried:
                # The token may have been renewed by add_account.ps1 since it was cached.
                for name in (gmail.refresh_token_secret, gmail.client_secret_secret):
                    if name:
                        secret_store.invalidate(name)
                self.session = connect(self.account)
                return self._request(method, path, params, json, retried=True)
            raise RuntimeError(
                f"Gmail login of account {self.account.id!r} failed ({error}); run "
                f"scripts/add_account.ps1 -Provider gmail -AccountId {self.account.id} again"
            ) from error
        if response.status_code >= 400:
            try:
                message = response.json()["error"]["message"]
            except Exception:
                message = response.text[:500]
            raise GmailApiError(response.status_code, message)
        return response.json() if response.content else {}

    def _list_ids(self, label_ids=(), query=None):
        params = {"maxResults": 500}
        if label_ids:
            params["labelIds"] = list(label_ids)
        if query:
            params["q"] = query
        ids = []
        while True:
            page = self._request("GET", "messages", params=params)
            ids += [message["id"] for message in page.get("messages", [])]
            if not page.get("nextPageToken"):
                return ids
            params["pageToken"] = page["nextPageToken"]

    def _label_ids(self):
        if self._labels is None:
            labels = self._request("GET", "labels").get("labels", [])
            self._labels = {label["name"]: label["id"] for label in labels}
        return self._labels

    def _label_id(self, folder, create=False):
        labels = self._label_ids()
        if folder not in labels and create:
            label = self._request("POST", "labels", json={
                "name": folder,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            })
            labels[folder] = label["id"]
        return labels.get(folder)

    def _cursor(self):
        return str(self._request("GET", "profile")["historyId"])

    def _fetch(self, message_id):
        """The message with its label ids, or None if it was deleted."""
        try:
            return self._request("GET", f"messages/{message_id}", params={"format": "raw"})
        except GmailApiError as error:
            if error.status == 404:
                return None
            raise

    def _message(self, folder, data):
        labels = data.get("labelIds", [])
        self._labels_of[data["id"]] = labels
        parsed = to_mail_message(ParsedMessage.from_bytes(_b64decode(data["raw"])), folder)
        return replace(
            parsed, ref=data["id"], provider_id=data["id"], seen="UNREAD" not in labels
        )

    def _load(self, folder, message_id):
        data = self._fetch(message_id)
        return None if data is None else self._message(folder, data)

    def _stub(self, folder, message_id, seen, importable=True):
        return MailMessage(
            folder=folder,
            ref=message_id,
            provider_id=message_id,
            seen=seen,
            importable=importable,
            loader=lambda: self._load(folder, message_id),
        )

    def folders(self):
        return list(self._label_ids())

    def watch(self, target, folders):
        labels = [label for label in map(self._label_id, folders) if label]
        response = self._request("POST", "watch", json={
            "topicName": target,
            "labelIds": labels,
            "labelFilterBehavior": "include",
        })
        return datetime.fromtimestamp(
            int(response["expiration"]) / 1000, timezone.utc
        ).replace(tzinfo=None)

    def sent_folder(self):
        return self.account.mailbox.sent or "SENT"

    def drafts_folder(self):
        return self.account.mailbox.drafts or "DRAFT"

    def read_folder(self, folder, since=None, sender=None):
        # Before listing: a change made meanwhile is read again, never missed.
        cursor = self._cursor()
        label = self._label_id(folder)
        if label is None:
            return FolderState(folder=folder, messages=[], complete=True, cursor=cursor)
        terms = []
        if since:
            terms.append(f"after:{_epoch(since)}")
        if sender:
            terms.append(f"from:{sender}")
        query = " ".join(terms) or None
        ids = self._list_ids([label], query)
        unread = set(self._list_ids([label, "UNREAD"], query))
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=IMPORT_DAYS)
        recent = set(ids) if since else set(self._list_ids([label], f"after:{_epoch(cutoff)}"))
        messages = [
            self._stub(folder, message_id, message_id not in unread, message_id in recent)
            for message_id in ids
        ]
        return FolderState(folder=folder, messages=messages, complete=not terms, cursor=cursor)

    def _history_since(self, cursor):
        """History records after cursor and the cursor after them; one listing per session."""
        if not str(cursor or "").isdigit():
            raise CursorExpired(f"Not a Gmail historyId: {cursor!r}")
        if cursor not in self._history:
            params = {
                "startHistoryId": cursor,
                "historyTypes": ["messageAdded", "labelAdded"],
                "maxResults": 500,
            }
            records = []
            while True:
                try:
                    page = self._request("GET", "history", params=params)
                except GmailApiError as error:
                    if error.status == 404:
                        raise CursorExpired(f"historyId {cursor} is too old") from error
                    raise
                records += page.get("history", [])
                if not page.get("nextPageToken"):
                    break
                params["pageToken"] = page["nextPageToken"]
            self._history[cursor] = (records, str(page.get("historyId", cursor)))
        return self._history[cursor]

    def _current_labels(self, message_id):
        if message_id not in self._labels_of:
            try:
                data = self._request(
                    "GET", f"messages/{message_id}", params={"format": "minimal"}
                )
            except GmailApiError as error:
                if error.status != 404:
                    raise
                data = {"labelIds": None}
            self._labels_of[message_id] = data.get("labelIds")
        return self._labels_of[message_id]

    def read_changes(self, folder, cursor):
        records, new_cursor = self._history_since(cursor)
        label = self._label_id(folder)
        candidates = []
        for record in records if label else ():
            for added in record.get("messagesAdded", []):
                if label in added["message"].get("labelIds", []):
                    candidates.append(added["message"]["id"])
            for added in record.get("labelsAdded", []):
                if label in added.get("labelIds", []):
                    candidates.append(added["message"]["id"])
        messages = []
        for message_id in dict.fromkeys(candidates):
            labels = self._current_labels(message_id)
            # Gone again, or deleted, by now: the next complete read tells.
            if labels is None or label not in labels:
                continue
            messages.append(self._stub(folder, message_id, "UNREAD" not in labels))
        return FolderState(folder=folder, messages=messages, complete=False, cursor=new_cursor)

    def get(self, folder, ref, headers_only=True):
        data = self._fetch(ref)
        if data is None or self._label_id(folder) not in data.get("labelIds", []):
            return None
        return self._message(folder, data)

    def find(self, folder, message_id):
        label = self._label_id(folder)
        if label is None:
            return []
        # Newest first from Gmail; oldest first here.
        return self._list_ids([label], f"rfc822msgid:{message_id.strip('<>')}")[::-1]

    def move(self, folder, refs, destination):
        remove = self._label_id(folder)
        self._request("POST", "messages/batchModify", json={
            "ids": list(refs),
            "addLabelIds": [self._label_id(destination, create=True)],
            "removeLabelIds": [remove] if remove else [],
        })

    def trash(self, folder, refs):
        for ref in refs:
            self._request("POST", f"messages/{ref}/trash")

    def append(self, raw, folder, seen=False):
        labels = [self._label_id(folder, create=True)] + ([] if seen else ["UNREAD"])
        self._request(
            "POST", "messages", params={"internalDateSource": "dateHeader"},
            json={"raw": _b64encode(raw), "labelIds": labels},
        )

    def create_draft(self, raw):
        message = {"raw": _b64encode(raw)}
        # Gmail shows a draft in the thread of the mail it answers only with its threadId.
        answered = (message_from_bytes(raw).get("In-Reply-To") or "").strip()
        if answered:
            ids = self._list_ids(query=f"rfc822msgid:{answered.strip('<>')}")
            if ids:
                found = self._request("GET", f"messages/{ids[0]}", params={"format": "minimal"})
                message["threadId"] = found["threadId"]
        self._request("POST", "drafts", json={"message": message})
