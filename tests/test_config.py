import pytest

from src import config
from src.api.utils import folder_rules

BASE = """
folder_lists:
  default:
    delivery: {retention_days_unread: 20}
    tech: {enabled: false, retention_days: 1}
accounts:
  - id: primary
    provider: imap
    email: ${IMAP_USER}
    user_first_name: ${SMIO_TEST_FIRST_NAME:-there}
    folder_list: default
    imap: {host: "${IMAP_HOST}", username: "${IMAP_USER}", password: "${IMAP_PASSWORD}"}
"""


@pytest.fixture
def use_config(monkeypatch):
    def apply(text):
        parsed = config.parse_config(text)
        monkeypatch.setattr(config, "_config", parsed)
        return parsed
    return apply


def test_env_references_are_expanded_with_defaults():
    parsed = config.parse_config(BASE)
    account = parsed.account("primary")

    assert account.imap.host == "imap.test"
    assert account.email == "owner@test"
    assert account.user_first_name == "there"


def test_missing_env_reference_is_rejected():
    with pytest.raises(ValueError, match="SMIO_TEST_UNSET"):
        config.parse_config(BASE.replace("${IMAP_HOST}", "${SMIO_TEST_UNSET}"))


@pytest.mark.parametrize(
    "change, message",
    [
        (("folder_list: default", "folder_list: missing"), "unknown folder list"),
        (("provider: imap", "provider: gmail"), "also has a 'imap:' section"),
        (("    imap: {", "    # imap: {"), "needs a 'imap:' section"),
        (("    tech:", "    other:"), "'other' is implicit"),
        (("retention_days_unread: 20", "retention_dayz: 20"), "retention_dayz"),
    ],
)
def test_invalid_config_is_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        config.parse_config(BASE.replace(*change))


def test_disabled_folder_keeps_label_but_is_not_sorted(use_config):
    use_config(BASE)

    assert folder_rules.classification_folders() == {"delivery", "tech"}
    assert folder_rules.folder_for_classification("delivery") == "INBOX/delivery"
    assert folder_rules.folder_for_classification("tech") is None
    assert folder_rules.folder_for_classification("other") is None
    assert folder_rules.folder_path("tech") == "INBOX/tech"
    assert folder_rules.classification_for_folder("INBOX/tech") == "tech"


def test_folder_root_and_names_come_from_the_account(use_config):
    parsed = use_config(
        BASE.replace(
            "    folder_list: default\n",
            "    folder_list: default\n"
            "    mailbox: {folder_root: smio, trash: '[Gmail]/Trash'}\n",
        ).replace("delivery: {", "delivery: {folder: Lieferungen, ")
    )
    account = parsed.account("primary")

    assert folder_rules.folder_for_classification("delivery", account) == "smio/Lieferungen"
    assert folder_rules.classification_for_folder("smio/Lieferungen", account) == "delivery"
    assert folder_rules.trash_folder(account) == "[Gmail]/Trash"


def test_broken_reload_keeps_previous_config(use_config, monkeypatch, tmp_path):
    previous = use_config(BASE)
    broken = tmp_path / "smio.yaml"
    broken.write_text("accounts: []\nfolder_lists: {}\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_PATH", broken)

    assert config.reload_config() is previous


def test_scripts_can_load_the_config_without_the_imap_password(monkeypatch):
    from src.api.utils import imap_client

    monkeypatch.delenv("IMAP_PASSWORD")
    parsed = config.parse_config(config.CONFIG_PATH.read_text(encoding="utf-8"))

    with pytest.raises(RuntimeError, match="no IMAP password"):
        imap_client.open_mailbox(parsed.accounts[0])


def test_repository_config_is_valid():
    parsed = config.parse_config(config.CONFIG_PATH.read_text(encoding="utf-8"))

    assert parsed.accounts[0].folder_list in parsed.folder_lists
