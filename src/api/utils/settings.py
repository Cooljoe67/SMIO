"""Mail settings of the primary account, read live from config/smio.yaml."""

from src.config import primary_account


def _imap_account():
    account = primary_account()
    if account.provider != "imap":
        raise NotImplementedError(
            f"Account {account.id!r}: provider {account.provider!r} is not supported yet"
        )
    return account


class EmailSettings:
    @property
    def host(self):
        return _imap_account().imap.host

    @property
    def user(self):
        return _imap_account().imap.username

    @property
    def password(self):
        return _imap_account().imap.password


email_settings = EmailSettings()


class AppSettings:
    @property
    def user_first_name(self):
        return primary_account().user_first_name


app_settings = AppSettings()
