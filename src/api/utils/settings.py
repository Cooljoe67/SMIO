from pydantic_settings import BaseSettings

class EmailSettings(BaseSettings):
    host: str
    user: str
    password: str

    model_config = {
        "env_prefix": "IMAP_",
        "env_file": ".env",
        "extra": "ignore"
    }

email_settings = EmailSettings()


class GmailSettings(BaseSettings):
    """OAuth client details for Gmail API delivery of SMIO summaries."""

    client_id: str = ""
    client_secret: str = ""
    refresh_token: str = ""
    to_address: str = ""
    from_address: str = "cooljoe67@gmail.com"
    from_name: str = "Smio, der Mail Organizer"

    model_config = {
        "env_prefix": "GMAIL_",
        "env_file": ".env",
        "extra": "ignore"
    }

gmail_settings = GmailSettings()


class AppSettings(BaseSettings):
    user_first_name: str = "there"

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

app_settings = AppSettings()
