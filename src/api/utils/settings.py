from pydantic_settings import BaseSettings

class EmailSettings(BaseSettings):
    host: str
    user: str
    password: str

    model_config = {
        "env_prefix": "IMAP_",
        "env_file": ".env"
    }

email_settings = EmailSettings()


class SmtpSettings(BaseSettings):
    # Optional: only needed for outgoing summary mails. Empty host = disabled.
    host: str = ""
    port: int = 465
    user: str = ""
    password: str = ""
    to_address: str = ""
    from_name: str = "SMIO, your mail organizer"

    model_config = {
        "env_prefix": "SMTP_",
        "env_file": ".env"
    }

smtp_settings = SmtpSettings()


class AppSettings(BaseSettings):
    user_first_name: str = "there"

    model_config = {
        "env_file": ".env"
    }

app_settings = AppSettings()
