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


class AppSettings(BaseSettings):
    user_first_name: str = "there"

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

app_settings = AppSettings()
