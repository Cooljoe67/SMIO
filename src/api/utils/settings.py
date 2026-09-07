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
