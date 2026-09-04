from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    IMAP_HOST: str
    EMAIL_USER: str
    EMAIL_PASS: str

    class Config:
        env_file = ".env"

settings = Settings()
