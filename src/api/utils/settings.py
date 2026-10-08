"""Settings of the primary account, read live from config/smio.yaml."""

from src.config import primary_account


class AppSettings:
    @property
    def user_first_name(self):
        return primary_account().user_first_name


app_settings = AppSettings()
