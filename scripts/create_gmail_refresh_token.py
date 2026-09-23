"""Run locally once to authorize Gmail API delivery and print a refresh token."""

import argparse

from google_auth_oauthlib.flow import InstalledAppFlow


GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client-secrets",
        required=True,
        help="Path to the OAuth desktop-client JSON downloaded from Google Cloud.",
    )
    args = parser.parse_args()

    flow = InstalledAppFlow.from_client_secrets_file(
        args.client_secrets,
        scopes=[GMAIL_SEND_SCOPE],
    )
    credentials = flow.run_local_server(
        port=0,
        access_type="offline",
        prompt="consent",
    )
    if not credentials.refresh_token:
        raise RuntimeError("Google did not return a refresh token")

    print("\nStore this as the gmail-refresh-token secret:")
    print(credentials.refresh_token)


if __name__ == "__main__":
    main()