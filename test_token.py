from google.oauth2 import service_account
from google.auth.transport.requests import Request

SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform"
]

credentials = (
    service_account.Credentials
    .from_service_account_file(
        "credentials/service-account.json",
        scopes=SCOPES
    )
)

credentials.refresh(Request())

print("Access token obtained successfully")
print(credentials.project_id)
print(credentials.token[:20])