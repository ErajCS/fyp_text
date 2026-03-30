import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# ⚠️  SCOPE REQUIREMENT: Must be full drive scope so the app can LIST existing
# folders (PDFs, Images, Videos) and upload INTO them instead of duplicating.
# drive.file / drive.readonly scopes cannot list folders created outside the app.
SCOPES = ['https://www.googleapis.com/auth/drive']

def main():
    """Generates a token.json for Google Drive API access."""
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('oauth_credentials.json'):
                print("Error: 'oauth_credentials.json' not found!")
                print("Please download your OAuth client ID JSON from Google Cloud Console and rename it to 'oauth_credentials.json' in this folder.")
                return
            
            flow = InstalledAppFlow.from_client_secrets_file('oauth_credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
            
    print("✅ token.json has been generated successfully!")

if __name__ == '__main__':
    main()
