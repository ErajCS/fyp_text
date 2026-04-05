"""
drive_service.py — Google Drive helper for PQNK Knowledge Intelligence System

Setup:
  1. Create a Google Cloud project and enable the Drive API.
  2. Create a Service Account → download the JSON key.
  3. Share your target Drive folder with the service account email (Editor).
  4. Set these env vars in your .env:
       GOOGLE_DRIVE_CREDENTIALS_JSON = /absolute/path/to/service-account.json
       GOOGLE_DRIVE_FOLDER_ID        = <folder-id-from-drive-url>

If either env var is missing the module works in NO-OP mode (local storage only).
"""

import os
import io
import logging

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/drive"]
_REQUIRED_SCOPE = "https://www.googleapis.com/auth/drive"

# ── Lazy-init ─────────────────────────────────────────────────────────────────
_drive = None   # google.drive.v3 Resource object, or None

def _get_drive():
    global _drive
    if _drive is not None:
        return _drive

    creds = None
    # 1. Try OAuth2 (token.json) - Preferred for personal accounts (uses user quota)
    token_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "token.json")
    if os.path.exists(token_path):
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        try:
            creds = Credentials.from_authorized_user_file(token_path, _SCOPES)
            # ── Scope mismatch guard ──────────────────────────────────────────
            # If the saved token was authorised with a narrower scope (e.g.
            # drive.file), it cannot list pre-existing folders and will always
            # create duplicates.  Delete the token so the user re-authorises.
            token_scopes = set(creds.scopes or [])
            if _REQUIRED_SCOPE not in token_scopes:
                logger.warning(
                    f"token.json has insufficient scopes {token_scopes}. "
                    "Deleting stale token — please re-run generate_oauth_token.py"
                )
                os.remove(token_path)
                creds = None
            elif creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                logger.info("Authenticated via OAuth2 (token.json) — token refreshed")
            else:
                logger.info("Authenticated via OAuth2 (token.json)")
        except Exception as e:
            logger.warning(f"OAuth2 auth failed, falling back to service account: {e}")
            creds = None

    # 2. Fallback to Service Account
    if not creds:
        creds_json = os.getenv("GOOGLE_DRIVE_CREDENTIALS_JSON")
        if not creds_json or not os.path.exists(creds_json):
            logger.error("No valid Google Drive credentials found (.env or token.json)")
            return None
        
        try:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                creds_json, scopes=["https://www.googleapis.com/auth/drive"]
            )
            logger.info(f"Authenticated via Service Account: {creds_json}")
        except Exception as e:
            logger.error(f"Failed to load service account: {e}")
            return None

    try:
        from googleapiclient.discovery import build
        _drive = build("drive", "v3", credentials=creds)
        return _drive
    except Exception as exc:
        logger.error(f"Failed to build Drive service: {exc}")
        return None


def is_configured() -> bool:
    """Return True if Drive credentials are available."""
    return _get_drive() is not None


# ── Public helpers ─────────────────────────────────────────────────────────────

def _get_folder_id(parent_id: str, folder_name: str) -> str:
    """Find or create a subfolder by name inside a parent folder."""
    drive = _get_drive()
    if not drive: return ""
    
    query = f"name = '{folder_name}' and '{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    try:
        results = drive.files().list(
            q=query, 
            fields="files(id)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        files = results.get("files", [])
        if files:
            return files[0]["id"]
        
        # Create it if not found
        meta = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id]
        }
        folder = drive.files().create(
            body=meta, 
            fields="id",
            supportsAllDrives=True
        ).execute()
        return folder.get("id")
    except Exception as exc:
        logger.error(f"Error getting/creating folder {folder_name}: {exc}")
        return ""


def upload_file(local_path: str, filename: str, mime_type: str = "application/octet-stream", subfolder_path: list = None) -> dict:
    """
    Upload a local file to a nested structure in Drive.
    """
    drive = _get_drive()
    if not drive:
        logger.error("Drive upload skipped: Service not initialized (check .env and credentials).")
        return {}

    root_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()
    if not root_id:
        logger.error("Drive upload skipped: GOOGLE_DRIVE_FOLDER_ID not found in environment.")
        return {}

    current_parent = root_id
    if subfolder_path:
        for folder_name in subfolder_path:
            logger.info(f"Resolving folder '{folder_name}' inside '{current_parent}'")
            current_parent = _get_folder_id(current_parent, folder_name)
            if not current_parent:
                logger.error(f"Failed to resolve folder path: {subfolder_path}")
                return {}

    try:
        from googleapiclient.http import MediaFileUpload
        meta    = {"name": filename, "parents": [current_parent]}
        media   = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)
        result  = drive.files().create(
            body=meta, 
            media_body=media, 
            fields="id, webViewLink",
            supportsAllDrives=True
        ).execute()

        file_id   = result.get("id", "")
        view_link = result.get("webViewLink", "")

        if file_id:
            drive.permissions().create(
                fileId=file_id,
                body={"type": "anyone", "role": "reader"},
                supportsAllDrives=True
            ).execute()
            if not view_link:
                view_link = f"https://drive.google.com/file/d/{file_id}/view"

        logger.info(f"✅ Drive Sync OK: {filename} uploaded to {current_parent}")
        return {"file_id": file_id, "view_link": view_link}
    except Exception as exc:
        logger.error(f"Drive upload failed for {filename}: {exc}")
        return {}


def list_folder_contents(folder_id: str):
    """List files and folders inside a given ID."""
    drive = _get_drive()
    if not drive: return []
    query = f"'{folder_id}' in parents and trashed = false"
    try:
        results = drive.files().list(
            q=query, 
            fields="files(id, name, mimeType, webViewLink)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        return results.get("files", [])
    except Exception as exc:
        logger.error(f"Drive list failed for {folder_id}: {exc}")
        return []


def delete_file(file_id: str) -> bool:
    """Delete a file from Drive by its file_id. Returns True on success."""
    if not file_id:
        return False
    drive = _get_drive()
    if not drive:
        return False
    try:
        drive.files().delete(
            fileId=file_id,
            supportsAllDrives=True
        ).execute()
        logger.info(f"Drive delete OK: {file_id}")
        return True
    except Exception as exc:
        logger.error(f"Drive delete failed for {file_id}: {exc}")
        return False


def get_mime_type(filename: str) -> str:
    """Guess MIME type from file extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {
        "pdf":  "application/pdf",
        "doc":  "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "txt":  "text/plain",
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "gif":  "image/gif",
        "webp": "image/webp",
        "svg":  "image/svg+xml",
        "mp4":  "video/mp4",
        "webm": "video/webm",
        "mov":  "video/quicktime",
        "mp3":  "audio/mpeg",
        "m4a":  "audio/mp4",
        "wav":  "audio/wav",
        "ogg":  "audio/ogg",
        "flac": "audio/flac",
    }.get(ext, "application/octet-stream")
