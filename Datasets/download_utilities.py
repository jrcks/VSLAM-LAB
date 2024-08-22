from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from io import BytesIO


def create_google_drive_service(url_download_root, client_secrets_file):
    flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, url_download_root)
    creds = flow.run_local_server(port=0)
    service = build('drive', 'v3', credentials=creds)
    return service


def download_file_from_google_drive(file_id, output_file, google_drive_service):
    request = google_drive_service.files().get_media(fileId=file_id)
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fd=fh, request=request)
    done = False

    while not done:
        status, done = downloader.next_chunk()
        print("    Download progress" + f"'{output_file}'" + " : {0} %".format(int(status.progress() * 100)))
    fh.seek(0)

    with open(output_file, 'wb') as f:
        f.write(fh.read())
        f.close()
