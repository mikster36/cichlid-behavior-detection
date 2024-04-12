import subprocess
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Dropbox folder path
DROPBOX_ROOT_FOLDER_PATH = 'dropbox:/DLC annotations/behavior_analysis_output/Bower-circling'
# TESTING_DIRECTORIES = ['MC_singlenuc24_4_Tk47_030320', 'MC_singlenuc37_2_Tk17_030320']

def list_directories_in_folder(dropbox_folder_path):
    # Execute rclone lsd command to list directories in the specified folder
    result = subprocess.run(['rclone', 'lsd', dropbox_folder_path], capture_output=True, text=True)
    if result.returncode == 0:
        # Parse directory names from the output
        directories = [line.split()[4] for line in result.stdout.splitlines()]
        print('Directories successfully listed')
        return directories
    else:
        print("Error:", result.stderr)
        return []

def list_files_in_folder(dropbox_folder_path):
    # Execute rclone lsf command to list files in the specified directory
    result = subprocess.run(['rclone', 'lsf', dropbox_folder_path], capture_output=True, text=True)
    if result.returncode == 0:
        # Parse filenames from the output
        files = result.stdout.splitlines()
        print('Files successfully listed')
        return files
    else:
        print("Error:", result.stderr)
        return []

def create_labeled_dict(root_folder_path):
    labels = {}

    # List directories within the root folder
    directories = list_directories_in_folder(root_folder_path)
    
    # Iterate through each directory and list files
    for directory in directories:
        labels[directory] = {}
        for subfolder in ['/false_positives', '/true_positives']:
            directory_path = root_folder_path + "/" + directory + subfolder
            files_list = list_files_in_folder(directory_path)
            labels[directory][subfolder[1:]] = files_list
    
    print('Labeled dictionary successfully created')
    return labels

def write_to_sheets(spreadsheet, sheet, df):
    # Define the scope and credentials
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('charest_credentials.json', scope)

    # Authorize the client using the credentials
    client = gspread.authorize(credentials)

    # Open the spreadsheet 
    spreadsheet = client.open(spreadsheet)

    try:
        worksheet = spreadsheet.worksheet(sheet)
    except gspread.WorksheetNotFound:
        # If worksheet is not found, create a new one
        worksheet = spreadsheet.add_worksheet(title=sheet, rows=1000, cols=26)

    # Convert DataFrame to a list of lists for gspread
    data_to_write = df.values.tolist()

    # Specify the range where you want to write the data (start from cell A1)
    cell_range = f'B4:E1000'

    # Write data to the specified range
    worksheet.update(cell_range, data_to_write)

    print("Data has been written to Google Sheets successfully")

if __name__ == "__main__":
    labels = create_labeled_dict(DROPBOX_ROOT_FOLDER_PATH)
    
    # iterate through trials, ex: "MC_singlenuc24_4_Tk47_030320" 
    for trial in labels:
        trial_data = []
        # iteratate through trial classification, ex: "false_postives"
        for classification in labels[trial]:
            for clip in labels[trial][classification]:
                start, end = clip.split('-')
                end = end[:-4]
                start_minute, end_minute = float(start.split(':')[1]), float(end.split(':')[1])
                start_second, end_second = float(start.split(':')[2]) + (start_minute * 60), float(end.split(':')[2]) + (end_minute * 60)
                clip_length = round(end_second - start_second, 3)
                clip_data = [start, end, clip_length, classification]
                trial_data.append(clip_data)
        
        df = pd.DataFrame(trial_data, columns=['Clip Start Frame', 'Clip End Frame', 'Clip Length (s)', 'Classification'])
        write_to_sheets("FS Behavior Analysis", trial, df)