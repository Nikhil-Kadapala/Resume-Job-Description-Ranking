import os
import argparse
import json
from together import Together
from together.utils import check_file

try:
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
except Exception as e:
    print(f"Error: {e}")
    exit()

def check_and_upload_files(filepath:str):
    """
    Check the file format and upload the training data to Together.
    """
    client = Together(api_key=TOGETHER_API_KEY)
    try:
        sft_report = check_file(filepath)
        print(json.dumps(sft_report, indent=4))
        #assert sft_report["is_check_passed"] == True
    except ValueError as e:
        print(f"Error: {e}")
        exit()

    # Upload the data to Together
    train_file_resp = client.files.upload(filepath, check=True)
    print(train_file_resp.id)  # Save this ID for starting your fine-tuning job

def main():
    parser = argparse.ArgumentParser(description="Upload files to Together for fine-tuning.")
    parser.add_argument("-f", "--file", type=str, help="Path to the file to be uploaded.")
    args = parser.parse_args()

    # Check if the file exists
    if not os.path.exists(args.file):
        print(f"File {args.file} does not exist.")
        return
    
    # Upload the file
    check_and_upload_files(args.file)

if __name__ == "__main__":
    main()