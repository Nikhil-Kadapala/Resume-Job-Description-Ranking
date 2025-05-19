import os
import argparse
from typing import Dict
from markitdown import MarkItDown

def parse_resume(file_path: str) -> Dict:
    """ 
    Parses a Resume file and converts it to Markdown format.
    
    Args: 
        file_path (str): The path to the resume file.

    Returns: A string containing the Markdown content of the resume.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Initialize MarkItDown with plugins disabled
    md = MarkItDown(enable_plugins=False)  # Set to True to enable plugins
    file_content: str = None

    try:
        # Convert the file content to Markdown
        result = md.convert(file_path)
        file_content = result.text_content
    except Exception as e:
        print(f"Error converting file {file_path}: {e}")
    
    return file_content

def main():
    """
    Main function to parse a resume file and print the Markdown content.
    """
    parser = argparse.ArgumentParser(description="Parse a resume file and print the Markdown content.")
    parser.add_argument("file_path", type=str, help="The path to the resume file.")
    parser.add_argument("-o", "--output", type=str, help="The path to the output file.")
    args = parser.parse_args()

    markdown_content = parse_resume(args.file_path)
    if args.output:
        with open(args.output, "w") as f:
            f.write(markdown_content)
    else:
        print(markdown_content)

if __name__ == "__main__":
    main()