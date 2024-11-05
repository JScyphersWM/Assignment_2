import os
import re

def clean_code(code):
    # Remove single-line comments
    code = re.sub(r'#.*', '', code)
    
    # Remove multi-line comments
    code = re.sub(r'\'\'\'[\s\S]*?\'\'\'|"""[\s\S]*?"""', '', code)
    
    # Remove special characters
    code = re.sub(r'[^\w\s\(\):,\.\[\]\'\"]', '', code)
    return code

def extract_functions(code):
    functions = re.findall(r'^\s*def .+?:\n(?:\s{4}.*(?:\n|$))*(?:\s{4}.*(?:\n|$))*', code, re.MULTILINE)
    return functions

def condense_python_files(input_directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(input_directory):
            if filename.endswith(".py"):
                filepath = os.path.join(input_directory, filename)
                
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
                    code = infile.read()
                    functions = extract_functions(code)
                    
                    for function in functions:
                        cleaned_function = clean_code(function)
                        outfile.write(cleaned_function + "\n\n")

input_directory = 'input'
output_file = 'condensed_files.txt'
condense_python_files(input_directory, output_file)
