import re
import csv

input_file = 'condensed_files.txt'
output_file = 'processed_files.csv'

def extract_functions(file_content):
    functions_with_if = []
    functions_without_if = []
    lines = file_content.splitlines()
    current_function = []
    inside_function = False

    for line in lines:
        if line.strip().startswith("def ") and line.strip().endswith(":"):
            if current_function:
                function_code = "\n".join(current_function)
                has_if = 'if ' in function_code
                if has_if:
                    functions_with_if.append(function_code)
                else:
                    functions_without_if.append(function_code)
                current_function = []  

            inside_function = True  
            current_function.append(line)  
        elif inside_function and (line.startswith(" ") or line.startswith("\t")):
            current_function.append(line)
        elif inside_function and line.strip() == "":
            current_function.append(line)
        else:
            inside_function = False

    if current_function:
        function_code = "\n".join(current_function)
        has_if = 'if ' in function_code
        if has_if:
            functions_with_if.append(function_code)
        else:
            functions_without_if.append(function_code)

    return functions_with_if, functions_without_if

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    functions_with_if, functions_without_if = extract_functions(content)
    count_with_if = len(functions_with_if)
    count_without_if = len(functions_without_if)
    
    all_functions = []
    for function in functions_with_if:
        all_functions.append({"function_code": function, "has_if": True})
    for function in functions_without_if:
        all_functions.append({"function_code": function, "has_if": False})
    

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['function_code', 'has_if']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_functions)
    
    #print(f"Total functions with if statements: {count_with_if}")
    #print(f"Total functions without if statements: {count_without_if}")

process_file(input_file, output_file)


