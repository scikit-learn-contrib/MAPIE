import os
import re

def fix_docstrings(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    fixed_lines = []
    bullet_list_active = False
    in_directive = False
    directive_indent = 0

    for line in lines:
        stripped_line = line.strip()

        if re.match(r'^- ', stripped_line):  # Detect bullet points
            bullet_list_active = True
            indent_level = len(line) - len(line.lstrip())
        elif bullet_list_active and (stripped_line == "" or not stripped_line.startswith('- ')):
            fixed_lines.append('\n')  # Ensure a blank line after a list
            bullet_list_active = False
        elif re.match(r'^\.\. \w+::', stripped_line):  # Check for directive start
            in_directive = True
            directive_indent = len(line) - len(line.lstrip())
        elif in_directive and (len(line) - len(line.lstrip())) <= directive_indent:
            in_directive = False

        if 'unexpected indentation' in line or (in_directive and not stripped_line.startswith('   ')):
            continue  # Skip adding lines with unexpected indentation

        if 'undefined substitution referenced' in line:
            line = '#' + line  # Comment out lines with undefined substitutions

        fixed_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(fixed_lines)

def fix_rst_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.rst'):
                fix_docstrings(os.path.join(root, file))

# Example usage:
fix_rst_files('/Users/llacombe/CODE/quantmetry/MAPIE/doc')
