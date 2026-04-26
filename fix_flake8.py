import os
import re

def wrap_line(line, indent):
    # simple wrap
    words = line.strip().split()
    lines = []
    current_line = []
    current_len = len(indent)
    for word in words:
        if current_len + len(word) + 1 > 79:
            lines.append(indent + ' '.join(current_line))
            current_line = [word]
            current_len = len(indent) + len(word)
        else:
            current_line.append(word)
            current_len += len(word) + 1
    if current_line:
        lines.append(indent + ' '.join(current_line))
    return '\n'.join(lines) + '\n'

files_to_fix = [
    'kiori/__init__.py',
    'kiori/agent.py',
    'kiori/executor.py',
    'kiori/memory.py',
    'kiori/models.py',
    'kiori/router.py',
]

for filepath in files_to_fix:
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    with open(filepath, 'w') as f:
        for i, line in enumerate(lines):
            if len(line) > 79:
                indent_match = re.match(r'^(\s*)', line)
                indent = indent_match.group(1) if indent_match else ''
                
                # If it's a docstring text line (not containing """)
                if '"""' not in line and not line.strip().startswith(('def', 'class', 'import', 'from', 'return', 'if', 'elif', 'else')):
                    # Check if it's within a docstring (simple heuristic: starts with uppercase and ends with . or contains args)
                    line = wrap_line(line, indent)
            f.write(line)
