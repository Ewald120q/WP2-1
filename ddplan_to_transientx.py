#!/usr/bin/env python3
"""
Quick script to convert DDplan output to TransientX format
Usage: python ddplan_to_transientx.py [filterbank_file] > output.txt
"""

import sys
import subprocess
import tempfile
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python ddplan_to_transientx.py [filterbank_file]")
        sys.exit(1)
    
    filterbank_file = sys.argv[1]
    
    # Run DDplan_modified.py with TransientX output
    cmd = [
        "python", "DDplan_modified.py", 
        "--transientx", 
        "--output", "/dev/null",  # Suppress plot output
        filterbank_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Extract only the TransientX format lines
        lines = result.stdout.split('\n')
        in_transientx_section = False
        
        for line in lines:
            if line.strip() == "# TransientX ddplan format:":
                in_transientx_section = True
                print(line)
            elif in_transientx_section and line.strip().startswith('#'):
                print(line)
            elif in_transientx_section and line.strip() and not line.strip().startswith('Type'):
                if any(char.isdigit() for char in line):
                    print(line)
            elif in_transientx_section and not line.strip():
                break
                
    except Exception as e:
        print(f"Error running DDplan: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()