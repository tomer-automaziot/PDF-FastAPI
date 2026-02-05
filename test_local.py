#!/usr/bin/env python3
"""Local test script for the invoice extractor."""

import os
import sys

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import extract_invoice_data, extract_text_from_pdf
import json

def test_pdf(pdf_path: str):
    """Test extraction on a single PDF."""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(pdf_path)}")
    print('='*60)

    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()

    # Show extracted text (first 500 chars)
    text = extract_text_from_pdf(pdf_bytes)
    print(f"\nExtracted text preview ({len(text)} chars total):")
    print(text[:800] + "..." if len(text) > 800 else text)

    # Extract data
    print("\n--- Calling Grok API ---")
    try:
        result = extract_invoice_data(pdf_bytes)
        print("\nExtracted Data:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("XAI_API_KEY"):
        print("ERROR: XAI_API_KEY environment variable not set")
        print("Run: export XAI_API_KEY=your_key_here")
        sys.exit(1)

    # Test directory
    pdf_dir = "/Users/tomerbenami/Documents/automziot/Marlen/PDF_file_samples"

    # Test specific file or all files
    if len(sys.argv) > 1:
        test_pdf(sys.argv[1])
    else:
        # Test first PDF as example
        test_file = os.path.join(pdf_dir, "Invoice311760092.pdf")
        if os.path.exists(test_file):
            test_pdf(test_file)
        else:
            print(f"Test file not found: {test_file}")
