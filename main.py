import openai
import base64
import json
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import fitz  # PyMuPDF for PDF text extraction and image conversion

app = FastAPI(title="Invoice Data Extractor API")

# Models
TEXT_MODEL = "grok-4-1-fast-reasoning"
VISION_MODEL = "grok-2-vision-1212"

# Lazy-loaded client
_client = None

def get_client():
    """Get or create the xAI/Grok client (lazy initialization)."""
    global _client
    if _client is None:
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable is not set")
        _client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
    return _client


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: Optional[float] = None
    total: Optional[float] = None


class InvoiceData(BaseModel):
    company_name: str
    invoice_number: str
    items: list[LineItem]
    total_before_tax: float
    currency: str = "ILS"
    document_type: str  # invoice / shipping_certificate / receipt
    extraction_confidence: str  # high / medium / low


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text content from PDF."""
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_content = []

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text = page.get_text()
        if text.strip():
            text_content.append(f"--- Page {page_num + 1} ---\n{text}")

    pdf_document.close()
    return "\n\n".join(text_content)


def pdf_to_images(pdf_bytes: bytes) -> list[str]:
    """Convert PDF pages to base64 encoded images."""
    images = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        # Render page to image with good resolution
        mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        base64_image = base64.standard_b64encode(img_bytes).decode("utf-8")
        images.append(base64_image)

    pdf_document.close()
    return images


EXTRACTION_INSTRUCTIONS = """
1. **company_name**: The name of the company that SENT/ISSUED the invoice (the SELLER/VENDOR who is charging money).
   - This is NOT the customer/recipient that appears after "לכבוד:" (which means "To:")
   - The seller company usually appears at the TOP of the document, often with a logo
   - Look for "עוסק מורשה" (authorized dealer) number near the company name
   - Usually has "בע"מ" (Ltd) or "ח.פ." suffix
   - Examples: "קומפיוטר סי דאטה בע"מ", "מתקינט בע"מ", "טונוס ציוד מתכלה בע"מ", "אופוס"

2. **invoice_number**: The invoice/document number
   - Look for: "חשבונית מס", "מספר חשבונית", "תעודת משלוח", followed by a number

3. **items**: List of all line items with:
   - description: The item/service description (in Hebrew or English as appears)
   - quantity: The quantity (כמות)
   - unit_price: Price per unit if available (מחיר)
   - total: Total for this line item (סה"כ)

4. **total_before_tax**: The total amount BEFORE VAT/tax
   - Look for "סה"כ לפני מע"מ", "סה"כ חייב מע"מ"
   - If only total with VAT is shown, calculate: total / 1.18 (18% VAT in Israel)
   - Do NOT include the VAT amount

5. **document_type**: One of: "invoice", "shipping_certificate", "receipt"
   - "חשבונית מס" = invoice
   - "תעודת משלוח" = shipping_certificate
   - "חשבונית מס/קבלה" or "קבלה" = receipt

6. **extraction_confidence**: Your confidence level - "high", "medium", or "low"

IMPORTANT: Return ONLY a valid JSON object. No markdown, no explanation, no code blocks.
Use this exact structure:
{"company_name": "string", "invoice_number": "string", "items": [{"description": "string", "quantity": number, "unit_price": number, "total": number}], "total_before_tax": number, "currency": "ILS", "document_type": "invoice", "extraction_confidence": "high"}"""


def extract_with_text(text: str) -> dict:
    """Extract invoice data using text-based model."""
    prompt = f"""Analyze this invoice/receipt/shipping certificate text and extract the following information.
The document is in Hebrew. Here is the extracted text:

--- DOCUMENT TEXT ---
{text}
--- END DOCUMENT TEXT ---

Please extract:
{EXTRACTION_INSTRUCTIONS}"""

    response = get_client().chat.completions.create(
        model=TEXT_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def extract_with_vision(images: list[str]) -> dict:
    """Extract invoice data using vision model for scanned/image PDFs."""
    content = []
    for img_base64 in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
        })

    content.append({
        "type": "text",
        "text": f"""Analyze this invoice/receipt/shipping certificate image and extract the following information.
The document is in Hebrew.

Please extract:
{EXTRACTION_INSTRUCTIONS}"""
    })

    response = get_client().chat.completions.create(
        model=VISION_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": content}]
    )
    return response.choices[0].message.content.strip()


def extract_invoice_data(pdf_bytes: bytes) -> dict:
    """Extract invoice data from PDF. Uses text extraction first, falls back to vision."""

    # Try text extraction first
    text = extract_text_from_pdf(pdf_bytes)

    if text.strip() and len(text.strip()) > 50:
        # Use text-based extraction (faster and cheaper)
        print("Using text-based extraction...")
        response_text = extract_with_text(text)
    else:
        # Fall back to vision for scanned/image PDFs
        print("Text extraction failed, using vision model...")
        images = pdf_to_images(pdf_bytes)
        if not images:
            raise ValueError("Could not process PDF")
        response_text = extract_with_vision(images)

    # Clean up response if it contains markdown code blocks
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse response as JSON: {e}\nResponse: {response_text}")


@app.get("/")
async def root():
    return {"message": "Invoice Data Extractor API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/extract", response_model=InvoiceData)
async def extract_invoice(file: UploadFile = File(...)):
    """
    Extract invoice data from a PDF file.

    - **file**: PDF file to process

    Returns extracted invoice data including company name, invoice number,
    line items, and total before tax.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Read file content
        pdf_bytes = await file.read()

        # Extract data
        result = extract_invoice_data(pdf_bytes)

        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/extract-multiple")
async def extract_multiple_invoices(files: list[UploadFile] = File(...)):
    """
    Extract invoice data from multiple PDF files.

    - **files**: List of PDF files to process

    Returns a list of extracted invoice data for each file.
    """
    results = []

    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            results.append({
                "filename": file.filename,
                "error": "Only PDF files are supported",
                "data": None
            })
            continue

        try:
            pdf_bytes = await file.read()
            data = extract_invoice_data(pdf_bytes)
            results.append({
                "filename": file.filename,
                "error": None,
                "data": data
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "data": None
            })

    return JSONResponse(content={"results": results})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
