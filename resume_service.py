import PyPDF2
from io import BytesIO
from typing import Dict, Any
from backend.ai_service import ai_service

class ResumeService:
    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes) -> str:
        try:
            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    text += page_text + "\n"
            
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    @staticmethod
    async def analyze_resume(pdf_bytes: bytes, target_role: str = "Software Engineer") -> Dict[str, Any]:
        text = ResumeService.extract_text_from_pdf(pdf_bytes)
        
        if not text:
            raise Exception("Could not extract text from PDF")
        
        analysis = await ai_service.analyze_resume(text, target_role)
        analysis["extracted_text"] = text
        
        return analysis

resume_service = ResumeService()