from fastapi import UploadFile
from app.services.commands.upload_document_command import UploadDocumentCommand
from app.services.commands.summarize_pdf_command import SummarizePDFCommand

class DocumentMapper:
    """Maps document DTOs and inputs to service commands."""
    
    @staticmethod
    async def map_to_upload_command(file: UploadFile) -> UploadDocumentCommand:
        content = await file.read()
        return UploadDocumentCommand(
            filename=file.filename or "unnamed.pdf",
            content=content
        )
        
    @staticmethod
    def map_to_summarize_command(document_name: str) -> SummarizePDFCommand:
        return SummarizePDFCommand(document_name=document_name)
