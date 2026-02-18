from app.controller.dto import UploadResponse, DocumentsResponse, DocumentInfo, DocumentStatus
from fastapi import UploadFile, File, HTTPException
from app.service.pre_analyze_document_service import pre_analyze_document_service
from datetime import datetime
from pathlib import Path

class DocumentService:
    async def upload_document(self, file: UploadFile = File(description="PDF file to upload")) -> UploadResponse:
        """Upload a PDF document and trigger pre-analysis"""
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Generate timestamp-based folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_name = f"{file.filename.replace('.pdf', '')}_{timestamp}"
        doc_folder = Path(f"0_data/{doc_name}")
        
        # Create document folder
        doc_folder.mkdir(parents=True, exist_ok=True)
        
        # Save the PDF file
        pdf_path = doc_folder / f"{doc_name}.pdf"
        try:
            with open(pdf_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Run pre-analysis in the background
            def run_analysis():
                try:
                    pre_analyze_document_service.pre_analyze_document(str(pdf_path), doc_name)
                except Exception as e:
                    print(f"Error during pre-analysis: {e}")
            
            # Start analysis in background
            import threading
            analysis_thread = threading.Thread(target=run_analysis)
            analysis_thread.start()
            
            return UploadResponse(
                message="Document uploaded successfully and analysis started",
                document_name=doc_name,
                status=DocumentStatus.ANALYZING
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


    async def list_documents(self) -> DocumentsResponse:
        """Get list of all available documents"""
        data_dir = Path("0_data")
        documents = []
        
        if not data_dir.exists():
            return DocumentsResponse(documents=[])
        
        for item in data_dir.iterdir():
            if item.is_dir():
                # Check if required analysis files exist
                doc_name = item.name
                chunks_file = item / f"{doc_name}_chunks.json"
                embeddings_file = item / f"{doc_name}_chunk_embeddings.json"
                
                status = DocumentStatus.READY if chunks_file.exists() and embeddings_file.exists() else DocumentStatus.PROCESSING
                
                documents.append(DocumentInfo(
                    name=doc_name,
                    status=status,
                    path=str(item)
                ))
        
        return DocumentsResponse(documents=documents)
    
document_service = DocumentService()