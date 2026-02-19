from app.controller.dto import UploadResponse, DocumentsResponse, DocumentInfo, DocumentStatus
from fastapi import UploadFile, File, HTTPException
from app.service.pre_analyze_document_service import pre_analyze_document_service
from datetime import datetime
from pathlib import Path
import logging
import traceback

logger = logging.getLogger('app.service')

class DocumentService:
    async def upload_document(self, file: UploadFile = File(description="PDF file to upload")) -> UploadResponse:
        """Upload a PDF document and trigger pre-analysis"""
        try:
            # Validate file type
            if not file.filename:
                error_msg = "No filename provided"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
            
            if not file.filename.endswith('.pdf'):
                error_msg = f"Invalid file type: {file.filename}. Only PDF files are allowed"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            
            # Generate timestamp-based folder name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            doc_name = f"{file.filename.replace('.pdf', '')}_{timestamp}"
            doc_folder = Path(f"0_data/{doc_name}")
            
            logger.info(f"Starting upload for document: {doc_name}")
            print(f"\n📄 Uploading document: {doc_name}")
            
            # Create document folder
            try:
                doc_folder.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created document folder: {doc_folder}")
            except Exception as e:
                error_msg = f"Failed to create document folder {doc_folder}: {str(e)}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                print(f"Stack trace: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Error creating document folder: {str(e)}")
            
            # Save the PDF file
            pdf_path = doc_folder / f"{doc_name}.pdf"
            try:
                content = await file.read()
                with open(pdf_path, "wb") as buffer:
                    buffer.write(content)
                logger.info(f"Successfully saved PDF file: {pdf_path}")
                print(f"✓ Successfully saved PDF file")
            except Exception as e:
                error_msg = f"Failed to save PDF file {pdf_path}: {str(e)}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                print(f"Stack trace: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
            
            # Run pre-analysis in the background
            def run_analysis():
                try:
                    logger.info(f"Starting background pre-analysis for: {doc_name}")
                    print(f"🔍 Starting background analysis for: {doc_name}")
                    pre_analyze_document_service.pre_analyze_document(str(pdf_path), doc_name)
                    logger.info(f"Background pre-analysis completed successfully for: {doc_name}")
                    print(f"✅ Background analysis completed for: {doc_name}")
                except Exception as e:
                    error_msg = f"Error during pre-analysis for {doc_name}: {str(e)}"
                    logger.error(error_msg)
                    print(f"❌ {error_msg}")
                    print(f"Stack trace: {traceback.format_exc()}")
            
            # Start analysis in background
            import threading
            try:
                analysis_thread = threading.Thread(target=run_analysis)
                analysis_thread.start()
                logger.info(f"Background analysis thread started for: {doc_name}")
                print(f"✓ Background analysis thread started")
            except Exception as e:
                error_msg = f"Failed to start background analysis thread: {str(e)}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                # Don't fail the upload if background thread fails to start
            
            logger.info(f"Document upload successful: {doc_name}")
            print(f"✅ Document uploaded successfully: {doc_name}\n")
            
            return UploadResponse(
                message="Document uploaded successfully and analysis started",
                document_name=doc_name,
                status=DocumentStatus.ANALYZING
            )
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            error_msg = f"Unexpected error during document upload: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


    async def list_documents(self) -> DocumentsResponse:
        """Get list of all available documents"""
        try:
            data_dir = Path("0_data")
            documents = []
            
            if not data_dir.exists():
                logger.info("Data directory does not exist, returning empty list")
                return DocumentsResponse(documents=[])
            
            logger.info("Scanning for documents")
            
            for item in data_dir.iterdir():
                try:
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
                except Exception as e:
                    error_msg = f"Error processing document folder {item}: {str(e)}"
                    logger.warning(error_msg)
                    print(f"⚠️  {error_msg}")
                    # Continue with other documents
                    continue
            
            logger.info(f"Found {len(documents)} documents")
            return DocumentsResponse(documents=documents)
            
        except Exception as e:
            error_msg = f"Error listing documents: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=error_msg)
    
document_service = DocumentService()