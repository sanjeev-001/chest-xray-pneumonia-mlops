"""
Data Pipeline Service Main Module
Handles data ingestion, validation, preprocessing, and versioning
"""

import logging
import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .ingestion import DataIngestionManager, DatasetInfo, IngestionResult
from .preprocessing import ImagePreprocessor, PreprocessingConfig, ProcessingResult
from .validation import DataValidator, ValidationConfig, ValidationResult, DatasetValidationSummary
from .versioning import DataVersionController, DataVersion, StorageConfig
from .storage import create_storage_backend, DataStorageManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Pipeline Service",
    description="Handles data ingestion, validation, preprocessing, and versioning",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
data_ingestion_manager = DataIngestionManager()
image_preprocessor = ImagePreprocessor()
data_validator = DataValidator()

# Initialize versioning and storage (configure based on environment)
storage_config = None
if os.getenv("STORAGE_BACKEND"):
    storage_config = StorageConfig(
        backend_type=os.getenv("STORAGE_BACKEND", "local"),
        endpoint_url=os.getenv("STORAGE_ENDPOINT_URL"),
        access_key=os.getenv("STORAGE_ACCESS_KEY"),
        secret_key=os.getenv("STORAGE_SECRET_KEY"),
        bucket_name=os.getenv("STORAGE_BUCKET_NAME", "mlops-data"),
        region=os.getenv("STORAGE_REGION", "us-east-1")
    )

data_version_controller = DataVersionController(storage_config=storage_config)

# Initialize storage manager if backend is configured
storage_manager = None
if storage_config:
    try:
        storage_backend = create_storage_backend(
            storage_config.backend_type,
            bucket_name=storage_config.bucket_name,
            access_key=storage_config.access_key,
            secret_key=storage_config.secret_key,
            endpoint_url=storage_config.endpoint_url,
            region=storage_config.region
        )
        storage_manager = DataStorageManager(storage_backend)
    except Exception as e:
        logger.warning(f"Failed to initialize storage backend: {e}")
        storage_manager = None


# Pydantic models for API
class DatasetInfoModel(BaseModel):
    name: str
    url: str
    expected_size: Optional[int] = None
    checksum: Optional[str] = None
    format: str = "zip"


class IngestionRequest(BaseModel):
    dataset_info: DatasetInfoModel
    force_redownload: bool = False


class ValidationRequest(BaseModel):
    dataset_path: str


class PreprocessingRequest(BaseModel):
    image_paths: List[str]
    return_tensors: bool = False


class DataVersionRequest(BaseModel):
    dataset_path: str
    dataset_name: str
    split_ratios: List[float] = [0.7, 0.15, 0.15]
    stratify_by_class: bool = True
    metadata: Optional[dict] = None


class StorageRequest(BaseModel):
    version_id: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "data-pipeline"}


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    return {"status": "ready", "service": "data-pipeline"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Data Pipeline Service", "version": "0.1.0"}


@app.get("/datasets")
async def list_datasets():
    """List all available datasets"""
    try:
        datasets = data_ingestion_manager.list_available_datasets()
        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets/{dataset_name}")
async def get_dataset_info(dataset_name: str):
    """Get information about a specific dataset"""
    try:
        info = data_ingestion_manager.get_dataset_info(dataset_name)
        return info
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/download")
async def download_dataset(request: IngestionRequest):
    """Download a dataset from URL"""
    try:
        dataset_info = DatasetInfo(
            name=request.dataset_info.name,
            url=request.dataset_info.url,
            expected_size=request.dataset_info.expected_size,
            checksum=request.dataset_info.checksum,
            format=request.dataset_info.format
        )
        
        dataset_path = data_ingestion_manager.download_dataset(
            dataset_info, 
            force_redownload=request.force_redownload
        )
        
        return {
            "success": True,
            "dataset_path": str(dataset_path),
            "message": f"Dataset {dataset_info.name} downloaded successfully"
        }
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/local")
async def ingest_local_dataset(source_path: str, dataset_name: str):
    """Ingest a dataset from local filesystem"""
    try:
        result = data_ingestion_manager.ingest_local_dataset(source_path, dataset_name)
        
        return {
            "success": result.success,
            "dataset_path": str(result.dataset_path),
            "total_files": result.total_files,
            "valid_files": result.valid_files,
            "invalid_files": result.invalid_files,
            "errors": result.errors,
            "metadata": result.metadata
        }
    except Exception as e:
        logger.error(f"Error ingesting local dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate/dataset")
async def validate_dataset(request: ValidationRequest):
    """Validate an entire dataset"""
    try:
        summary = data_validator.validate_dataset(request.dataset_path)
        
        return {
            "total_files": summary.total_files,
            "valid_files": summary.valid_files,
            "invalid_files": summary.invalid_files,
            "common_issues": summary.common_issues,
            "dataset_statistics": summary.dataset_statistics,
            "recommendations": summary.recommendations
        }
    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate/image")
async def validate_image(image_path: str):
    """Validate a single image"""
    try:
        result = data_validator.validate_image(image_path)
        
        return {
            "is_valid": result.is_valid,
            "file_path": str(result.file_path),
            "file_size_bytes": result.file_size_bytes,
            "image_dimensions": result.image_dimensions,
            "format": result.format,
            "brightness_score": result.brightness_score,
            "contrast_score": result.contrast_score,
            "blur_score": result.blur_score,
            "issues": result.issues,
            "warnings": result.warnings
        }
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess/batch")
async def preprocess_batch(request: PreprocessingRequest):
    """Preprocess a batch of images"""
    try:
        results = image_preprocessor.preprocess_batch(
            request.image_paths,
            return_tensors=request.return_tensors
        )
        
        # Convert results to serializable format
        serialized_results = []
        for result in results:
            serialized_result = {
                "success": result.success,
                "original_shape": result.original_shape,
                "final_shape": result.final_shape,
                "processing_steps": result.processing_steps,
                "error_message": result.error_message
            }
            # Note: processed_image is not serialized due to size
            serialized_results.append(serialized_result)
        
        return {
            "results": serialized_results,
            "total_processed": len(results),
            "successful": sum(1 for r in results if r.success)
        }
    except Exception as e:
        logger.error(f"Error preprocessing batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preprocess/image")
async def preprocess_image(image_path: str, return_tensor: bool = False):
    """Preprocess a single image"""
    try:
        result = image_preprocessor.preprocess_image(image_path, return_tensor=return_tensor)
        
        return {
            "success": result.success,
            "original_shape": result.original_shape,
            "final_shape": result.final_shape,
            "processing_steps": result.processing_steps,
            "error_message": result.error_message
        }
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/preprocess/statistics")
async def calculate_dataset_statistics(dataset_path: str):
    """Calculate dataset statistics for normalization"""
    try:
        # Find all image files in dataset
        dataset_path = Path(dataset_path)
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            image_files.extend(dataset_path.rglob(f"*{ext}"))
            image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
        
        if not image_files:
            raise HTTPException(status_code=404, detail="No image files found in dataset")
        
        # Calculate statistics
        stats = image_preprocessor.calculate_statistics(image_files)
        
        return stats
    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/version/create")
async def create_data_version(request: DataVersionRequest):
    """Create a new data version with train/val/test splits"""
    try:
        data_version = data_version_controller.create_data_version(
            dataset_path=request.dataset_path,
            dataset_name=request.dataset_name,
            split_ratios=tuple(request.split_ratios),
            stratify_by_class=request.stratify_by_class,
            metadata=request.metadata
        )
        
        return {
            "version_id": data_version.version_id,
            "timestamp": data_version.timestamp,
            "dataset_name": data_version.dataset_name,
            "total_samples": data_version.total_samples,
            "splits": {
                name: {
                    "total_samples": split.total_samples,
                    "class_distribution": split.class_distribution
                }
                for name, split in data_version.splits.items()
            },
            "storage_path": data_version.storage_path
        }
    except Exception as e:
        logger.error(f"Error creating data version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/version/list")
async def list_data_versions(dataset_name: Optional[str] = None):
    """List all data versions, optionally filtered by dataset name"""
    try:
        versions = data_version_controller.list_versions(dataset_name)
        
        return {
            "versions": [
                {
                    "version_id": v.version_id,
                    "timestamp": v.timestamp,
                    "dataset_name": v.dataset_name,
                    "total_samples": v.total_samples,
                    "splits": {
                        name: split.total_samples
                        for name, split in v.splits.items()
                    }
                }
                for v in versions
            ]
        }
    except Exception as e:
        logger.error(f"Error listing data versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/version/{version_id}")
async def get_data_version(version_id: str):
    """Get detailed information about a specific data version"""
    try:
        version = data_version_controller.get_version(version_id)
        if not version:
            raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
        
        return {
            "version_id": version.version_id,
            "timestamp": version.timestamp,
            "dataset_name": version.dataset_name,
            "total_samples": version.total_samples,
            "splits": {
                name: {
                    "total_samples": split.total_samples,
                    "class_distribution": split.class_distribution,
                    "file_count": len(split.file_paths)
                }
                for name, split in version.splits.items()
            },
            "metadata": version.metadata,
            "storage_path": version.storage_path
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/version/{version_id}/split/{split_name}")
async def get_split_paths(version_id: str, split_name: str):
    """Get file paths for a specific split of a data version"""
    try:
        paths = data_version_controller.get_split_paths(version_id, split_name)
        
        return {
            "version_id": version_id,
            "split_name": split_name,
            "file_paths": [str(p) for p in paths],
            "total_files": len(paths)
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting split paths: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/version/{version_id}")
async def delete_data_version(version_id: str, remove_from_storage: bool = False):
    """Delete a data version"""
    try:
        data_version_controller.delete_version(version_id, remove_from_storage)
        return {"message": f"Successfully deleted version {version_id}"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
    except Exception as e:
        logger.error(f"Error deleting data version: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/storage/push")
async def push_to_storage(request: StorageRequest):
    """Push a data version to remote storage"""
    if not storage_manager:
        raise HTTPException(status_code=501, detail="Storage backend not configured")
    
    try:
        data_version_controller.push_to_storage(request.version_id)
        return {"message": f"Successfully pushed version {request.version_id} to storage"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version {request.version_id} not found")
    except Exception as e:
        logger.error(f"Error pushing to storage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/storage/pull")
async def pull_from_storage(request: StorageRequest):
    """Pull a data version from remote storage"""
    if not storage_manager:
        raise HTTPException(status_code=501, detail="Storage backend not configured")
    
    try:
        data_version_controller.pull_from_storage(request.version_id)
        return {"message": f"Successfully pulled version {request.version_id} from storage"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Version {request.version_id} not found")
    except Exception as e:
        logger.error(f"Error pulling from storage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/storage/versions")
async def list_storage_versions():
    """List all dataset versions available in remote storage"""
    if not storage_manager:
        raise HTTPException(status_code=501, detail="Storage backend not configured")
    
    try:
        versions = storage_manager.list_dataset_versions()
        return {"versions": versions}
    except Exception as e:
        logger.error(f"Error listing storage versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)