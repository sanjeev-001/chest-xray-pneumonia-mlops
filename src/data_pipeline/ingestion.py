"""
Data Ingestion Manager for chest X-ray datasets
Handles downloading and initial validation of medical image datasets
"""

import os
import logging
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a dataset"""
    name: str
    url: str
    expected_size: Optional[int] = None
    checksum: Optional[str] = None
    format: str = "zip"  # zip, tar, tar.gz, directory


@dataclass
class IngestionResult:
    """Result of data ingestion process"""
    success: bool
    dataset_path: Path
    total_files: int
    valid_files: int
    invalid_files: int
    errors: List[str]
    metadata: Dict


class DataIngestionManager:
    """Manages downloading and initial validation of chest X-ray datasets"""
    
    def __init__(self, base_data_dir: str = "data/raw"):
        self.base_data_dir = Path(base_data_dir)
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Common chest X-ray datasets
        self.known_datasets = {
            "chest_xray_pneumonia": DatasetInfo(
                name="chest_xray_pneumonia",
                url="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
                format="zip"
            ),
            "nih_chest_xray": DatasetInfo(
                name="nih_chest_xray", 
                url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
                format="tar.gz"
            )
        }
    
    def download_dataset(self, dataset_info: DatasetInfo, force_redownload: bool = False) -> Path:
        """
        Download a dataset from URL
        
        Args:
            dataset_info: Information about the dataset to download
            force_redownload: Whether to redownload if already exists
            
        Returns:
            Path to downloaded dataset directory
        """
        dataset_dir = self.base_data_dir / dataset_info.name
        
        if dataset_dir.exists() and not force_redownload:
            logger.info(f"Dataset {dataset_info.name} already exists at {dataset_dir}")
            return dataset_dir
        
        logger.info(f"Downloading dataset {dataset_info.name} from {dataset_info.url}")
        
        # Create dataset directory
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename from URL
        parsed_url = urlparse(dataset_info.url)
        filename = Path(parsed_url.path).name
        if not filename:
            filename = f"{dataset_info.name}.{dataset_info.format}"
        
        download_path = dataset_dir / filename
        
        try:
            # Download file with progress tracking
            response = requests.get(dataset_info.url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(download_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {progress:.1f}%")
            
            # Verify checksum if provided
            if dataset_info.checksum:
                if not self._verify_checksum(download_path, dataset_info.checksum):
                    raise ValueError("Checksum verification failed")
            
            # Extract if it's an archive
            if dataset_info.format in ['zip', 'tar', 'tar.gz']:
                self._extract_archive(download_path, dataset_dir)
                # Remove archive after extraction
                download_path.unlink()
            
            logger.info(f"Successfully downloaded and extracted {dataset_info.name}")
            return dataset_dir
            
        except Exception as e:
            logger.error(f"Failed to download dataset {dataset_info.name}: {str(e)}")
            raise
    
    def ingest_local_dataset(self, source_path: str, dataset_name: str) -> IngestionResult:
        """
        Ingest a dataset from local filesystem
        
        Args:
            source_path: Path to local dataset
            dataset_name: Name for the dataset
            
        Returns:
            IngestionResult with ingestion statistics
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")
        
        dataset_dir = self.base_data_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Ingesting local dataset from {source_path} to {dataset_dir}")
        
        total_files = 0
        valid_files = 0
        invalid_files = 0
        errors = []
        
        # Copy and validate files
        for file_path in source_path.rglob("*"):
            if file_path.is_file():
                total_files += 1
                
                # Check if it's a supported image format
                if file_path.suffix.lower() in self.supported_formats:
                    try:
                        # Create relative path structure
                        rel_path = file_path.relative_to(source_path)
                        dest_path = dataset_dir / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Copy file
                        import shutil
                        shutil.copy2(file_path, dest_path)
                        valid_files += 1
                        
                    except Exception as e:
                        invalid_files += 1
                        errors.append(f"Failed to copy {file_path}: {str(e)}")
                else:
                    invalid_files += 1
                    errors.append(f"Unsupported format: {file_path}")
        
        metadata = {
            "source_path": str(source_path),
            "ingestion_timestamp": str(Path().cwd()),
            "supported_formats": list(self.supported_formats)
        }
        
        return IngestionResult(
            success=valid_files > 0,
            dataset_path=dataset_dir,
            total_files=total_files,
            valid_files=valid_files,
            invalid_files=invalid_files,
            errors=errors,
            metadata=metadata
        )
    
    def list_available_datasets(self) -> List[str]:
        """List all available datasets in the data directory"""
        if not self.base_data_dir.exists():
            return []
        
        return [d.name for d in self.base_data_dir.iterdir() if d.is_dir()]
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a specific dataset"""
        dataset_dir = self.base_data_dir / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found")
        
        # Count files by category
        image_files = []
        other_files = []
        
        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() in self.supported_formats:
                    image_files.append(file_path)
                else:
                    other_files.append(file_path)
        
        return {
            "name": dataset_name,
            "path": str(dataset_dir),
            "total_images": len(image_files),
            "other_files": len(other_files),
            "supported_formats": list(self.supported_formats),
            "directory_structure": self._get_directory_structure(dataset_dir)
        }
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest() == expected_checksum
    
    def _extract_archive(self, archive_path: Path, extract_dir: Path):
        """Extract archive file"""
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.suffix in ['.tar', '.gz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    def _get_directory_structure(self, path: Path, max_depth: int = 3) -> Dict:
        """Get directory structure for metadata"""
        structure = {}
        
        if max_depth <= 0:
            return structure
        
        try:
            for item in path.iterdir():
                if item.is_dir():
                    structure[item.name] = {
                        "type": "directory",
                        "contents": self._get_directory_structure(item, max_depth - 1)
                    }
                else:
                    structure[item.name] = {
                        "type": "file",
                        "size": item.stat().st_size
                    }
        except PermissionError:
            structure["<permission_denied>"] = {"type": "error"}
        
        return structure