"""
Data Version Controller for managing dataset versions with DVC
Handles data versioning, storage backends, and data splitting
"""

import os
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml
import hashlib
import random

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Information about a data split"""
    name: str  # train, val, test
    file_paths: List[str]
    class_distribution: Dict[str, int]
    total_samples: int


@dataclass
class DataVersion:
    """Information about a data version"""
    version_id: str
    timestamp: str
    dataset_name: str
    total_samples: int
    splits: Dict[str, DataSplit]
    metadata: Dict
    dvc_file_path: Optional[str] = None
    storage_path: Optional[str] = None


@dataclass
class StorageConfig:
    """Configuration for storage backend"""
    backend_type: str  # "local", "s3", "minio"
    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    bucket_name: Optional[str] = None
    region: Optional[str] = None


class DataVersionController:
    """Manages data versioning with DVC and storage backends"""
    
    def __init__(self, project_root: str = ".", storage_config: StorageConfig = None):
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.versions_dir = self.data_dir / "versions"
        self.dvc_dir = self.project_root / ".dvc"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.versions_dir.mkdir(exist_ok=True)
        
        self.storage_config = storage_config
        
        # Initialize DVC if not already initialized
        self._initialize_dvc()
        
        # Configure storage backend if provided
        if storage_config:
            self._configure_storage_backend(storage_config)
    
    def create_data_version(self, dataset_path: Union[str, Path], 
                          dataset_name: str,
                          split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                          stratify_by_class: bool = True,
                          metadata: Dict = None) -> DataVersion:
        """
        Create a new data version with train/val/test splits
        
        Args:
            dataset_path: Path to the dataset directory
            dataset_name: Name of the dataset
            split_ratios: Ratios for train/val/test splits
            stratify_by_class: Whether to stratify splits by class
            metadata: Additional metadata
            
        Returns:
            DataVersion object with version information
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        # Generate version ID
        timestamp = datetime.now().isoformat()
        version_id = self._generate_version_id(dataset_name, timestamp)
        
        logger.info(f"Creating data version {version_id} for dataset {dataset_name}")
        
        # Create version directory
        version_dir = self.versions_dir / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Discover and organize files by class
        file_class_mapping = self._discover_files_and_classes(dataset_path)
        
        # Create data splits
        splits = self._create_data_splits(
            file_class_mapping, 
            split_ratios, 
            stratify_by_class
        )
        
        # Copy files to version directory with split structure
        self._organize_files_by_splits(file_class_mapping, splits, version_dir)
        
        # Create DVC tracking
        dvc_file_path = self._create_dvc_tracking(version_dir, version_id)
        
        # Create data version object
        data_version = DataVersion(
            version_id=version_id,
            timestamp=timestamp,
            dataset_name=dataset_name,
            total_samples=sum(len(files) for files in file_class_mapping.values()),
            splits=splits,
            metadata=metadata or {},
            dvc_file_path=str(dvc_file_path),
            storage_path=str(version_dir)
        )
        
        # Save version metadata
        self._save_version_metadata(data_version)
        
        logger.info(f"Successfully created data version {version_id}")
        return data_version
    
    def list_versions(self, dataset_name: Optional[str] = None) -> List[DataVersion]:
        """
        List all data versions, optionally filtered by dataset name
        
        Args:
            dataset_name: Optional dataset name filter
            
        Returns:
            List of DataVersion objects
        """
        versions = []
        
        for version_dir in self.versions_dir.iterdir():
            if version_dir.is_dir():
                metadata_file = version_dir / "version_metadata.json"
                if metadata_file.exists():
                    try:
                        version = self._load_version_metadata(metadata_file)
                        if dataset_name is None or version.dataset_name == dataset_name:
                            versions.append(version)
                    except Exception as e:
                        logger.warning(f"Failed to load version metadata from {metadata_file}: {e}")
        
        # Sort by timestamp (newest first)
        versions.sort(key=lambda v: v.timestamp, reverse=True)
        return versions
    
    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """
        Get a specific data version by ID
        
        Args:
            version_id: Version identifier
            
        Returns:
            DataVersion object or None if not found
        """
        version_dir = self.versions_dir / version_id
        metadata_file = version_dir / "version_metadata.json"
        
        if metadata_file.exists():
            return self._load_version_metadata(metadata_file)
        
        return None
    
    def delete_version(self, version_id: str, remove_from_storage: bool = False):
        """
        Delete a data version
        
        Args:
            version_id: Version identifier
            remove_from_storage: Whether to remove from remote storage
        """
        version_dir = self.versions_dir / version_id
        
        if not version_dir.exists():
            raise FileNotFoundError(f"Version {version_id} not found")
        
        # Remove from DVC tracking if configured
        if remove_from_storage and self.storage_config:
            try:
                self._remove_from_dvc_storage(version_id)
            except Exception as e:
                logger.warning(f"Failed to remove from DVC storage: {e}")
        
        # Remove local version directory
        shutil.rmtree(version_dir)
        logger.info(f"Deleted data version {version_id}")
    
    def push_to_storage(self, version_id: str):
        """
        Push a data version to remote storage using DVC
        
        Args:
            version_id: Version identifier
        """
        if not self.storage_config:
            raise ValueError("Storage backend not configured")
        
        version_dir = self.versions_dir / version_id
        dvc_file = version_dir.with_suffix('.dvc')
        
        if not dvc_file.exists():
            raise FileNotFoundError(f"DVC file not found for version {version_id}")
        
        try:
            # Push to remote storage
            result = subprocess.run(
                ["dvc", "push", str(dvc_file)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Successfully pushed version {version_id} to storage")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push to storage: {e.stderr}")
            raise
    
    def pull_from_storage(self, version_id: str):
        """
        Pull a data version from remote storage using DVC
        
        Args:
            version_id: Version identifier
        """
        if not self.storage_config:
            raise ValueError("Storage backend not configured")
        
        version_dir = self.versions_dir / version_id
        dvc_file = version_dir.with_suffix('.dvc')
        
        if not dvc_file.exists():
            raise FileNotFoundError(f"DVC file not found for version {version_id}")
        
        try:
            # Pull from remote storage
            result = subprocess.run(
                ["dvc", "pull", str(dvc_file)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Successfully pulled version {version_id} from storage")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull from storage: {e.stderr}")
            raise
    
    def get_split_paths(self, version_id: str, split_name: str) -> List[Path]:
        """
        Get file paths for a specific split of a data version
        
        Args:
            version_id: Version identifier
            split_name: Split name (train, val, test)
            
        Returns:
            List of file paths
        """
        version = self.get_version(version_id)
        if not version:
            raise FileNotFoundError(f"Version {version_id} not found")
        
        if split_name not in version.splits:
            raise ValueError(f"Split {split_name} not found in version {version_id}")
        
        version_dir = self.versions_dir / version_id
        split_paths = []
        
        for rel_path in version.splits[split_name].file_paths:
            full_path = version_dir / rel_path
            if full_path.exists():
                split_paths.append(full_path)
        
        return split_paths
    
    def _initialize_dvc(self):
        """Initialize DVC repository if not already initialized"""
        if not self.dvc_dir.exists():
            try:
                subprocess.run(
                    ["dvc", "init"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info("Initialized DVC repository")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to initialize DVC: {e.stderr}")
            except FileNotFoundError:
                logger.warning("DVC not found. Please install DVC for data versioning.")
    
    def _configure_storage_backend(self, config: StorageConfig):
        """Configure DVC storage backend"""
        try:
            if config.backend_type == "s3":
                # Configure S3 remote
                subprocess.run([
                    "dvc", "remote", "add", "-d", "storage", 
                    f"s3://{config.bucket_name}"
                ], cwd=self.project_root, check=True)
                
                if config.access_key and config.secret_key:
                    subprocess.run([
                        "dvc", "remote", "modify", "storage", 
                        "access_key_id", config.access_key
                    ], cwd=self.project_root, check=True)
                    
                    subprocess.run([
                        "dvc", "remote", "modify", "storage", 
                        "secret_access_key", config.secret_key
                    ], cwd=self.project_root, check=True)
            
            elif config.backend_type == "minio":
                # Configure MinIO remote
                subprocess.run([
                    "dvc", "remote", "add", "-d", "storage", 
                    f"s3://{config.bucket_name}"
                ], cwd=self.project_root, check=True)
                
                subprocess.run([
                    "dvc", "remote", "modify", "storage", 
                    "endpointurl", config.endpoint_url
                ], cwd=self.project_root, check=True)
                
                if config.access_key and config.secret_key:
                    subprocess.run([
                        "dvc", "remote", "modify", "storage", 
                        "access_key_id", config.access_key
                    ], cwd=self.project_root, check=True)
                    
                    subprocess.run([
                        "dvc", "remote", "modify", "storage", 
                        "secret_access_key", config.secret_key
                    ], cwd=self.project_root, check=True)
            
            logger.info(f"Configured {config.backend_type} storage backend")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure storage backend: {e}")
        except FileNotFoundError:
            logger.warning("DVC not found. Storage backend configuration skipped.")
    
    def _discover_files_and_classes(self, dataset_path: Path) -> Dict[str, List[Path]]:
        """Discover files and organize by class"""
        file_class_mapping = {}
        
        # Look for class-based directory structure
        class_dirs = []
        for item in dataset_path.iterdir():
            if item.is_dir() and item.name.upper() in ['NORMAL', 'PNEUMONIA']:
                class_dirs.append(item)
        
        if class_dirs:
            # Class-based structure
            for class_dir in class_dirs:
                class_name = class_dir.name.upper()
                file_class_mapping[class_name] = []
                
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                    files = list(class_dir.rglob(f"*{ext}"))
                    files.extend(class_dir.rglob(f"*{ext.upper()}"))
                    file_class_mapping[class_name].extend(files)
        else:
            # Flat structure - try to infer class from filename
            all_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                files = list(dataset_path.rglob(f"*{ext}"))
                files.extend(dataset_path.rglob(f"*{ext.upper()}"))
                all_files.extend(files)
            
            # Group by inferred class
            file_class_mapping = {"NORMAL": [], "PNEUMONIA": []}
            for file_path in all_files:
                filename_lower = file_path.name.lower()
                if "normal" in filename_lower:
                    file_class_mapping["NORMAL"].append(file_path)
                elif "pneumonia" in filename_lower:
                    file_class_mapping["PNEUMONIA"].append(file_path)
                else:
                    # Default to NORMAL if can't determine
                    file_class_mapping["NORMAL"].append(file_path)
        
        return file_class_mapping
    
    def _create_data_splits(self, file_class_mapping: Dict[str, List[Path]], 
                          split_ratios: Tuple[float, float, float],
                          stratify: bool) -> Dict[str, DataSplit]:
        """Create train/val/test splits"""
        train_ratio, val_ratio, test_ratio = split_ratios
        
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        splits = {
            "train": DataSplit("train", [], {}, 0),
            "val": DataSplit("val", [], {}, 0),
            "test": DataSplit("test", [], {}, 0)
        }
        
        for class_name, files in file_class_mapping.items():
            if not files:
                continue
            
            # Shuffle files
            files_copy = files.copy()
            random.shuffle(files_copy)
            
            n_files = len(files_copy)
            n_train = int(n_files * train_ratio)
            n_val = int(n_files * val_ratio)
            n_test = n_files - n_train - n_val
            
            # Split files
            train_files = files_copy[:n_train]
            val_files = files_copy[n_train:n_train + n_val]
            test_files = files_copy[n_train + n_val:]
            
            # Add to splits
            splits["train"].file_paths.extend([str(f) for f in train_files])
            splits["val"].file_paths.extend([str(f) for f in val_files])
            splits["test"].file_paths.extend([str(f) for f in test_files])
            
            # Update class distributions
            for split_name, file_list in [("train", train_files), ("val", val_files), ("test", test_files)]:
                splits[split_name].class_distribution[class_name] = len(file_list)
                splits[split_name].total_samples += len(file_list)
        
        return splits
    
    def _organize_files_by_splits(self, file_class_mapping: Dict[str, List[Path]], 
                                splits: Dict[str, DataSplit], 
                                version_dir: Path):
        """Organize files in version directory by splits"""
        for split_name, split_data in splits.items():
            split_dir = version_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Create class directories within split
            for class_name in file_class_mapping.keys():
                class_dir = split_dir / class_name
                class_dir.mkdir(exist_ok=True)
        
        # Copy files to appropriate split/class directories
        for split_name, split_data in splits.items():
            for file_path_str in split_data.file_paths:
                source_path = Path(file_path_str)
                
                # Determine class from file path or name
                class_name = self._determine_class_from_path(source_path, file_class_mapping)
                
                # Create destination path
                dest_path = version_dir / split_name / class_name / source_path.name
                
                # Copy file
                shutil.copy2(source_path, dest_path)
                
                # Update file path in split data to be relative to version dir
                rel_path = dest_path.relative_to(version_dir)
                split_data.file_paths[split_data.file_paths.index(file_path_str)] = str(rel_path)
    
    def _determine_class_from_path(self, file_path: Path, 
                                 file_class_mapping: Dict[str, List[Path]]) -> str:
        """Determine class name from file path"""
        for class_name, files in file_class_mapping.items():
            if file_path in files:
                return class_name
        return "NORMAL"  # Default fallback
    
    def _create_dvc_tracking(self, version_dir: Path, version_id: str) -> Path:
        """Create DVC tracking for version directory"""
        try:
            # Add directory to DVC tracking
            subprocess.run(
                ["dvc", "add", str(version_dir)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            dvc_file = version_dir.with_suffix('.dvc')
            logger.info(f"Created DVC tracking file: {dvc_file}")
            return dvc_file
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to create DVC tracking: {e.stderr}")
            return version_dir / "dummy.dvc"
        except FileNotFoundError:
            logger.warning("DVC not found. Skipping DVC tracking.")
            return version_dir / "dummy.dvc"
    
    def _generate_version_id(self, dataset_name: str, timestamp: str) -> str:
        """Generate unique version ID"""
        # Create hash from dataset name and timestamp
        hash_input = f"{dataset_name}_{timestamp}".encode()
        hash_digest = hashlib.md5(hash_input).hexdigest()[:8]
        
        # Format: dataset_name_YYYYMMDD_HHMMSS_hash
        clean_timestamp = timestamp.replace(':', '').replace('-', '').replace('T', '_')[:15]
        return f"{dataset_name}_{clean_timestamp}_{hash_digest}"
    
    def _save_version_metadata(self, data_version: DataVersion):
        """Save version metadata to JSON file"""
        version_dir = self.versions_dir / data_version.version_id
        metadata_file = version_dir / "version_metadata.json"
        
        # Convert DataVersion to dict for JSON serialization
        metadata = asdict(data_version)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_version_metadata(self, metadata_file: Path) -> DataVersion:
        """Load version metadata from JSON file"""
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        # Convert splits back to DataSplit objects
        splits = {}
        for split_name, split_data in data['splits'].items():
            splits[split_name] = DataSplit(
                name=split_data['name'],
                file_paths=split_data['file_paths'],
                class_distribution=split_data['class_distribution'],
                total_samples=split_data['total_samples']
            )
        
        data['splits'] = splits
        return DataVersion(**data)
    
    def _remove_from_dvc_storage(self, version_id: str):
        """Remove version from DVC storage"""
        dvc_file = self.versions_dir / f"{version_id}.dvc"
        if dvc_file.exists():
            subprocess.run(
                ["dvc", "remove", str(dvc_file)],
                cwd=self.project_root,
                check=True
            )