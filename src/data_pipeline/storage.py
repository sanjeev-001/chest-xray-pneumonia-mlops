"""
Storage Backend for MinIO/S3 integration
Handles direct storage operations for data pipeline
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from minio import Minio
from minio.error import S3Error
import io

logger = logging.getLogger(__name__)


@dataclass
class StorageObject:
    """Information about a storage object"""
    key: str
    size: int
    last_modified: str
    etag: str


class StorageBackend:
    """Abstract base class for storage backends"""
    
    def upload_file(self, local_path: Union[str, Path], remote_key: str) -> bool:
        """Upload a file to storage"""
        raise NotImplementedError
    
    def download_file(self, remote_key: str, local_path: Union[str, Path]) -> bool:
        """Download a file from storage"""
        raise NotImplementedError
    
    def list_objects(self, prefix: str = "") -> List[StorageObject]:
        """List objects in storage"""
        raise NotImplementedError
    
    def delete_object(self, remote_key: str) -> bool:
        """Delete an object from storage"""
        raise NotImplementedError
    
    def object_exists(self, remote_key: str) -> bool:
        """Check if an object exists in storage"""
        raise NotImplementedError


class S3StorageBackend(StorageBackend):
    """AWS S3 storage backend"""
    
    def __init__(self, bucket_name: str, 
                 access_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 region: str = "us-east-1",
                 endpoint_url: Optional[str] = None):
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize S3 client
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        self.s3_client = session.client('s3', endpoint_url=endpoint_url)
        
        # Create bucket if it doesn't exist
        self._ensure_bucket_exists()
    
    def upload_file(self, local_path: Union[str, Path], remote_key: str) -> bool:
        """Upload a file to S3"""
        try:
            local_path = Path(local_path)
            if not local_path.exists():
                logger.error(f"Local file does not exist: {local_path}")
                return False
            
            self.s3_client.upload_file(str(local_path), self.bucket_name, remote_key)
            logger.info(f"Successfully uploaded {local_path} to s3://{self.bucket_name}/{remote_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to S3: {str(e)}")
            return False
    
    def download_file(self, remote_key: str, local_path: Union[str, Path]) -> bool:
        """Download a file from S3"""
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.s3_client.download_file(self.bucket_name, remote_key, str(local_path))
            logger.info(f"Successfully downloaded s3://{self.bucket_name}/{remote_key} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {remote_key} from S3: {str(e)}")
            return False
    
    def list_objects(self, prefix: str = "") -> List[StorageObject]:
        """List objects in S3 bucket"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            objects = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    objects.append(StorageObject(
                        key=obj['Key'],
                        size=obj['Size'],
                        last_modified=obj['LastModified'].isoformat(),
                        etag=obj['ETag'].strip('"')
                    ))
            
            return objects
            
        except Exception as e:
            logger.error(f"Failed to list objects in S3: {str(e)}")
            return []
    
    def delete_object(self, remote_key: str) -> bool:
        """Delete an object from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=remote_key)
            logger.info(f"Successfully deleted s3://{self.bucket_name}/{remote_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete {remote_key} from S3: {str(e)}")
            return False
    
    def object_exists(self, remote_key: str) -> bool:
        """Check if an object exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=remote_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logger.error(f"Error checking object existence: {str(e)}")
                return False
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                try:
                    if self.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    logger.info(f"Created S3 bucket: {self.bucket_name}")
                except Exception as create_error:
                    logger.warning(f"Failed to create bucket {self.bucket_name}: {create_error}")


class MinIOStorageBackend(StorageBackend):
    """MinIO storage backend"""
    
    def __init__(self, endpoint_url: str, bucket_name: str,
                 access_key: str, secret_key: str,
                 secure: bool = True):
        self.bucket_name = bucket_name
        
        # Parse endpoint URL
        if endpoint_url.startswith('http://'):
            endpoint = endpoint_url[7:]
            secure = False
        elif endpoint_url.startswith('https://'):
            endpoint = endpoint_url[8:]
            secure = True
        else:
            endpoint = endpoint_url
        
        # Initialize MinIO client
        self.minio_client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        
        # Create bucket if it doesn't exist
        self._ensure_bucket_exists()
    
    def upload_file(self, local_path: Union[str, Path], remote_key: str) -> bool:
        """Upload a file to MinIO"""
        try:
            local_path = Path(local_path)
            if not local_path.exists():
                logger.error(f"Local file does not exist: {local_path}")
                return False
            
            self.minio_client.fput_object(
                self.bucket_name,
                remote_key,
                str(local_path)
            )
            logger.info(f"Successfully uploaded {local_path} to MinIO bucket {self.bucket_name}/{remote_key}")
            return True
            
        except S3Error as e:
            logger.error(f"Failed to upload {local_path} to MinIO: {str(e)}")
            return False
    
    def download_file(self, remote_key: str, local_path: Union[str, Path]) -> bool:
        """Download a file from MinIO"""
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.minio_client.fget_object(
                self.bucket_name,
                remote_key,
                str(local_path)
            )
            logger.info(f"Successfully downloaded MinIO {self.bucket_name}/{remote_key} to {local_path}")
            return True
            
        except S3Error as e:
            logger.error(f"Failed to download {remote_key} from MinIO: {str(e)}")
            return False
    
    def list_objects(self, prefix: str = "") -> List[StorageObject]:
        """List objects in MinIO bucket"""
        try:
            objects = []
            for obj in self.minio_client.list_objects(self.bucket_name, prefix=prefix):
                objects.append(StorageObject(
                    key=obj.object_name,
                    size=obj.size,
                    last_modified=obj.last_modified.isoformat(),
                    etag=obj.etag
                ))
            
            return objects
            
        except S3Error as e:
            logger.error(f"Failed to list objects in MinIO: {str(e)}")
            return []
    
    def delete_object(self, remote_key: str) -> bool:
        """Delete an object from MinIO"""
        try:
            self.minio_client.remove_object(self.bucket_name, remote_key)
            logger.info(f"Successfully deleted MinIO {self.bucket_name}/{remote_key}")
            return True
            
        except S3Error as e:
            logger.error(f"Failed to delete {remote_key} from MinIO: {str(e)}")
            return False
    
    def object_exists(self, remote_key: str) -> bool:
        """Check if an object exists in MinIO"""
        try:
            self.minio_client.stat_object(self.bucket_name, remote_key)
            return True
        except S3Error as e:
            if e.code == 'NoSuchKey':
                return False
            else:
                logger.error(f"Error checking object existence: {str(e)}")
                return False
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist"""
        try:
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)
                logger.info(f"Created MinIO bucket: {self.bucket_name}")
        except S3Error as e:
            logger.warning(f"Failed to create bucket {self.bucket_name}: {str(e)}")


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend for development/testing"""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def upload_file(self, local_path: Union[str, Path], remote_key: str) -> bool:
        """Copy file to local storage"""
        try:
            local_path = Path(local_path)
            if not local_path.exists():
                logger.error(f"Local file does not exist: {local_path}")
                return False
            
            dest_path = self.base_path / remote_key
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(local_path, dest_path)
            logger.info(f"Successfully copied {local_path} to {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy {local_path} to local storage: {str(e)}")
            return False
    
    def download_file(self, remote_key: str, local_path: Union[str, Path]) -> bool:
        """Copy file from local storage"""
        try:
            source_path = self.base_path / remote_key
            if not source_path.exists():
                logger.error(f"Source file does not exist: {source_path}")
                return False
            
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(source_path, local_path)
            logger.info(f"Successfully copied {source_path} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy {remote_key} from local storage: {str(e)}")
            return False
    
    def list_objects(self, prefix: str = "") -> List[StorageObject]:
        """List objects in local storage"""
        try:
            objects = []
            search_path = self.base_path / prefix if prefix else self.base_path
            
            if search_path.is_file():
                # Single file
                stat = search_path.stat()
                objects.append(StorageObject(
                    key=str(search_path.relative_to(self.base_path)),
                    size=stat.st_size,
                    last_modified=str(stat.st_mtime),
                    etag=""
                ))
            elif search_path.is_dir():
                # Directory - recursively find files
                for file_path in search_path.rglob("*"):
                    if file_path.is_file():
                        stat = file_path.stat()
                        objects.append(StorageObject(
                            key=str(file_path.relative_to(self.base_path)),
                            size=stat.st_size,
                            last_modified=str(stat.st_mtime),
                            etag=""
                        ))
            
            return objects
            
        except Exception as e:
            logger.error(f"Failed to list objects in local storage: {str(e)}")
            return []
    
    def delete_object(self, remote_key: str) -> bool:
        """Delete an object from local storage"""
        try:
            file_path = self.base_path / remote_key
            if file_path.exists():
                if file_path.is_file():
                    file_path.unlink()
                else:
                    import shutil
                    shutil.rmtree(file_path)
                logger.info(f"Successfully deleted {file_path}")
                return True
            else:
                logger.warning(f"File does not exist: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete {remote_key} from local storage: {str(e)}")
            return False
    
    def object_exists(self, remote_key: str) -> bool:
        """Check if an object exists in local storage"""
        file_path = self.base_path / remote_key
        return file_path.exists()


def create_storage_backend(backend_type: str, **kwargs) -> StorageBackend:
    """
    Factory function to create storage backend
    
    Args:
        backend_type: Type of backend ('s3', 'minio', 'local')
        **kwargs: Backend-specific configuration
        
    Returns:
        StorageBackend instance
    """
    if backend_type.lower() == 's3':
        return S3StorageBackend(**kwargs)
    elif backend_type.lower() == 'minio':
        return MinIOStorageBackend(**kwargs)
    elif backend_type.lower() == 'local':
        return LocalStorageBackend(**kwargs)
    else:
        raise ValueError(f"Unsupported storage backend type: {backend_type}")


class DataStorageManager:
    """High-level manager for data storage operations"""
    
    def __init__(self, storage_backend: StorageBackend):
        self.storage_backend = storage_backend
    
    def upload_dataset_version(self, version_dir: Path, version_id: str) -> bool:
        """Upload an entire dataset version to storage"""
        try:
            success_count = 0
            total_files = 0
            
            for file_path in version_dir.rglob("*"):
                if file_path.is_file():
                    total_files += 1
                    rel_path = file_path.relative_to(version_dir)
                    remote_key = f"datasets/{version_id}/{rel_path}"
                    
                    if self.storage_backend.upload_file(file_path, remote_key):
                        success_count += 1
            
            logger.info(f"Uploaded {success_count}/{total_files} files for version {version_id}")
            return success_count == total_files
            
        except Exception as e:
            logger.error(f"Failed to upload dataset version {version_id}: {str(e)}")
            return False
    
    def download_dataset_version(self, version_id: str, local_dir: Path) -> bool:
        """Download an entire dataset version from storage"""
        try:
            # List all objects for this version
            objects = self.storage_backend.list_objects(f"datasets/{version_id}/")
            
            if not objects:
                logger.warning(f"No objects found for version {version_id}")
                return False
            
            success_count = 0
            for obj in objects:
                # Extract relative path
                rel_path = obj.key.replace(f"datasets/{version_id}/", "")
                local_path = local_dir / rel_path
                
                if self.storage_backend.download_file(obj.key, local_path):
                    success_count += 1
            
            logger.info(f"Downloaded {success_count}/{len(objects)} files for version {version_id}")
            return success_count == len(objects)
            
        except Exception as e:
            logger.error(f"Failed to download dataset version {version_id}: {str(e)}")
            return False
    
    def list_dataset_versions(self) -> List[str]:
        """List all dataset versions in storage"""
        try:
            objects = self.storage_backend.list_objects("datasets/")
            versions = set()
            
            for obj in objects:
                # Extract version ID from path: datasets/version_id/...
                parts = obj.key.split('/')
                if len(parts) >= 2:
                    versions.add(parts[1])
            
            return sorted(list(versions))
            
        except Exception as e:
            logger.error(f"Failed to list dataset versions: {str(e)}")
            return []
    
    def delete_dataset_version(self, version_id: str) -> bool:
        """Delete an entire dataset version from storage"""
        try:
            # List all objects for this version
            objects = self.storage_backend.list_objects(f"datasets/{version_id}/")
            
            success_count = 0
            for obj in objects:
                if self.storage_backend.delete_object(obj.key):
                    success_count += 1
            
            logger.info(f"Deleted {success_count}/{len(objects)} files for version {version_id}")
            return success_count == len(objects)
            
        except Exception as e:
            logger.error(f"Failed to delete dataset version {version_id}: {str(e)}")
            return False