"""
Unit tests for data pipeline components
Tests image preprocessing, validation, and data versioning
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import json

from data_pipeline.ingestion import DataIngestionManager, DatasetInfo, IngestionResult
from data_pipeline.preprocessing import ImagePreprocessor, PreprocessingConfig, ProcessingResult
from data_pipeline.validation import DataValidator, ValidationConfig, ValidationResult
from data_pipeline.versioning import DataVersionController, DataVersion, StorageConfig
from data_pipeline.storage import LocalStorageBackend, create_storage_backend


class TestImagePreprocessor:
    """Test image preprocessing functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_image(self, temp_dir):
        """Create a sample chest X-ray-like image"""
        # Create a grayscale image that simulates a chest X-ray
        image = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        # Simulate lung areas (darker regions)
        image[100:400, 150:200] = np.random.randint(20, 80, (300, 50))
        image[100:400, 300:350] = np.random.randint(20, 80, (300, 50))
        
        image_path = temp_dir / "sample_xray.jpg"
        cv2.imwrite(str(image_path), image)
        return image_path
    
    @pytest.fixture
    def corrupted_image(self, temp_dir):
        """Create a corrupted image file"""
        corrupted_path = temp_dir / "corrupted.jpg"
        with open(corrupted_path, 'wb') as f:
            f.write(b"This is not an image file")
        return corrupted_path
    
    def test_preprocess_valid_image(self, sample_image):
        """Test preprocessing of a valid image"""
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess_image(sample_image)
        
        assert result.success is True
        assert result.processed_image is not None
        assert result.original_shape is not None
        assert result.final_shape == (224, 224, 3)  # RGB output
        assert "loaded_grayscale" in result.processing_steps
        assert "applied_clahe" in result.processing_steps
        assert "resized_with_aspect_ratio" in result.processing_steps
        assert "converted_to_rgb" in result.processing_steps
        assert "normalized_pixels" in result.processing_steps
    
    def test_preprocess_with_tensor_output(self, sample_image):
        """Test preprocessing with tensor output"""
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess_image(sample_image, return_tensor=True)
        
        assert result.success is True
        assert result.processed_image is not None
        assert "converted_to_tensor" in result.processing_steps
        # Tensor should have shape (C, H, W)
        assert len(result.final_shape) == 3
        assert result.final_shape[0] == 3  # RGB channels
    
    def test_preprocess_nonexistent_image(self):
        """Test preprocessing of non-existent image"""
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess_image("nonexistent.jpg")
        
        assert result.success is False
        assert "File not found" in result.error_message
    
    def test_preprocess_corrupted_image(self, corrupted_image):
        """Test preprocessing of corrupted image"""
        preprocessor = ImagePreprocessor()
        result = preprocessor.preprocess_image(corrupted_image)
        
        assert result.success is False
        assert result.error_message is not None
    
    def test_preprocess_batch(self, temp_dir):
        """Test batch preprocessing"""
        preprocessor = ImagePreprocessor()
        
        # Create multiple sample images
        image_paths = []
        for i in range(3):
            image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            image_path = temp_dir / f"image_{i}.jpg"
            cv2.imwrite(str(image_path), image)
            image_paths.append(image_path)
        
        results = preprocessor.preprocess_batch(image_paths)
        
        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(r.final_shape == (224, 224, 3) for r in results)
    
    def test_custom_preprocessing_config(self, sample_image):
        """Test preprocessing with custom configuration"""
        config = PreprocessingConfig(
            target_size=(128, 128),
            apply_clahe=False,
            maintain_aspect_ratio=False
        )
        preprocessor = ImagePreprocessor(config)
        result = preprocessor.preprocess_image(sample_image)
        
        assert result.success is True
        assert result.final_shape == (128, 128, 3)
        assert "applied_clahe" not in result.processing_steps
        assert "resized_direct" in result.processing_steps
    
    def test_calculate_statistics(self, temp_dir):
        """Test dataset statistics calculation"""
        preprocessor = ImagePreprocessor()
        
        # Create sample images with known properties
        image_paths = []
        for i in range(5):
            # Create images with different brightness levels
            brightness = 50 + i * 40  # 50, 90, 130, 170, 210
            image = np.full((100, 100), brightness, dtype=np.uint8)
            image_path = temp_dir / f"image_{i}.jpg"
            cv2.imwrite(str(image_path), image)
            image_paths.append(image_path)
        
        stats = preprocessor.calculate_statistics(image_paths)
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["valid_images"] == 5
        assert 0 <= stats["mean"] <= 1  # Normalized values
    
    def test_augmentation_pipeline(self):
        """Test augmentation pipeline creation"""
        preprocessor = ImagePreprocessor()
        
        # Training pipeline should have augmentations
        train_pipeline = preprocessor.get_augmentation_pipeline(training=True)
        assert train_pipeline is not None
        
        # Validation pipeline should be simpler
        val_pipeline = preprocessor.get_augmentation_pipeline(training=False)
        assert val_pipeline is not None


class TestDataValidator:
    """Test data validation functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def valid_image(self, temp_dir):
        """Create a valid test image"""
        image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        image_path = temp_dir / "valid_image.jpg"
        cv2.imwrite(str(image_path), image)
        return image_path
    
    @pytest.fixture
    def small_image(self, temp_dir):
        """Create an image that's too small"""
        image = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        image_path = temp_dir / "small_image.jpg"
        cv2.imwrite(str(image_path), image)
        return image_path
    
    @pytest.fixture
    def dark_image(self, temp_dir):
        """Create a very dark image"""
        image = np.full((256, 256), 5, dtype=np.uint8)  # Very dark
        image_path = temp_dir / "dark_image.jpg"
        cv2.imwrite(str(image_path), image)
        return image_path
    
    @pytest.fixture
    def bright_image(self, temp_dir):
        """Create a very bright image"""
        image = np.full((256, 256), 250, dtype=np.uint8)  # Very bright
        image_path = temp_dir / "bright_image.jpg"
        cv2.imwrite(str(image_path), image)
        return image_path
    
    def test_validate_valid_image(self, valid_image):
        """Test validation of a valid image"""
        validator = DataValidator()
        result = validator.validate_image(valid_image)
        
        assert result.is_valid is True
        assert result.file_path == valid_image
        assert result.file_size_bytes > 0
        assert result.image_dimensions is not None
        assert result.format == ".jpg"
        assert result.brightness_score is not None
        assert result.contrast_score is not None
        assert result.blur_score is not None
        assert len(result.issues) == 0
    
    def test_validate_nonexistent_image(self):
        """Test validation of non-existent image"""
        validator = DataValidator()
        result = validator.validate_image("nonexistent.jpg")
        
        assert result.is_valid is False
        assert "File does not exist" in result.issues
    
    def test_validate_small_image(self, small_image):
        """Test validation of image that's too small"""
        validator = DataValidator()
        result = validator.validate_image(small_image)
        
        assert result.is_valid is False
        assert any("too small" in issue for issue in result.issues)
    
    def test_validate_dark_image(self, dark_image):
        """Test validation of very dark image"""
        validator = DataValidator()
        result = validator.validate_image(dark_image)
        
        assert result.is_valid is False
        assert any("too dark" in issue for issue in result.issues)
    
    def test_validate_bright_image(self, bright_image):
        """Test validation of very bright image"""
        validator = DataValidator()
        result = validator.validate_image(bright_image)
        
        assert result.is_valid is False
        assert any("too bright" in issue for issue in result.issues)
    
    def test_validate_dataset(self, temp_dir):
        """Test validation of entire dataset"""
        validator = DataValidator()
        
        # Create a mix of valid and invalid images
        # Valid images
        for i in range(3):
            image = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
            image_path = temp_dir / f"valid_{i}.jpg"
            cv2.imwrite(str(image_path), image)
        
        # Invalid images (too small)
        for i in range(2):
            image = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
            image_path = temp_dir / f"invalid_{i}.jpg"
            cv2.imwrite(str(image_path), image)
        
        summary = validator.validate_dataset(temp_dir)
        
        assert summary.total_files == 5
        assert summary.valid_files == 3
        assert summary.invalid_files == 2
        assert len(summary.validation_results) == 5
        assert len(summary.common_issues) > 0
        assert "dataset_statistics" in summary.__dict__
        assert len(summary.recommendations) >= 0
    
    def test_custom_validation_config(self, valid_image):
        """Test validation with custom configuration"""
        config = ValidationConfig(
            min_image_size=(128, 128),
            max_image_size=(1024, 1024),
            min_brightness=0.1,
            max_brightness=0.9
        )
        validator = DataValidator(config)
        result = validator.validate_image(valid_image)
        
        # Should pass with relaxed constraints
        assert result.is_valid is True
    
    def test_validate_directory_structure(self, temp_dir):
        """Test directory structure validation"""
        validator = DataValidator()
        
        # Create class-based structure
        normal_dir = temp_dir / "NORMAL"
        pneumonia_dir = temp_dir / "PNEUMONIA"
        normal_dir.mkdir()
        pneumonia_dir.mkdir()
        
        # Add some images
        for i in range(2):
            image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            cv2.imwrite(str(normal_dir / f"normal_{i}.jpg"), image)
            cv2.imwrite(str(pneumonia_dir / f"pneumonia_{i}.jpg"), image)
        
        structure_info = validator.validate_directory_structure(temp_dir)
        
        assert structure_info["structure_type"] == "class_based"
        assert "NORMAL" in structure_info["class_directories"]
        assert "PNEUMONIA" in structure_info["class_directories"]


class TestDataIngestionManager:
    """Test data ingestion functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_dataset(self, temp_dir):
        """Create a sample dataset structure"""
        dataset_dir = temp_dir / "sample_dataset"
        normal_dir = dataset_dir / "NORMAL"
        pneumonia_dir = dataset_dir / "PNEUMONIA"
        
        normal_dir.mkdir(parents=True)
        pneumonia_dir.mkdir(parents=True)
        
        # Create sample images
        for i in range(3):
            image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            cv2.imwrite(str(normal_dir / f"normal_{i}.jpg"), image)
            cv2.imwrite(str(pneumonia_dir / f"pneumonia_{i}.jpg"), image)
        
        return dataset_dir
    
    def test_ingest_local_dataset(self, sample_dataset, temp_dir):
        """Test ingestion of local dataset"""
        ingestion_manager = DataIngestionManager(str(temp_dir / "data"))
        
        result = ingestion_manager.ingest_local_dataset(
            str(sample_dataset), 
            "test_dataset"
        )
        
        assert result.success is True
        assert result.total_files == 6  # 3 normal + 3 pneumonia
        assert result.valid_files == 6
        assert result.invalid_files == 0
        assert len(result.errors) == 0
    
    def test_ingest_nonexistent_dataset(self, temp_dir):
        """Test ingestion of non-existent dataset"""
        ingestion_manager = DataIngestionManager(str(temp_dir / "data"))
        
        with pytest.raises(FileNotFoundError):
            ingestion_manager.ingest_local_dataset(
                "nonexistent_path", 
                "test_dataset"
            )
    
    def test_list_available_datasets(self, sample_dataset, temp_dir):
        """Test listing available datasets"""
        ingestion_manager = DataIngestionManager(str(temp_dir / "data"))
        
        # Initially no datasets
        datasets = ingestion_manager.list_available_datasets()
        assert len(datasets) == 0
        
        # Ingest a dataset
        ingestion_manager.ingest_local_dataset(str(sample_dataset), "test_dataset")
        
        # Should now have one dataset
        datasets = ingestion_manager.list_available_datasets()
        assert len(datasets) == 1
        assert "test_dataset" in datasets
    
    def test_get_dataset_info(self, sample_dataset, temp_dir):
        """Test getting dataset information"""
        ingestion_manager = DataIngestionManager(str(temp_dir / "data"))
        
        # Ingest dataset first
        ingestion_manager.ingest_local_dataset(str(sample_dataset), "test_dataset")
        
        # Get dataset info
        info = ingestion_manager.get_dataset_info("test_dataset")
        
        assert info["name"] == "test_dataset"
        assert info["total_images"] == 6
        assert "directory_structure" in info
        assert "supported_formats" in info


class TestDataVersionController:
    """Test data versioning functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_dataset(self, temp_dir):
        """Create a sample dataset for versioning"""
        dataset_dir = temp_dir / "dataset"
        normal_dir = dataset_dir / "NORMAL"
        pneumonia_dir = dataset_dir / "PNEUMONIA"
        
        normal_dir.mkdir(parents=True)
        pneumonia_dir.mkdir(parents=True)
        
        # Create sample images
        for i in range(10):
            image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
            cv2.imwrite(str(normal_dir / f"normal_{i}.jpg"), image)
        
        for i in range(8):
            image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
            cv2.imwrite(str(pneumonia_dir / f"pneumonia_{i}.jpg"), image)
        
        return dataset_dir
    
    def test_create_data_version(self, sample_dataset, temp_dir):
        """Test creating a data version"""
        version_controller = DataVersionController(str(temp_dir))
        
        data_version = version_controller.create_data_version(
            dataset_path=sample_dataset,
            dataset_name="chest_xray_test",
            split_ratios=(0.7, 0.2, 0.1),
            metadata={"test": True}
        )
        
        assert data_version.dataset_name == "chest_xray_test"
        assert data_version.total_samples == 18  # 10 normal + 8 pneumonia
        assert len(data_version.splits) == 3
        assert "train" in data_version.splits
        assert "val" in data_version.splits
        assert "test" in data_version.splits
        
        # Check split sizes are approximately correct
        train_size = data_version.splits["train"].total_samples
        val_size = data_version.splits["val"].total_samples
        test_size = data_version.splits["test"].total_samples
        
        assert train_size + val_size + test_size == 18
        assert train_size >= 10  # Should be around 70% of 18 = 12-13
        assert val_size >= 2   # Should be around 20% of 18 = 3-4
        assert test_size >= 1  # Should be around 10% of 18 = 1-2
    
    def test_list_versions(self, sample_dataset, temp_dir):
        """Test listing data versions"""
        version_controller = DataVersionController(str(temp_dir))
        
        # Initially no versions
        versions = version_controller.list_versions()
        assert len(versions) == 0
        
        # Create a version
        version_controller.create_data_version(
            dataset_path=sample_dataset,
            dataset_name="test_dataset"
        )
        
        # Should now have one version
        versions = version_controller.list_versions()
        assert len(versions) == 1
        assert versions[0].dataset_name == "test_dataset"
    
    def test_get_version(self, sample_dataset, temp_dir):
        """Test getting a specific version"""
        version_controller = DataVersionController(str(temp_dir))
        
        # Create a version
        created_version = version_controller.create_data_version(
            dataset_path=sample_dataset,
            dataset_name="test_dataset"
        )
        
        # Retrieve the version
        retrieved_version = version_controller.get_version(created_version.version_id)
        
        assert retrieved_version is not None
        assert retrieved_version.version_id == created_version.version_id
        assert retrieved_version.dataset_name == created_version.dataset_name
        assert retrieved_version.total_samples == created_version.total_samples
    
    def test_get_split_paths(self, sample_dataset, temp_dir):
        """Test getting file paths for a specific split"""
        version_controller = DataVersionController(str(temp_dir))
        
        # Create a version
        data_version = version_controller.create_data_version(
            dataset_path=sample_dataset,
            dataset_name="test_dataset"
        )
        
        # Get train split paths
        train_paths = version_controller.get_split_paths(data_version.version_id, "train")
        
        assert len(train_paths) > 0
        assert all(path.exists() for path in train_paths)
        assert all(path.suffix.lower() in ['.jpg', '.jpeg', '.png'] for path in train_paths)
    
    def test_delete_version(self, sample_dataset, temp_dir):
        """Test deleting a data version"""
        version_controller = DataVersionController(str(temp_dir))
        
        # Create a version
        data_version = version_controller.create_data_version(
            dataset_path=sample_dataset,
            dataset_name="test_dataset"
        )
        
        version_id = data_version.version_id
        
        # Verify version exists
        assert version_controller.get_version(version_id) is not None
        
        # Delete version
        version_controller.delete_version(version_id)
        
        # Verify version is deleted
        assert version_controller.get_version(version_id) is None


class TestStorageBackend:
    """Test storage backend functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_file(self, temp_dir):
        """Create a sample file for storage tests"""
        file_path = temp_dir / "sample.txt"
        with open(file_path, 'w') as f:
            f.write("This is a test file for storage backend testing.")
        return file_path
    
    def test_local_storage_backend(self, temp_dir, sample_file):
        """Test local storage backend operations"""
        storage_dir = temp_dir / "storage"
        backend = LocalStorageBackend(storage_dir)
        
        # Test upload
        success = backend.upload_file(sample_file, "test/sample.txt")
        assert success is True
        
        # Test object exists
        assert backend.object_exists("test/sample.txt") is True
        assert backend.object_exists("nonexistent.txt") is False
        
        # Test list objects
        objects = backend.list_objects("test/")
        assert len(objects) == 1
        assert objects[0].key == "test/sample.txt"
        
        # Test download
        download_path = temp_dir / "downloaded.txt"
        success = backend.download_file("test/sample.txt", download_path)
        assert success is True
        assert download_path.exists()
        
        # Verify content
        with open(download_path, 'r') as f:
            content = f.read()
        assert "test file for storage" in content
        
        # Test delete
        success = backend.delete_object("test/sample.txt")
        assert success is True
        assert backend.object_exists("test/sample.txt") is False
    
    def test_create_storage_backend_factory(self, temp_dir):
        """Test storage backend factory function"""
        # Test local backend creation
        backend = create_storage_backend("local", base_path=str(temp_dir))
        assert isinstance(backend, LocalStorageBackend)
        
        # Test invalid backend type
        with pytest.raises(ValueError):
            create_storage_backend("invalid_type")


# Integration test for the complete data pipeline
class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_dataset(self, temp_dir):
        """Create a realistic sample dataset"""
        dataset_dir = temp_dir / "chest_xray_dataset"
        normal_dir = dataset_dir / "NORMAL"
        pneumonia_dir = dataset_dir / "PNEUMONIA"
        
        normal_dir.mkdir(parents=True)
        pneumonia_dir.mkdir(parents=True)
        
        # Create more realistic chest X-ray-like images
        for i in range(20):
            # Normal images - more uniform
            image = np.random.randint(80, 180, (512, 512), dtype=np.uint8)
            # Add some lung-like structures
            image[100:400, 150:200] = np.random.randint(60, 120, (300, 50))
            image[100:400, 300:350] = np.random.randint(60, 120, (300, 50))
            cv2.imwrite(str(normal_dir / f"normal_{i:03d}.jpg"), image)
        
        for i in range(15):
            # Pneumonia images - more varied intensity
            image = np.random.randint(40, 220, (512, 512), dtype=np.uint8)
            # Add some opacity patterns
            image[150:350, 200:300] = np.random.randint(120, 200, (200, 100))
            cv2.imwrite(str(pneumonia_dir / f"pneumonia_{i:03d}.jpg"), image)
        
        return dataset_dir
    
    def test_complete_pipeline_workflow(self, sample_dataset, temp_dir):
        """Test complete data pipeline workflow"""
        # 1. Data Ingestion
        ingestion_manager = DataIngestionManager(str(temp_dir / "raw_data"))
        ingestion_result = ingestion_manager.ingest_local_dataset(
            str(sample_dataset), 
            "chest_xray_v1"
        )
        
        assert ingestion_result.success is True
        assert ingestion_result.total_files == 35
        assert ingestion_result.valid_files == 35
        
        # 2. Data Validation
        validator = DataValidator()
        validation_summary = validator.validate_dataset(ingestion_result.dataset_path)
        
        assert validation_summary.total_files == 35
        # Most images should be valid (allowing for some quality issues)
        assert validation_summary.valid_files >= 30
        
        # 3. Data Versioning
        version_controller = DataVersionController(str(temp_dir))
        data_version = version_controller.create_data_version(
            dataset_path=ingestion_result.dataset_path,
            dataset_name="chest_xray_v1",
            split_ratios=(0.7, 0.2, 0.1),
            metadata={
                "source": "test_dataset",
                "validation_summary": {
                    "total_files": validation_summary.total_files,
                    "valid_files": validation_summary.valid_files
                }
            }
        )
        
        assert data_version.total_samples == validation_summary.valid_files
        assert len(data_version.splits) == 3
        
        # 4. Image Preprocessing
        preprocessor = ImagePreprocessor()
        
        # Test preprocessing on train split
        train_paths = version_controller.get_split_paths(data_version.version_id, "train")
        assert len(train_paths) > 0
        
        # Preprocess a few images from train split
        sample_paths = train_paths[:3]
        preprocessing_results = preprocessor.preprocess_batch(sample_paths)
        
        assert len(preprocessing_results) == 3
        assert all(r.success for r in preprocessing_results)
        assert all(r.final_shape == (224, 224, 3) for r in preprocessing_results)
        
        # 5. Calculate dataset statistics
        stats = preprocessor.calculate_statistics(train_paths[:10])  # Use subset for speed
        
        assert "mean" in stats
        assert "std" in stats
        assert stats["valid_images"] == 10
        assert 0 <= stats["mean"] <= 1
        
        print(f"Pipeline test completed successfully:")
        print(f"- Ingested {ingestion_result.valid_files} files")
        print(f"- Validated {validation_summary.valid_files} valid files")
        print(f"- Created version {data_version.version_id}")
        print(f"- Train split: {data_version.splits['train'].total_samples} samples")
        print(f"- Val split: {data_version.splits['val'].total_samples} samples")
        print(f"- Test split: {data_version.splits['test'].total_samples} samples")
        print(f"- Dataset mean brightness: {stats['mean']:.3f}")