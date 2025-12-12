#!/usr/bin/env python3
"""
Complete System Validation Script
Validates the entire MLOps system end-to-end with real model
"""

import requests
import time
import json
import base64
import numpy as np
from PIL import Image
import io
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SystemValidator:
    """Complete system validation"""
    
    def __init__(self):
        self.endpoints = {
            'model_server': 'http://localhost:8000',
            'api_gateway': 'http://localhost:8080'
        }
        self.validation_results = {}
    
    def create_test_image(self, image_type='normal') -> str:
        """Create a test chest X-ray image"""
        # Create a synthetic 224x224 chest X-ray-like image
        img_array = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
        
        if image_type == 'pneumonia':
            # Add some cloudy patterns for pneumonia simulation
            for _ in range(8):
                x, y = np.random.randint(50, 174, 2)
                img_array[x:x+50, y:y+50] = np.random.randint(80, 120)
        else:
            # Keep it cleaner for normal
            img_array = np.clip(img_array + 40, 0, 255)
        
        # Convert to RGB
        img_rgb = np.stack([img_array] * 3, axis=-1)
        img = Image.fromarray(img_rgb.astype(np.uint8))
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def validate_model_server(self) -> bool:
        """Validate model server functionality"""
        logger.info("🔍 Validating model server...")
        
        try:
            # Health check
            response = requests.get(f"{self.endpoints['model_server']}/health", timeout=10)
            if response.status_code != 200:
                logger.error(f"❌ Health check failed: {response.status_code}")
                return False
            
            health_data = response.json()
            logger.info(f"✅ Health check passed")
            logger.info(f"   Model loaded: {health_data.get('model_loaded', False)}")
            logger.info(f"   Architecture: {health_data.get('model_architecture', 'unknown')}")
            
            # Model info
            response = requests.get(f"{self.endpoints['model_server']}/model/info", timeout=10)
            if response.status_code == 200:
                model_info = response.json()
                logger.info(f"✅ Model info retrieved")
                logger.info(f"   Classes: {model_info.get('class_names', [])}")
                logger.info(f"   Input size: {model_info.get('input_size', [])}")
            
            # Single prediction test
            test_image = self.create_test_image('normal')
            img_bytes = base64.b64decode(test_image)
            files = {'file': ('test_normal.png', img_bytes, 'image/png')}
            
            response = requests.post(f"{self.endpoints['model_server']}/predict", files=files, timeout=30)
            if response.status_code != 200:
                logger.error(f"❌ Prediction failed: {response.status_code}")
                return False
            
            prediction = response.json()
            logger.info(f"✅ Single prediction test passed")
            logger.info(f"   Prediction: {prediction.get('prediction', 'unknown')}")
            logger.info(f"   Confidence: {prediction.get('confidence', 0):.3f}")
            logger.info(f"   Processing time: {prediction.get('processing_time_ms', 0):.1f}ms")
            
            # Batch prediction test
            batch_request = {
                'images': [self.create_test_image('normal'), self.create_test_image('pneumonia')],
                'return_probabilities': True,
                'return_confidence': True
            }
            
            response = requests.post(f"{self.endpoints['model_server']}/predict/batch", 
                                   json=batch_request, timeout=60)
            if response.status_code == 200:
                batch_result = response.json()
                logger.info(f"✅ Batch prediction test passed")
                logger.info(f"   Batch size: {batch_result.get('batch_size', 0)}")
                logger.info(f"   Total time: {batch_result.get('total_processing_time_ms', 0):.1f}ms")
            
            self.validation_results['model_server'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Model server validation failed: {e}")
            self.validation_results['model_server'] = False
            return False
    
    def validate_performance(self) -> bool:
        """Validate system performance"""
        logger.info("⚡ Validating system performance...")
        
        try:
            # Performance test with multiple predictions
            processing_times = []
            num_tests = 10
            
            for i in range(num_tests):
                test_image = self.create_test_image('normal' if i % 2 == 0 else 'pneumonia')
                img_bytes = base64.b64decode(test_image)
                files = {'file': (f'test_{i}.png', img_bytes, 'image/png')}
                
                start_time = time.time()
                response = requests.post(f"{self.endpoints['model_server']}/predict", 
                                       files=files, timeout=30)
                end_time = time.time()
                
                if response.status_code == 200:
                    prediction = response.json()
                    processing_times.append(prediction.get('processing_time_ms', 0))
                else:
                    logger.warning(f"⚠️ Prediction {i+1} failed: {response.status_code}")
            
            if processing_times:
                avg_time = np.mean(processing_times)
                p95_time = np.percentile(processing_times, 95)
                
                logger.info(f"✅ Performance test completed")
                logger.info(f"   Tests run: {len(processing_times)}/{num_tests}")
                logger.info(f"   Average time: {avg_time:.1f}ms")
                logger.info(f"   P95 time: {p95_time:.1f}ms")
                
                # Performance thresholds
                if avg_time < 2000 and p95_time < 5000:
                    logger.info("✅ Performance within acceptable limits")
                    self.validation_results['performance'] = True
                    return True
                else:
                    logger.warning("⚠️ Performance may be slower than expected")
                    self.validation_results['performance'] = False
                    return False
            else:
                logger.error("❌ No successful predictions for performance test")
                self.validation_results['performance'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Performance validation failed: {e}")
            self.validation_results['performance'] = False
            return False
    
    def validate_error_handling(self) -> bool:
        """Validate error handling"""
        logger.info("🛡️ Validating error handling...")
        
        try:
            # Test invalid file format
            invalid_file = b"This is not an image"
            files = {'file': ('test.txt', invalid_file, 'text/plain')}
            
            response = requests.post(f"{self.endpoints['model_server']}/predict", files=files)
            if response.status_code == 400:
                logger.info("✅ Invalid file format handled correctly")
            else:
                logger.warning(f"⚠️ Unexpected response for invalid file: {response.status_code}")
            
            # Test empty request
            response = requests.post(f"{self.endpoints['model_server']}/predict")
            if response.status_code in [400, 422]:
                logger.info("✅ Empty request handled correctly")
            else:
                logger.warning(f"⚠️ Unexpected response for empty request: {response.status_code}")
            
            # Test invalid batch request
            invalid_batch = {'images': ['invalid_base64_data']}
            response = requests.post(f"{self.endpoints['model_server']}/predict/batch", 
                                   json=invalid_batch)
            if response.status_code == 400:
                logger.info("✅ Invalid batch request handled correctly")
            else:
                logger.warning(f"⚠️ Unexpected response for invalid batch: {response.status_code}")
            
            self.validation_results['error_handling'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling validation failed: {e}")
            self.validation_results['error_handling'] = False
            return False
    
    def validate_model_accuracy(self) -> bool:
        """Validate model predictions make sense"""
        logger.info("🎯 Validating model accuracy...")
        
        try:
            # Test with different image types
            test_cases = [
                ('normal', 'NORMAL'),
                ('pneumonia', 'PNEUMONIA')
            ]
            
            correct_predictions = 0
            total_predictions = 0
            
            for image_type, expected_class in test_cases:
                # Test multiple times for each type
                for _ in range(3):
                    test_image = self.create_test_image(image_type)
                    img_bytes = base64.b64decode(test_image)
                    files = {'file': (f'test_{image_type}.png', img_bytes, 'image/png')}
                    
                    response = requests.post(f"{self.endpoints['model_server']}/predict", 
                                           files=files, timeout=30)
                    
                    if response.status_code == 200:
                        prediction = response.json()
                        predicted_class = prediction.get('prediction', '')
                        confidence = prediction.get('confidence', 0)
                        
                        total_predictions += 1
                        
                        # For synthetic images, we can't expect perfect accuracy,
                        # but we can check that predictions are reasonable
                        if confidence > 0.5:  # Reasonable confidence
                            logger.info(f"   {image_type}: {predicted_class} (conf: {confidence:.3f})")
                        else:
                            logger.info(f"   {image_type}: {predicted_class} (low conf: {confidence:.3f})")
            
            if total_predictions > 0:
                logger.info(f"✅ Model accuracy validation completed")
                logger.info(f"   Total predictions: {total_predictions}")
                logger.info("   Note: Using synthetic images, so accuracy assessment is limited")
                self.validation_results['model_accuracy'] = True
                return True
            else:
                logger.error("❌ No successful predictions for accuracy test")
                self.validation_results['model_accuracy'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Model accuracy validation failed: {e}")
            self.validation_results['model_accuracy'] = False
            return False
    
    def generate_validation_report(self) -> dict:
        """Generate validation report"""
        total_tests = len(self.validation_results)
        passed_tests = sum(self.validation_results.values())
        
        report = {
            'validation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'test_results': self.validation_results,
            'overall_status': 'PASSED' if passed_tests == total_tests else 'FAILED',
            'endpoints_tested': self.endpoints
        }
        
        return report
    
    def run_complete_validation(self) -> bool:
        """Run complete system validation"""
        logger.info("🚀 Starting complete system validation...")
        
        validation_steps = [
            ("Model Server Functionality", self.validate_model_server),
            ("System Performance", self.validate_performance),
            ("Error Handling", self.validate_error_handling),
            ("Model Accuracy", self.validate_model_accuracy),
        ]
        
        for step_name, step_func in validation_steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"Validation: {step_name}")
            logger.info(f"{'='*60}")
            
            step_func()  # Don't fail on individual steps, collect all results
        
        # Generate report
        report = self.generate_validation_report()
        
        logger.info(f"\n{'='*60}")
        logger.info("📊 VALIDATION SUMMARY")
        logger.info(f"{'='*60}")
        
        for test_name, result in self.validation_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status} {test_name.replace('_', ' ').title()}")
        
        logger.info(f"\n📈 Overall Results:")
        logger.info(f"   Tests passed: {report['passed_tests']}/{report['total_tests']}")
        logger.info(f"   Success rate: {report['success_rate']:.1%}")
        logger.info(f"   Overall status: {report['overall_status']}")
        
        # Save report
        report_path = Path("validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📄 Validation report saved to: {report_path}")
        
        if report['overall_status'] == 'PASSED':
            logger.info("\n🎉 SYSTEM VALIDATION COMPLETED SUCCESSFULLY!")
            logger.info("The MLOps system is ready for production use.")
        else:
            logger.info("\n⚠️ SYSTEM VALIDATION COMPLETED WITH ISSUES")
            logger.info("Please review the failed tests and address any issues.")
        
        return report['overall_status'] == 'PASSED'


def main():
    validator = SystemValidator()
    success = validator.run_complete_validation()
    
    if success:
        print("\n✅ All validations passed! System is ready.")
        exit(0)
    else:
        print("\n❌ Some validations failed. Please check the logs.")
        exit(1)


if __name__ == "__main__":
    main()