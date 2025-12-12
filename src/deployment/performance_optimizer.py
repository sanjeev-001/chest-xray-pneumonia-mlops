"""
Performance Optimizer for Chest X-Ray Model Server
Implements caching, model optimization, and performance monitoring
"""

import time
import logging
import hashlib
import pickle
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import defaultdict, deque
import psutil
import gc

import torch
import torch.nn as nn
from torch.jit import script, trace
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class ModelCache:
    """
    Intelligent caching system for model predictions
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
        logger.info(f"ModelCache initialized: max_size={max_size}, ttl={ttl_seconds}s")
    
    def _generate_key(self, image_data: bytes) -> str:
        """Generate cache key from image data"""
        return hashlib.md5(image_data).hexdigest()
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds)
    
    def _evict_expired(self):
        """Remove expired entries"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp > timedelta(seconds=self.ttl_seconds)
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]
    
    def _evict_lru(self):
        """Evict least recently used entries"""
        if len(self.cache) <= self.max_size:
            return
        
        # Sort by access time and remove oldest
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        keys_to_remove = [key for key, _ in sorted_keys[:len(self.cache) - self.max_size]]
        
        for key in keys_to_remove:
            del self.cache[key]
            del self.access_times[key]
    
    def get(self, image_data: bytes) -> Optional[Dict[str, Any]]:
        """Get cached prediction"""
        with self.lock:
            key = self._generate_key(image_data)
            
            if key in self.cache:
                result, timestamp = self.cache[key]
                
                if not self._is_expired(timestamp):
                    self.access_times[key] = datetime.now()
                    self.hit_count += 1
                    logger.debug(f"Cache hit for key {key[:8]}...")
                    return result
                else:
                    # Remove expired entry
                    del self.cache[key]
                    del self.access_times[key]
            
            self.miss_count += 1
            logger.debug(f"Cache miss for key {key[:8]}...")
            return None
    
    def put(self, image_data: bytes, result: Dict[str, Any]):
        """Store prediction in cache"""
        with self.lock:
            key = self._generate_key(image_data)
            timestamp = datetime.now()
            
            self.cache[key] = (result, timestamp)
            self.access_times[key] = timestamp
            
            # Cleanup
            self._evict_expired()
            self._evict_lru()
            
            logger.debug(f"Cached result for key {key[:8]}...")
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            logger.info("Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / max(total_requests, 1)
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "ttl_seconds": self.ttl_seconds
            }

class ModelOptimizer:
    """
    Model optimization for faster inference
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.original_model = model
        self.device = device
        self.optimized_model = None
        self.optimization_applied = False
        
        logger.info("ModelOptimizer initialized")
    
    def optimize_for_inference(self) -> nn.Module:
        """Optimize model for faster inference"""
        logger.info("Optimizing model for inference...")
        
        try:
            # Set to evaluation mode
            self.original_model.eval()
            
            # Apply optimizations
            optimized_model = self.original_model
            
            # 1. Disable gradient computation
            for param in optimized_model.parameters():
                param.requires_grad = False
            
            # 2. Use torch.jit.script for optimization (if supported)
            try:
                # Create dummy input for tracing
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                
                # Trace the model
                traced_model = torch.jit.trace(optimized_model, dummy_input)
                traced_model.eval()
                
                # Optimize traced model
                traced_model = torch.jit.optimize_for_inference(traced_model)
                
                logger.info("Applied TorchScript optimization")
                optimized_model = traced_model
                
            except Exception as e:
                logger.warning(f"TorchScript optimization failed: {e}")
            
            # 3. Apply fusion optimizations (if available)
            try:
                if hasattr(torch.jit, 'fuse'):
                    optimized_model = torch.jit.fuse(optimized_model)
                    logger.info("Applied fusion optimization")
            except Exception as e:
                logger.warning(f"Fusion optimization failed: {e}")
            
            self.optimized_model = optimized_model
            self.optimization_applied = True
            
            logger.info("Model optimization completed")
            return self.optimized_model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return self.original_model
    
    def benchmark_model(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance"""
        logger.info(f"Benchmarking model performance ({num_iterations} iterations)...")
        
        # Prepare test input
        test_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = self.original_model(test_input)
        
        # Benchmark original model
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.original_model(test_input)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        original_time = time.time() - start_time
        
        # Benchmark optimized model if available
        optimized_time = original_time
        if self.optimized_model is not None:
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = self.optimized_model(test_input)
            
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            optimized_time = time.time() - start_time
        
        results = {
            "original_time_ms": (original_time / num_iterations) * 1000,
            "optimized_time_ms": (optimized_time / num_iterations) * 1000,
            "speedup": original_time / optimized_time if optimized_time > 0 else 1.0,
            "iterations": num_iterations
        }
        
        logger.info(f"Benchmark results: {results['speedup']:.2f}x speedup")
        return results

class PerformanceMonitor:
    """
    Real-time performance monitoring
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.response_times = deque(maxlen=window_size)
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Resource monitoring
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.gpu_usage = deque(maxlen=100) if torch.cuda.is_available() else None
        
        # Prediction statistics
        self.prediction_stats = defaultdict(int)
        
        logger.info("PerformanceMonitor initialized")
    
    def record_request(self, response_time_ms: float, prediction: str, error: bool = False):
        """Record request metrics"""
        self.response_times.append(response_time_ms)
        self.request_count += 1
        
        if error:
            self.error_count += 1
        else:
            self.prediction_stats[prediction] += 1
        
        # Update resource usage
        self._update_resource_usage()
    
    def _update_resource_usage(self):
        """Update system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.cpu_usage.append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.append(memory.percent)
            
            # GPU usage (if available)
            if torch.cuda.is_available() and self.gpu_usage is not None:
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                self.gpu_usage.append(gpu_memory)
                
        except Exception as e:
            logger.warning(f"Failed to update resource usage: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        uptime = time.time() - self.start_time
        
        # Response time statistics
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            min_response_time = min(self.response_times)
            max_response_time = max(self.response_times)
            p95_response_time = np.percentile(list(self.response_times), 95)
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = 0
        
        # Request rate
        requests_per_second = self.request_count / max(uptime, 1)
        error_rate = self.error_count / max(self.request_count, 1)
        
        # Resource usage
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        avg_gpu = sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else 0
        
        return {
            "uptime_seconds": uptime,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": error_rate,
            "requests_per_second": requests_per_second,
            "response_time": {
                "avg_ms": avg_response_time,
                "min_ms": min_response_time,
                "max_ms": max_response_time,
                "p95_ms": p95_response_time
            },
            "resource_usage": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "gpu_percent": avg_gpu
            },
            "predictions": dict(self.prediction_stats)
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status based on performance metrics"""
        stats = self.get_stats()
        
        # Health thresholds
        cpu_threshold = 80
        memory_threshold = 85
        error_rate_threshold = 0.05
        response_time_threshold = 2000  # 2 seconds
        
        issues = []
        
        if stats["resource_usage"]["cpu_percent"] > cpu_threshold:
            issues.append(f"High CPU usage: {stats['resource_usage']['cpu_percent']:.1f}%")
        
        if stats["resource_usage"]["memory_percent"] > memory_threshold:
            issues.append(f"High memory usage: {stats['resource_usage']['memory_percent']:.1f}%")
        
        if stats["error_rate"] > error_rate_threshold:
            issues.append(f"High error rate: {stats['error_rate']:.3f}")
        
        if stats["response_time"]["avg_ms"] > response_time_threshold:
            issues.append(f"Slow response time: {stats['response_time']['avg_ms']:.1f}ms")
        
        status = "healthy" if not issues else "degraded"
        
        return {
            "status": status,
            "issues": issues,
            "metrics": stats
        }

class BatchProcessor:
    """
    Batch processing for improved throughput
    """
    
    def __init__(self, max_batch_size: int = 8, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.lock = threading.Lock()
        
        logger.info(f"BatchProcessor initialized: batch_size={max_batch_size}, wait_time={max_wait_time}s")
    
    def add_request(self, image_tensor: torch.Tensor, request_id: str) -> bool:
        """Add request to batch queue"""
        with self.lock:
            self.pending_requests.append((image_tensor, request_id, time.time()))
            return len(self.pending_requests) >= self.max_batch_size
    
    def get_batch(self) -> Tuple[torch.Tensor, List[str], List[float]]:
        """Get batch of requests for processing"""
        with self.lock:
            if not self.pending_requests:
                return None, [], []
            
            current_time = time.time()
            
            # Check if we should process based on size or time
            should_process = (
                len(self.pending_requests) >= self.max_batch_size or
                (self.pending_requests and 
                 current_time - self.pending_requests[0][2] >= self.max_wait_time)
            )
            
            if not should_process:
                return None, [], []
            
            # Extract batch
            batch_size = min(len(self.pending_requests), self.max_batch_size)
            batch_requests = self.pending_requests[:batch_size]
            self.pending_requests = self.pending_requests[batch_size:]
            
            # Prepare batch tensor
            images = [req[0] for req in batch_requests]
            request_ids = [req[1] for req in batch_requests]
            timestamps = [req[2] for req in batch_requests]
            
            batch_tensor = torch.cat(images, dim=0)
            
            return batch_tensor, request_ids, timestamps

class MemoryManager:
    """
    Memory management and optimization
    """
    
    def __init__(self):
        self.gc_threshold = 100  # Number of requests before GC
        self.request_count = 0
        
        logger.info("MemoryManager initialized")
    
    def cleanup_request(self):
        """Cleanup after request processing"""
        self.request_count += 1
        
        # Periodic garbage collection
        if self.request_count % self.gc_threshold == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("Performed memory cleanup")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        stats = {
            "system_memory": dict(psutil.virtual_memory()._asdict()),
            "process_memory": psutil.Process().memory_info()._asdict()
        }
        
        if torch.cuda.is_available():
            stats["gpu_memory"] = {
                "allocated": torch.cuda.memory_allocated(),
                "cached": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
                "max_cached": torch.cuda.max_memory_reserved()
            }
        
        return stats
    
    def optimize_memory(self):
        """Optimize memory usage"""
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Memory optimization completed")

class PerformanceOptimizer:
    """
    Main performance optimization coordinator
    """
    
    def __init__(self, model: nn.Module, device: torch.device, config: Dict[str, Any] = None):
        self.model = model
        self.device = device
        self.config = config or {}
        
        # Initialize components
        self.cache = ModelCache(
            max_size=self.config.get('cache_size', 1000),
            ttl_seconds=self.config.get('cache_ttl', 3600)
        )
        
        self.model_optimizer = ModelOptimizer(model, device)
        self.performance_monitor = PerformanceMonitor()
        self.memory_manager = MemoryManager()
        
        # Batch processing (optional)
        if self.config.get('enable_batching', False):
            self.batch_processor = BatchProcessor(
                max_batch_size=self.config.get('batch_size', 8),
                max_wait_time=self.config.get('batch_wait_time', 0.1)
            )
        else:
            self.batch_processor = None
        
        # Optimize model on initialization
        if self.config.get('auto_optimize', True):
            self.optimized_model = self.model_optimizer.optimize_for_inference()
        else:
            self.optimized_model = self.model
        
        logger.info("PerformanceOptimizer initialized")
    
    def predict_with_optimization(self, image_data: bytes, use_cache: bool = True) -> Dict[str, Any]:
        """Optimized prediction with caching and monitoring"""
        start_time = time.time()
        
        try:
            # Check cache first
            if use_cache:
                cached_result = self.cache.get(image_data)
                if cached_result is not None:
                    # Add cache hit info
                    cached_result['cached'] = True
                    cached_result['processing_time_ms'] = (time.time() - start_time) * 1000
                    
                    self.performance_monitor.record_request(
                        cached_result['processing_time_ms'],
                        cached_result['prediction']
                    )
                    
                    return cached_result
            
            # Process image and make prediction
            # (This would integrate with your existing model server logic)
            # For now, return a placeholder structure
            
            processing_time = (time.time() - start_time) * 1000
            
            result = {
                'prediction': 'NORMAL',  # Placeholder
                'confidence': 0.95,      # Placeholder
                'processing_time_ms': processing_time,
                'cached': False,
                'optimized': self.model_optimizer.optimization_applied
            }
            
            # Cache result
            if use_cache:
                self.cache.put(image_data, result)
            
            # Record metrics
            self.performance_monitor.record_request(processing_time, result['prediction'])
            
            # Memory cleanup
            self.memory_manager.cleanup_request()
            
            return result
            
        except Exception as e:
            error_time = (time.time() - start_time) * 1000
            self.performance_monitor.record_request(error_time, 'ERROR', error=True)
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'cache': self.cache.stats(),
            'performance': self.performance_monitor.get_stats(),
            'memory': self.memory_manager.get_memory_stats(),
            'health': self.performance_monitor.get_health_status(),
            'optimization': {
                'model_optimized': self.model_optimizer.optimization_applied,
                'batching_enabled': self.batch_processor is not None
            }
        }
    
    def benchmark(self) -> Dict[str, Any]:
        """Run performance benchmark"""
        return self.model_optimizer.benchmark_model()
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.cache.clear()
    
    def optimize_memory(self):
        """Optimize memory usage"""
        self.memory_manager.optimize_memory()