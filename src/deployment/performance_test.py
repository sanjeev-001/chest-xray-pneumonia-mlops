#!/usr/bin/env python3
"""
Performance Testing Tool for Chest X-Ray API
Load testing and performance benchmarking
"""

import asyncio
import aiohttp
import time
import statistics
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

class PerformanceTester:
    """
    Performance testing and benchmarking tool
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.results = []
        self.lock = threading.Lock()
    
    async def single_request(self, session: aiohttp.ClientSession, image_path: str) -> Dict[str, Any]:
        """Make a single prediction request"""
        start_time = time.time()
        
        try:
            with open(image_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=Path(image_path).name, content_type='image/jpeg')
                
                async with session.post(f"{self.base_url}/predict", data=data) as response:
                    end_time = time.time()
                    response_time = (end_time - start_time) * 1000  # ms
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'success': True,
                            'response_time_ms': response_time,
                            'prediction': result.get('prediction'),
                            'confidence': result.get('confidence'),
                            'processing_time_ms': result.get('processing_time_ms'),
                            'cached': result.get('cached', False)
                        }
                    else:
                        return {
                            'success': False,
                            'response_time_ms': response_time,
                            'error': f"HTTP {response.status}"
                        }
        
        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            return {
                'success': False,
                'response_time_ms': response_time,
                'error': str(e)
            }
    
    async def load_test(self, image_path: str, num_requests: int, concurrency: int) -> Dict[str, Any]:
        """Run load test with specified concurrency"""
        print(f"🚀 Starting load test: {num_requests} requests, {concurrency} concurrent")
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(session):
            async with semaphore:
                return await self.single_request(session, image_path)
        
        # Run requests
        start_time = time.time()
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            tasks = [bounded_request(session) for _ in range(num_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Process results
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
        failed_results = [r for r in results if isinstance(r, dict) and not r.get('success')]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        # Calculate statistics
        if successful_results:
            response_times = [r['response_time_ms'] for r in successful_results]
            processing_times = [r.get('processing_time_ms', 0) for r in successful_results]
            cached_count = sum(1 for r in successful_results if r.get('cached'))
            
            stats = {
                'total_requests': num_requests,
                'successful_requests': len(successful_results),
                'failed_requests': len(failed_results) + len(exceptions),
                'success_rate': len(successful_results) / num_requests,
                'total_time_seconds': total_time,
                'requests_per_second': num_requests / total_time,
                'cached_responses': cached_count,
                'cache_hit_rate': cached_count / len(successful_results),
                'response_time_stats': {
                    'min_ms': min(response_times),
                    'max_ms': max(response_times),
                    'mean_ms': statistics.mean(response_times),
                    'median_ms': statistics.median(response_times),
                    'p95_ms': np.percentile(response_times, 95),
                    'p99_ms': np.percentile(response_times, 99)
                },
                'processing_time_stats': {
                    'min_ms': min(processing_times),
                    'max_ms': max(processing_times),
                    'mean_ms': statistics.mean(processing_times),
                    'median_ms': statistics.median(processing_times)
                } if processing_times else None
            }
        else:
            stats = {
                'total_requests': num_requests,
                'successful_requests': 0,
                'failed_requests': num_requests,
                'success_rate': 0,
                'total_time_seconds': total_time,
                'requests_per_second': 0,
                'error': 'All requests failed'
            }
        
        return stats
    
    async def throughput_test(self, image_path: str, duration_seconds: int, concurrency: int) -> Dict[str, Any]:
        """Run throughput test for specified duration"""
        print(f"📊 Starting throughput test: {duration_seconds}s duration, {concurrency} concurrent")
        
        results = []
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def continuous_requests(session):
            while time.time() < end_time:
                async with semaphore:
                    result = await self.single_request(session, image_path)
                    with self.lock:
                        results.append(result)
        
        # Run continuous requests
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            tasks = [continuous_requests(session) for _ in range(concurrency)]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        actual_duration = time.time() - start_time
        
        # Process results
        successful_results = [r for r in results if r.get('success')]
        failed_results = [r for r in results if not r.get('success')]
        
        if successful_results:
            response_times = [r['response_time_ms'] for r in successful_results]
            
            stats = {
                'duration_seconds': actual_duration,
                'total_requests': len(results),
                'successful_requests': len(successful_results),
                'failed_requests': len(failed_results),
                'success_rate': len(successful_results) / len(results),
                'throughput_rps': len(successful_results) / actual_duration,
                'response_time_stats': {
                    'min_ms': min(response_times),
                    'max_ms': max(response_times),
                    'mean_ms': statistics.mean(response_times),
                    'median_ms': statistics.median(response_times),
                    'p95_ms': np.percentile(response_times, 95)
                }
            }
        else:
            stats = {
                'duration_seconds': actual_duration,
                'total_requests': len(results),
                'successful_requests': 0,
                'failed_requests': len(results),
                'success_rate': 0,
                'throughput_rps': 0,
                'error': 'All requests failed'
            }
        
        return stats
    
    async def latency_test(self, image_path: str, num_requests: int = 100) -> Dict[str, Any]:
        """Test latency with sequential requests"""
        print(f"⏱️ Starting latency test: {num_requests} sequential requests")
        
        results = []
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for i in range(num_requests):
                result = await self.single_request(session, image_path)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"   Completed {i + 1}/{num_requests} requests")
        
        # Process results
        successful_results = [r for r in results if r.get('success')]
        
        if successful_results:
            response_times = [r['response_time_ms'] for r in successful_results]
            processing_times = [r.get('processing_time_ms', 0) for r in successful_results]
            
            stats = {
                'total_requests': num_requests,
                'successful_requests': len(successful_results),
                'response_time_stats': {
                    'min_ms': min(response_times),
                    'max_ms': max(response_times),
                    'mean_ms': statistics.mean(response_times),
                    'median_ms': statistics.median(response_times),
                    'std_ms': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                    'p50_ms': np.percentile(response_times, 50),
                    'p90_ms': np.percentile(response_times, 90),
                    'p95_ms': np.percentile(response_times, 95),
                    'p99_ms': np.percentile(response_times, 99)
                },
                'processing_time_stats': {
                    'min_ms': min(processing_times),
                    'max_ms': max(processing_times),
                    'mean_ms': statistics.mean(processing_times),
                    'median_ms': statistics.median(processing_times)
                } if processing_times else None,
                'response_times': response_times  # For plotting
            }
        else:
            stats = {
                'total_requests': num_requests,
                'successful_requests': 0,
                'error': 'All requests failed'
            }
        
        return stats
    
    def plot_results(self, latency_stats: Dict[str, Any], output_file: str = "performance_results.png"):
        """Plot performance test results"""
        if 'response_times' not in latency_stats:
            print("❌ No response time data to plot")
            return
        
        response_times = latency_stats['response_times']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Response time over time
        ax1.plot(response_times, 'b-', alpha=0.7)
        ax1.set_title('Response Time Over Time')
        ax1.set_xlabel('Request Number')
        ax1.set_ylabel('Response Time (ms)')
        ax1.grid(True, alpha=0.3)
        
        # Response time histogram
        ax2.hist(response_times, bins=30, alpha=0.7, color='green')
        ax2.set_title('Response Time Distribution')
        ax2.set_xlabel('Response Time (ms)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Percentile plot
        percentiles = [50, 75, 90, 95, 99]
        values = [np.percentile(response_times, p) for p in percentiles]
        ax3.bar(percentiles, values, alpha=0.7, color='orange')
        ax3.set_title('Response Time Percentiles')
        ax3.set_xlabel('Percentile')
        ax3.set_ylabel('Response Time (ms)')
        ax3.grid(True, alpha=0.3)
        
        # Summary statistics
        stats_text = f"""
        Mean: {statistics.mean(response_times):.1f}ms
        Median: {statistics.median(response_times):.1f}ms
        Min: {min(response_times):.1f}ms
        Max: {max(response_times):.1f}ms
        Std Dev: {statistics.stdev(response_times):.1f}ms
        P95: {np.percentile(response_times, 95):.1f}ms
        P99: {np.percentile(response_times, 99):.1f}ms
        """
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace')
        ax4.set_title('Summary Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance plot saved to {output_file}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"status": "unhealthy", "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

async def main():
    parser = argparse.ArgumentParser(description="Performance Testing Tool for Chest X-Ray API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--test-type", choices=["load", "throughput", "latency", "all"], 
                       default="all", help="Type of test to run")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests for load/latency test")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests for load test")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds for throughput test")
    parser.add_argument("--output", default="performance_results.png", help="Output file for plots")
    parser.add_argument("--save-json", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Validate image file
    if not Path(args.image).exists():
        print(f"❌ Image file not found: {args.image}")
        return 1
    
    tester = PerformanceTester(args.url)
    
    # Health check first
    print("🔍 Checking API health...")
    health = await tester.health_check()
    if health.get("status") != "healthy":
        print(f"❌ API health check failed: {health}")
        return 1
    
    print(f"✅ API is healthy")
    print(f"   Model loaded: {health.get('model_loaded')}")
    print(f"   Device: {health.get('device')}")
    print()
    
    results = {}
    
    # Run tests
    if args.test_type in ["load", "all"]:
        print("=" * 60)
        load_results = await tester.load_test(args.image, args.requests, args.concurrency)
        results["load_test"] = load_results
        
        print("📋 Load Test Results:")
        print(f"   Total requests: {load_results['total_requests']}")
        print(f"   Success rate: {load_results['success_rate']:.2%}")
        print(f"   Requests/second: {load_results['requests_per_second']:.1f}")
        print(f"   Cache hit rate: {load_results.get('cache_hit_rate', 0):.2%}")
        if 'response_time_stats' in load_results:
            rt = load_results['response_time_stats']
            print(f"   Response time - Mean: {rt['mean_ms']:.1f}ms, P95: {rt['p95_ms']:.1f}ms")
        print()
    
    if args.test_type in ["throughput", "all"]:
        print("=" * 60)
        throughput_results = await tester.throughput_test(args.image, args.duration, args.concurrency)
        results["throughput_test"] = throughput_results
        
        print("📊 Throughput Test Results:")
        print(f"   Duration: {throughput_results['duration_seconds']:.1f}s")
        print(f"   Total requests: {throughput_results['total_requests']}")
        print(f"   Success rate: {throughput_results['success_rate']:.2%}")
        print(f"   Throughput: {throughput_results['throughput_rps']:.1f} RPS")
        if 'response_time_stats' in throughput_results:
            rt = throughput_results['response_time_stats']
            print(f"   Response time - Mean: {rt['mean_ms']:.1f}ms, P95: {rt['p95_ms']:.1f}ms")
        print()
    
    if args.test_type in ["latency", "all"]:
        print("=" * 60)
        latency_results = await tester.latency_test(args.image, args.requests)
        results["latency_test"] = latency_results
        
        print("⏱️ Latency Test Results:")
        print(f"   Total requests: {latency_results['total_requests']}")
        print(f"   Successful requests: {latency_results['successful_requests']}")
        if 'response_time_stats' in latency_results:
            rt = latency_results['response_time_stats']
            print(f"   Response time statistics:")
            print(f"     Mean: {rt['mean_ms']:.1f}ms")
            print(f"     Median: {rt['median_ms']:.1f}ms")
            print(f"     P95: {rt['p95_ms']:.1f}ms")
            print(f"     P99: {rt['p99_ms']:.1f}ms")
            print(f"     Std Dev: {rt['std_ms']:.1f}ms")
        
        # Generate plots
        tester.plot_results(latency_results, args.output)
        print()
    
    # Save results to JSON if requested
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {args.save_json}")
    
    print("✅ Performance testing completed!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))