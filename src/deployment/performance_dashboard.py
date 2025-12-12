#!/usr/bin/env python3
"""
Performance Monitoring Dashboard
Real-time performance monitoring for Chest X-Ray API
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import argparse

class PerformanceDashboard:
    """
    Real-time performance monitoring dashboard
    """
    
    def __init__(self, api_url: str = "http://localhost:8000", update_interval: int = 5):
        self.api_url = api_url.rstrip('/')
        self.update_interval = update_interval
        
        # Data storage (keep last 100 data points)
        self.timestamps = deque(maxlen=100)
        self.response_times = deque(maxlen=100)
        self.request_rates = deque(maxlen=100)
        self.error_rates = deque(maxlen=100)
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.cache_hit_rates = deque(maxlen=100)
        
        # Setup matplotlib
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Chest X-Ray API Performance Dashboard', fontsize=16, color='white')
        
        # Configure subplots
        self.setup_plots()
        
        # Animation
        self.animation = None
        
        print("📊 Performance Dashboard initialized")
    
    def setup_plots(self):
        """Setup subplot configurations"""
        # Response Time
        self.axes[0, 0].set_title('Response Time', color='white')
        self.axes[0, 0].set_ylabel('Time (ms)', color='white')
        self.axes[0, 0].grid(True, alpha=0.3)
        self.axes[0, 0].tick_params(colors='white')
        
        # Request Rate
        self.axes[0, 1].set_title('Request Rate', color='white')
        self.axes[0, 1].set_ylabel('Requests/sec', color='white')
        self.axes[0, 1].grid(True, alpha=0.3)
        self.axes[0, 1].tick_params(colors='white')
        
        # Error Rate
        self.axes[0, 2].set_title('Error Rate', color='white')
        self.axes[0, 2].set_ylabel('Error %', color='white')
        self.axes[0, 2].grid(True, alpha=0.3)
        self.axes[0, 2].tick_params(colors='white')
        
        # CPU Usage
        self.axes[1, 0].set_title('CPU Usage', color='white')
        self.axes[1, 0].set_ylabel('CPU %', color='white')
        self.axes[1, 0].grid(True, alpha=0.3)
        self.axes[1, 0].tick_params(colors='white')
        
        # Memory Usage
        self.axes[1, 1].set_title('Memory Usage', color='white')
        self.axes[1, 1].set_ylabel('Memory %', color='white')
        self.axes[1, 1].grid(True, alpha=0.3)
        self.axes[1, 1].tick_params(colors='white')
        
        # Cache Hit Rate
        self.axes[1, 2].set_title('Cache Hit Rate', color='white')
        self.axes[1, 2].set_ylabel('Hit Rate %', color='white')
        self.axes[1, 2].grid(True, alpha=0.3)
        self.axes[1, 2].tick_params(colors='white')
        
        plt.tight_layout()
    
    async def fetch_metrics(self) -> Dict[str, Any]:
        """Fetch metrics from API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/performance/stats") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"❌ Failed to fetch metrics: HTTP {response.status}")
                        return {}
        except Exception as e:
            print(f"❌ Error fetching metrics: {e}")
            return {}
    
    def update_data(self, metrics: Dict[str, Any]):
        """Update data collections with new metrics"""
        current_time = datetime.now()
        self.timestamps.append(current_time)
        
        # Extract metrics with defaults
        perf_metrics = metrics.get('performance', {})
        cache_metrics = metrics.get('cache', {})
        resource_metrics = perf_metrics.get('resource_usage', {})
        response_time_metrics = perf_metrics.get('response_time', {})
        
        # Response time (average)
        avg_response_time = response_time_metrics.get('avg_ms', 0)
        self.response_times.append(avg_response_time)
        
        # Request rate
        request_rate = perf_metrics.get('requests_per_second', 0)
        self.request_rates.append(request_rate)
        
        # Error rate (as percentage)
        error_rate = perf_metrics.get('error_rate', 0) * 100
        self.error_rates.append(error_rate)
        
        # CPU usage
        cpu_usage = resource_metrics.get('cpu_percent', 0)
        self.cpu_usage.append(cpu_usage)
        
        # Memory usage
        memory_usage = resource_metrics.get('memory_percent', 0)
        self.memory_usage.append(memory_usage)
        
        # Cache hit rate (as percentage)
        cache_hit_rate = cache_metrics.get('hit_rate', 0) * 100
        self.cache_hit_rates.append(cache_hit_rate)
    
    def update_plots(self, frame):
        """Update plots with latest data"""
        # Fetch new metrics
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            metrics = loop.run_until_complete(self.fetch_metrics())
            loop.close()
            
            if metrics:
                self.update_data(metrics)
        except Exception as e:
            print(f"❌ Error updating plots: {e}")
            return
        
        if not self.timestamps:
            return
        
        # Convert timestamps to relative seconds for x-axis
        base_time = self.timestamps[0]
        x_data = [(t - base_time).total_seconds() for t in self.timestamps]
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Reconfigure plots
        self.setup_plots()
        
        # Plot data
        if len(x_data) > 1:
            # Response Time
            self.axes[0, 0].plot(x_data, list(self.response_times), 'cyan', linewidth=2)
            self.axes[0, 0].fill_between(x_data, list(self.response_times), alpha=0.3, color='cyan')
            
            # Request Rate
            self.axes[0, 1].plot(x_data, list(self.request_rates), 'lime', linewidth=2)
            self.axes[0, 1].fill_between(x_data, list(self.request_rates), alpha=0.3, color='lime')
            
            # Error Rate
            self.axes[0, 2].plot(x_data, list(self.error_rates), 'red', linewidth=2)
            self.axes[0, 2].fill_between(x_data, list(self.error_rates), alpha=0.3, color='red')
            
            # CPU Usage
            self.axes[1, 0].plot(x_data, list(self.cpu_usage), 'orange', linewidth=2)
            self.axes[1, 0].fill_between(x_data, list(self.cpu_usage), alpha=0.3, color='orange')
            
            # Memory Usage
            self.axes[1, 1].plot(x_data, list(self.memory_usage), 'magenta', linewidth=2)
            self.axes[1, 1].fill_between(x_data, list(self.memory_usage), alpha=0.3, color='magenta')
            
            # Cache Hit Rate
            self.axes[1, 2].plot(x_data, list(self.cache_hit_rates), 'yellow', linewidth=2)
            self.axes[1, 2].fill_between(x_data, list(self.cache_hit_rates), alpha=0.3, color='yellow')
        
        # Add current values as text
        if self.response_times:
            current_values = [
                f"Current: {self.response_times[-1]:.1f}ms",
                f"Current: {self.request_rates[-1]:.1f} req/s",
                f"Current: {self.error_rates[-1]:.2f}%",
                f"Current: {self.cpu_usage[-1]:.1f}%",
                f"Current: {self.memory_usage[-1]:.1f}%",
                f"Current: {self.cache_hit_rates[-1]:.1f}%"
            ]
            
            for i, (ax, value_text) in enumerate(zip(self.axes.flat, current_values)):
                ax.text(0.02, 0.95, value_text, transform=ax.transAxes, 
                       fontsize=10, color='white', weight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Update title with timestamp
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.fig.suptitle(f'Chest X-Ray API Performance Dashboard - {current_time}', 
                         fontsize=16, color='white')
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        print(f"🚀 Starting performance monitoring...")
        print(f"   API URL: {self.api_url}")
        print(f"   Update interval: {self.update_interval}s")
        print("   Press Ctrl+C to stop")
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self.update_plots, interval=self.update_interval * 1000, blit=False
        )
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
    
    def save_snapshot(self, filename: str = None):
        """Save current dashboard as image"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_snapshot_{timestamp}.png"
        
        self.fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"📸 Dashboard snapshot saved to {filename}")

class PerformanceReporter:
    """
    Generate performance reports
    """
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
    
    async def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        print("📋 Generating performance report...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch current metrics
                async with session.get(f"{self.api_url}/performance/stats") as response:
                    if response.status == 200:
                        metrics = await response.json()
                    else:
                        return {"error": f"Failed to fetch metrics: HTTP {response.status}"}
                
                # Run benchmark
                async with session.get(f"{self.api_url}/performance/benchmark") as response:
                    if response.status == 200:
                        benchmark = await response.json()
                    else:
                        benchmark = {"error": "Benchmark failed"}
                
                # Get health status
                async with session.get(f"{self.api_url}/health") as response:
                    if response.status == 200:
                        health = await response.json()
                    else:
                        health = {"status": "unhealthy"}
        
        except Exception as e:
            return {"error": f"Failed to generate report: {str(e)}"}
        
        # Compile report
        report = {
            "timestamp": datetime.now().isoformat(),
            "api_url": self.api_url,
            "health_status": health,
            "performance_metrics": metrics,
            "benchmark_results": benchmark,
            "recommendations": self._generate_recommendations(metrics, benchmark)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: Dict[str, Any], benchmark: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        perf_metrics = metrics.get('performance', {})
        cache_metrics = metrics.get('cache', {})
        resource_metrics = perf_metrics.get('resource_usage', {})
        
        # Cache recommendations
        hit_rate = cache_metrics.get('hit_rate', 0)
        if hit_rate < 0.5:
            recommendations.append("Consider increasing cache size or TTL - low cache hit rate detected")
        
        # Resource recommendations
        cpu_usage = resource_metrics.get('cpu_percent', 0)
        memory_usage = resource_metrics.get('memory_percent', 0)
        
        if cpu_usage > 80:
            recommendations.append("High CPU usage detected - consider scaling up or optimizing model")
        
        if memory_usage > 85:
            recommendations.append("High memory usage detected - consider memory optimization")
        
        # Response time recommendations
        response_time = perf_metrics.get('response_time', {}).get('avg_ms', 0)
        if response_time > 1000:
            recommendations.append("High response time - consider model optimization or caching")
        
        # Error rate recommendations
        error_rate = perf_metrics.get('error_rate', 0)
        if error_rate > 0.01:
            recommendations.append("Elevated error rate detected - investigate error causes")
        
        # Benchmark recommendations
        if 'speedup' in benchmark and benchmark['speedup'] < 1.2:
            recommendations.append("Model optimization shows minimal improvement - consider alternative optimizations")
        
        if not recommendations:
            recommendations.append("Performance looks good - no immediate optimizations needed")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Save report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Performance report saved to {filename}")

async def main():
    parser = argparse.ArgumentParser(description="Performance Monitoring Dashboard")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--mode", choices=["dashboard", "report"], default="dashboard", 
                       help="Mode: dashboard (real-time) or report (one-time)")
    parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    if args.mode == "dashboard":
        dashboard = PerformanceDashboard(args.url, args.interval)
        dashboard.start_monitoring()
    
    elif args.mode == "report":
        reporter = PerformanceReporter(args.url)
        report = await reporter.generate_report()
        
        if "error" in report:
            print(f"❌ {report['error']}")
            return 1
        
        # Print summary
        print("📊 Performance Report Summary:")
        print(f"   API Status: {report['health_status'].get('status', 'unknown')}")
        
        perf_metrics = report.get('performance_metrics', {}).get('performance', {})
        if perf_metrics:
            print(f"   Request Rate: {perf_metrics.get('requests_per_second', 0):.1f} req/s")
            print(f"   Error Rate: {perf_metrics.get('error_rate', 0):.3f}")
            
            response_time = perf_metrics.get('response_time', {})
            if response_time:
                print(f"   Avg Response Time: {response_time.get('avg_ms', 0):.1f}ms")
        
        # Print recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Save report
        reporter.save_report(report, args.output)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))