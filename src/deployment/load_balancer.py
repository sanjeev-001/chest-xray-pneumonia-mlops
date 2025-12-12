"""
Load Balancer for Blue-Green Deployments
Routes traffic between active and inactive deployments
"""

import asyncio
import logging
import json
from typing import Dict, Optional, List
from datetime import datetime
import time

import httpx
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from deployment_manager import DeploymentManager, DeploymentStatus

logger = logging.getLogger(__name__)

class LoadBalancer:
    """
    Load balancer for blue-green deployments
    Routes traffic to active deployment with health checking
    """
    
    def __init__(self, deployment_manager: DeploymentManager):
        self.deployment_manager = deployment_manager
        self.health_check_interval = 30  # seconds
        self.request_timeout = 30  # seconds
        
        # Traffic routing
        self.active_backend = None
        self.backup_backend = None
        
        # Health status
        self.backend_health = {}
        
        # Metrics
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        
        # Start health checking
        asyncio.create_task(self._health_check_loop())
        
        logger.info("LoadBalancer initialized")
    
    async def _health_check_loop(self):
        """Continuous health checking of backends"""
        while True:
            try:
                await self._update_backends()
                await self._check_backend_health()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)
    
    async def _update_backends(self):
        """Update active and backup backends"""
        active_deployment = self.deployment_manager.get_active_deployment()
        
        if active_deployment and active_deployment.status == DeploymentStatus.ACTIVE:
            self.active_backend = f"http://localhost:{active_deployment.port}"
        else:
            self.active_backend = None
        
        # Find backup (most recent inactive healthy deployment)
        deployments = self.deployment_manager.list_deployments()
        backup_candidates = [
            dep for dep in deployments
            if (dep.status == DeploymentStatus.INACTIVE and 
                dep.health_status == "healthy" and
                dep.deployment_id != (active_deployment.deployment_id if active_deployment else None))
        ]
        
        if backup_candidates:
            backup = max(backup_candidates, key=lambda d: d.updated_at or datetime.min)
            self.backup_backend = f"http://localhost:{backup.port}"
        else:
            self.backup_backend = None
    
    async def _check_backend_health(self):
        """Check health of all backends"""
        backends = []
        if self.active_backend:
            backends.append(("active", self.active_backend))
        if self.backup_backend:
            backends.append(("backup", self.backup_backend))
        
        for backend_type, backend_url in backends:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{backend_url}/health")
                    
                    if response.status_code == 200:
                        health_data = response.json()
                        is_healthy = (health_data.get("status") == "healthy" and 
                                    health_data.get("model_loaded", False))
                        self.backend_health[backend_url] = is_healthy
                    else:
                        self.backend_health[backend_url] = False
                        
            except Exception as e:
                logger.warning(f"Health check failed for {backend_type} backend {backend_url}: {e}")
                self.backend_health[backend_url] = False
    
    async def route_request(self, request: Request) -> Optional[str]:
        """Determine which backend to route request to"""
        # Try active backend first
        if self.active_backend and self.backend_health.get(self.active_backend, False):
            return self.active_backend
        
        # Fallback to backup if active is unhealthy
        if self.backup_backend and self.backend_health.get(self.backup_backend, False):
            logger.warning("Routing to backup backend - active backend unhealthy")
            return self.backup_backend
        
        # No healthy backends available
        return None
    
    async def proxy_request(self, request: Request, target_url: str):
        """Proxy request to target backend"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Prepare request
            url = f"{target_url}{request.url.path}"
            if request.url.query:
                url += f"?{request.url.query}"
            
            headers = dict(request.headers)
            # Remove hop-by-hop headers
            headers.pop("host", None)
            headers.pop("connection", None)
            
            # Get request body
            body = await request.body()
            
            # Make request to backend
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=headers,
                    content=body
                )
                
                # Record response time
                response_time = (time.time() - start_time) * 1000
                self.response_times.append(response_time)
                
                # Keep only last 1000 response times
                if len(self.response_times) > 1000:
                    self.response_times = self.response_times[-1000:]
                
                # Prepare response headers
                response_headers = dict(response.headers)
                # Remove hop-by-hop headers
                response_headers.pop("connection", None)
                response_headers.pop("transfer-encoding", None)
                
                # Add load balancer headers
                response_headers["X-Load-Balancer"] = "chest-xray-lb"
                response_headers["X-Backend-URL"] = target_url
                response_headers["X-Response-Time"] = str(response_time)
                
                return StreamingResponse(
                    response.aiter_bytes(),
                    status_code=response.status_code,
                    headers=response_headers,
                    media_type=response.headers.get("content-type")
                )
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Proxy request failed: {e}")
            raise HTTPException(status_code=502, detail="Backend unavailable")
    
    def get_stats(self) -> Dict:
        """Get load balancer statistics"""
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_response_time_ms": avg_response_time,
            "active_backend": self.active_backend,
            "backup_backend": self.backup_backend,
            "backend_health": self.backend_health,
            "uptime_seconds": time.time() - start_time if 'start_time' in globals() else 0
        }

# FastAPI app
app = FastAPI(
    title="Chest X-Ray Load Balancer",
    description="Load balancer for blue-green deployments",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global load balancer instance
lb = None
start_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize load balancer on startup"""
    global lb
    deployment_manager = DeploymentManager()
    lb = LoadBalancer(deployment_manager)
    logger.info("Load balancer started")

@app.get("/lb/health")
async def lb_health():
    """Load balancer health check"""
    return {
        "status": "healthy",
        "active_backend": lb.active_backend if lb else None,
        "backup_backend": lb.backup_backend if lb else None,
        "uptime_seconds": time.time() - start_time
    }

@app.get("/lb/stats")
async def lb_stats():
    """Load balancer statistics"""
    if not lb:
        raise HTTPException(status_code=503, detail="Load balancer not initialized")
    
    return lb.get_stats()

@app.get("/lb/backends")
async def lb_backends():
    """List backend status"""
    if not lb:
        raise HTTPException(status_code=503, detail="Load balancer not initialized")
    
    return {
        "active_backend": lb.active_backend,
        "backup_backend": lb.backup_backend,
        "backend_health": lb.backend_health,
        "deployments": [
            {
                "deployment_id": dep.deployment_id,
                "model_version": dep.model_version,
                "status": dep.status.value,
                "port": dep.port,
                "health_status": dep.health_status,
                "url": f"http://localhost:{dep.port}"
            }
            for dep in lb.deployment_manager.list_deployments()
        ]
    }

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_all(request: Request, path: str):
    """Proxy all requests to appropriate backend"""
    if not lb:
        raise HTTPException(status_code=503, detail="Load balancer not initialized")
    
    # Route request
    backend_url = await lb.route_request(request)
    
    if not backend_url:
        raise HTTPException(status_code=503, detail="No healthy backends available")
    
    # Proxy request
    return await lb.proxy_request(request, backend_url)

def main():
    """Run load balancer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chest X-Ray Load Balancer")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=80, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    print("🔄 Starting Chest X-Ray Load Balancer...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Management UI: http://{args.host}:{args.port}/docs")
    print("=" * 50)
    
    uvicorn.run(
        "load_balancer:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()