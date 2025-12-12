#!/usr/bin/env python3
"""
MLflow UI Launcher
Simple script to launch MLflow tracking UI for experiment monitoring
"""

import subprocess
import sys
import os
import argparse
import logging
from pathlib import Path

def launch_mlflow_ui(
    tracking_uri: str = "sqlite:///mlflow.db",
    host: str = "127.0.0.1",
    port: int = 5000,
    backend_store_uri: str = None,
    default_artifact_root: str = None
):
    """
    Launch MLflow UI server
    
    Args:
        tracking_uri: MLflow tracking URI
        host: Host to bind the server to
        port: Port to run the server on
        backend_store_uri: Backend store URI (overrides tracking_uri)
        default_artifact_root: Default artifact root directory
    """
    
    print(f"🚀 Launching MLflow UI...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Tracking URI: {tracking_uri}")
    
    # Build MLflow UI command
    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--host", host,
        "--port", str(port)
    ]
    
    # Add backend store URI if provided
    if backend_store_uri:
        cmd.extend(["--backend-store-uri", backend_store_uri])
    elif tracking_uri:
        cmd.extend(["--backend-store-uri", tracking_uri])
    
    # Add default artifact root if provided
    if default_artifact_root:
        cmd.extend(["--default-artifact-root", default_artifact_root])
    
    print(f"   Command: {' '.join(cmd)}")
    print(f"\n📊 MLflow UI will be available at: http://{host}:{port}")
    print(f"   Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Launch MLflow UI
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 MLflow UI server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch MLflow UI: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ MLflow not found. Please install with: pip install mlflow")
        sys.exit(1)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Launch MLflow UI for experiment tracking")
    
    parser.add_argument("--host", default="127.0.0.1", 
                       help="Host to bind the server to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port to run the server on (default: 5000)")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db",
                       help="MLflow tracking URI (default: sqlite:///mlflow.db)")
    parser.add_argument("--backend-store-uri", default=None,
                       help="Backend store URI (overrides tracking-uri)")
    parser.add_argument("--artifact-root", default=None,
                       help="Default artifact root directory")
    
    args = parser.parse_args()
    
    # Check if MLflow database exists
    if args.tracking_uri.startswith("sqlite:///"):
        db_path = args.tracking_uri.replace("sqlite:///", "")
        if not Path(db_path).exists():
            print(f"⚠️  MLflow database not found: {db_path}")
            print(f"   The database will be created when you run your first experiment")
            print(f"   You can still launch the UI - it will show an empty state")
            
            response = input("\nContinue anyway? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Cancelled.")
                sys.exit(0)
    
    # Launch MLflow UI
    launch_mlflow_ui(
        tracking_uri=args.tracking_uri,
        host=args.host,
        port=args.port,
        backend_store_uri=args.backend_store_uri,
        default_artifact_root=args.artifact_root
    )

if __name__ == "__main__":
    main()