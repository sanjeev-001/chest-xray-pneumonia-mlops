#!/usr/bin/env python3
"""
Deployment CLI for Chest X-Ray Pneumonia Detection API
Command-line interface for managing blue-green deployments
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from deployment.deployment_manager import DeploymentManager, DeploymentConfig, DeploymentStatus

def format_deployment_info(deployment):
    """Format deployment info for display"""
    status_emoji = {
        DeploymentStatus.PENDING: "⏳",
        DeploymentStatus.DEPLOYING: "🚀",
        DeploymentStatus.ACTIVE: "✅",
        DeploymentStatus.INACTIVE: "⏸️",
        DeploymentStatus.FAILED: "❌",
        DeploymentStatus.ROLLING_BACK: "🔄"
    }
    
    emoji = status_emoji.get(deployment.status, "❓")
    
    print(f"{emoji} Deployment: {deployment.deployment_id}")
    print(f"   Model Version: {deployment.model_version}")
    print(f"   Status: {deployment.status.value}")
    print(f"   Port: {deployment.port}")
    print(f"   Health: {deployment.health_status}")
    print(f"   Created: {deployment.created_at}")
    if deployment.error_message:
        print(f"   Error: {deployment.error_message}")
    print()

def cmd_deploy(args):
    """Deploy new model version"""
    print("🚀 Starting deployment...")
    
    # Validate model file
    if not Path(args.model_path).exists():
        print(f"❌ Model file not found: {args.model_path}")
        return 1
    
    # Create deployment manager
    manager = DeploymentManager(
        base_port=args.base_port,
        docker_image=args.docker_image
    )
    
    # Create deployment config
    config = DeploymentConfig(
        model_path=args.model_path,
        model_version=args.model_version,
        image_tag=f"{args.docker_image}:{args.model_version}",
        port=0,  # Will be assigned by manager
        health_check_url="",  # Will be set by manager
        health_check_timeout=args.health_timeout,
        health_check_retries=args.health_retries,
        environment=args.environment
    )
    
    # Start deployment
    deployment_id = manager.deploy(config)
    
    print(f"📦 Deployment started: {deployment_id}")
    
    # Wait for completion if requested
    if args.wait:
        print("⏳ Waiting for deployment to complete...")
        
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            deployment = manager.get_deployment_status(deployment_id)
            if deployment:
                if deployment.status == DeploymentStatus.ACTIVE:
                    print("✅ Deployment completed successfully!")
                    format_deployment_info(deployment)
                    
                    # Auto-promote if requested
                    if args.auto_promote:
                        print("🔄 Auto-promoting deployment...")
                        if manager.promote_deployment(deployment_id):
                            print("✅ Deployment promoted to active!")
                        else:
                            print("❌ Failed to promote deployment")
                            return 1
                    
                    return 0
                elif deployment.status == DeploymentStatus.FAILED:
                    print("❌ Deployment failed!")
                    format_deployment_info(deployment)
                    return 1
            
            time.sleep(5)
        
        print("⏰ Deployment timeout reached")
        return 1
    
    return 0

def cmd_list(args):
    """List all deployments"""
    manager = DeploymentManager()
    deployments = manager.list_deployments()
    
    if not deployments:
        print("📭 No deployments found")
        return 0
    
    print(f"📋 Found {len(deployments)} deployments:")
    print()
    
    # Sort by creation time
    deployments.sort(key=lambda d: d.created_at or datetime.min, reverse=True)
    
    active_deployment = manager.get_active_deployment()
    
    for deployment in deployments:
        if active_deployment and deployment.deployment_id == active_deployment.deployment_id:
            print("🌟 ACTIVE DEPLOYMENT:")
        
        format_deployment_info(deployment)
    
    return 0

def cmd_status(args):
    """Get deployment status"""
    manager = DeploymentManager()
    
    if args.deployment_id:
        deployment = manager.get_deployment_status(args.deployment_id)
        if deployment:
            format_deployment_info(deployment)
        else:
            print(f"❌ Deployment {args.deployment_id} not found")
            return 1
    else:
        # Show active deployment
        active = manager.get_active_deployment()
        if active:
            print("🌟 Active Deployment:")
            format_deployment_info(active)
        else:
            print("📭 No active deployment")
    
    return 0

def cmd_promote(args):
    """Promote deployment to active"""
    manager = DeploymentManager()
    
    print(f"🔄 Promoting deployment {args.deployment_id}...")
    
    if manager.promote_deployment(args.deployment_id):
        print("✅ Deployment promoted successfully!")
        
        # Show new active deployment
        active = manager.get_active_deployment()
        if active:
            format_deployment_info(active)
    else:
        print("❌ Failed to promote deployment")
        return 1
    
    return 0

def cmd_rollback(args):
    """Rollback to previous deployment"""
    manager = DeploymentManager()
    
    print("🔄 Rolling back to previous deployment...")
    
    if manager.rollback():
        print("✅ Rollback completed successfully!")
        
        # Show new active deployment
        active = manager.get_active_deployment()
        if active:
            format_deployment_info(active)
    else:
        print("❌ Rollback failed")
        return 1
    
    return 0

def cmd_stop(args):
    """Stop deployment"""
    manager = DeploymentManager()
    
    print(f"⏹️ Stopping deployment {args.deployment_id}...")
    
    if manager.stop_deployment(args.deployment_id):
        print("✅ Deployment stopped successfully!")
    else:
        print("❌ Failed to stop deployment")
        return 1
    
    return 0

def cmd_cleanup(args):
    """Clean up old deployments"""
    manager = DeploymentManager()
    
    print(f"🧹 Cleaning up old deployments (keeping {args.keep} most recent)...")
    
    manager.cleanup_old_deployments(keep_count=args.keep)
    
    print("✅ Cleanup completed!")
    return 0

def cmd_health(args):
    """Check health of deployments"""
    manager = DeploymentManager()
    deployments = manager.list_deployments()
    
    print("🏥 Health Check Results:")
    print()
    
    for deployment in deployments:
        if deployment.status == DeploymentStatus.ACTIVE:
            # Perform live health check
            healthy = manager._health_check(deployment.port)
            status = "✅ Healthy" if healthy else "❌ Unhealthy"
            print(f"{deployment.deployment_id}: {status} (Port: {deployment.port})")
        else:
            print(f"{deployment.deployment_id}: ⏸️ Inactive")
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Deployment CLI for Chest X-Ray Pneumonia Detection API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy new model version
  python deploy_cli.py deploy --model-path models/best_model.pth --model-version v1.2.0

  # Deploy and auto-promote
  python deploy_cli.py deploy --model-path models/best_model.pth --model-version v1.2.0 --auto-promote

  # List all deployments
  python deploy_cli.py list

  # Check status of specific deployment
  python deploy_cli.py status --deployment-id deploy_1234567890

  # Promote deployment to active
  python deploy_cli.py promote deploy_1234567890

  # Rollback to previous deployment
  python deploy_cli.py rollback

  # Stop deployment
  python deploy_cli.py stop deploy_1234567890

  # Clean up old deployments
  python deploy_cli.py cleanup --keep 3
        """
    )
    
    # Global options
    parser.add_argument("--base-port", type=int, default=8000, help="Base port for deployments")
    parser.add_argument("--docker-image", default="chest-xray-api", help="Docker image name")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy new model version')
    deploy_parser.add_argument('--model-path', required=True, help='Path to model file')
    deploy_parser.add_argument('--model-version', required=True, help='Model version')
    deploy_parser.add_argument('--environment', default='staging', choices=['staging', 'production'], help='Environment')
    deploy_parser.add_argument('--health-timeout', type=int, default=30, help='Health check timeout')
    deploy_parser.add_argument('--health-retries', type=int, default=5, help='Health check retries')
    deploy_parser.add_argument('--wait', action='store_true', help='Wait for deployment to complete')
    deploy_parser.add_argument('--auto-promote', action='store_true', help='Auto-promote successful deployment')
    deploy_parser.set_defaults(func=cmd_deploy)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all deployments')
    list_parser.set_defaults(func=cmd_list)
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get deployment status')
    status_parser.add_argument('--deployment-id', help='Specific deployment ID (default: active)')
    status_parser.set_defaults(func=cmd_status)
    
    # Promote command
    promote_parser = subparsers.add_parser('promote', help='Promote deployment to active')
    promote_parser.add_argument('deployment_id', help='Deployment ID to promote')
    promote_parser.set_defaults(func=cmd_promote)
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to previous deployment')
    rollback_parser.set_defaults(func=cmd_rollback)
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop deployment')
    stop_parser.add_argument('deployment_id', help='Deployment ID to stop')
    stop_parser.set_defaults(func=cmd_stop)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old deployments')
    cleanup_parser.add_argument('--keep', type=int, default=3, help='Number of deployments to keep')
    cleanup_parser.set_defaults(func=cmd_cleanup)
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check health of deployments')
    health_parser.set_defaults(func=cmd_health)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())