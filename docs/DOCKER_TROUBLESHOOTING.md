# Docker Deployment Troubleshooting

## Common Issues and Solutions

### 1. Build Error: "COPY models/ failed" or Dependency Conflicts

**Error Message:**
```
ERROR: Build failed!
target deployment: failed to solve: failed to compute cache key
```
OR
```
ERROR: Could not find a version that satisfies the requirement opencv-python~=4.5.0
```

**Causes:** 
1. The Dockerfile was trying to copy the models directory during build
2. Dependency conflicts between packages (opencv-python, torch, etc.)

**Solutions:** ✅ **FIXED** 
1. Models are now mounted as a volume instead of copied during build
2. Each service now has its own minimal requirements.txt file
3. Using `opencv-python-headless` for Docker (no GUI dependencies)

The updated configuration:
- Dockerfile creates an empty `/app/models` directory
- docker-compose.yml mounts your local `./models` directory as a volume
- Model is available at runtime without being baked into the image

**Verify the fix:**
```bash
# Check if model exists
dir models\best_chest_xray_model.pth

# If model exists, proceed with deployment
docker-start.bat
```

### 2. Model File Not Found

**Error:** API starts but predictions fail with "Model not loaded"

**Solution:**
```bash
# Ensure model file exists
dir models\best_chest_xray_model.pth

# If missing, you need to:
# Option 1: Train a model
python training/train_model.py

# Option 2: Copy a pre-trained model
# Place your model file in: models/best_chest_xray_model.pth
```

### 3. Port Already in Use

**Error:** "Port 8004 is already allocated"

**Solution:**
```bash
# Option 1: Stop the conflicting service
docker ps
docker stop <container-id>

# Option 2: Change the port in docker-compose.yml
# Edit the deployment service:
ports:
  - "8005:8004"  # Use different external port
```

### 4. Out of Memory

**Error:** Container crashes or build fails with memory errors

**Solution:**
1. Open Docker Desktop
2. Go to Settings → Resources
3. Increase Memory to 8GB or more
4. Click "Apply & Restart"
5. Try again: `docker-start.bat`

### 5. Services Not Starting

**Error:** Services show as "Exited" or "Restarting"

**Solution:**
```bash
# Check logs for specific service
docker-compose logs deployment
docker-compose logs postgres
docker-compose logs minio

# Common fixes:
# 1. Wait longer (services need time to initialize)
timeout /t 60

# 2. Restart specific service
docker-compose restart deployment

# 3. Clean restart
docker-compose down
docker-compose up -d
```

### 6. Database Connection Failed

**Error:** "could not connect to server: Connection refused"

**Solution:**
```bash
# Wait for PostgreSQL to be ready
docker-compose logs postgres

# Look for: "database system is ready to accept connections"

# If not ready, wait and restart
timeout /t 30
docker-compose restart deployment
```

### 7. MinIO Connection Failed

**Error:** "Connection refused" when accessing MinIO

**Solution:**
```bash
# Check MinIO is running
docker-compose ps minio

# Check MinIO logs
docker-compose logs minio

# Restart MinIO
docker-compose restart minio

# Access MinIO console
# http://localhost:9001
# Username: minioadmin
# Password: minioadmin
```

### 8. Build Cache Issues

**Error:** Build uses old cached layers

**Solution:**
```bash
# Build without cache
docker-compose build --no-cache

# Or rebuild specific service
docker-compose build --no-cache deployment
```

### 9. Permission Denied Errors

**Error:** "Permission denied" when accessing files

**Solution:**
```bash
# On Windows, ensure Docker has access to the drive
# Docker Desktop → Settings → Resources → File Sharing
# Add your project directory

# Rebuild with proper permissions
docker-compose down
docker-compose build
docker-compose up -d
```

### 10. API Returns 503 Service Unavailable

**Error:** API is running but returns 503

**Cause:** Model not loaded or service not ready

**Solution:**
```bash
# Check if model exists
dir models\best_chest_xray_model.pth

# Check API logs
docker-compose logs deployment

# Wait for model to load (can take 30-60 seconds)
timeout /t 60

# Test health endpoint
curl http://localhost:8004/health
```

## Quick Diagnostic Commands

```bash
# Check all services status
docker-compose ps

# View all logs
docker-compose logs

# View specific service logs
docker-compose logs -f deployment

# Check resource usage
docker stats

# Restart everything
docker-compose restart

# Clean restart (removes containers but keeps volumes)
docker-compose down
docker-compose up -d

# Nuclear option (removes everything including data)
docker-compose down -v
docker-compose up -d
```

## Validation Checklist

Before deploying, ensure:

- [ ] Docker Desktop is running
- [ ] Model file exists: `models/best_chest_xray_model.pth`
- [ ] `.env` file exists (copy from `.env.example`)
- [ ] Ports 8001-8005, 5000, 9000-9001 are available
- [ ] At least 8GB RAM allocated to Docker
- [ ] At least 20GB disk space available

Run validation:
```bash
validate-deployment-setup.bat
```

## Getting Help

1. **Check logs first:**
   ```bash
   docker-compose logs -f
   ```

2. **Verify services are running:**
   ```bash
   docker-compose ps
   ```

3. **Test individual services:**
   ```bash
   curl http://localhost:8004/health
   curl http://localhost:8001/health
   curl http://localhost:8005/health
   ```

4. **Review documentation:**
   - `DOCKER_DEPLOYMENT_GUIDE.md` - Complete guide
   - `QUICK_DOCKER_START.md` - Quick start
   - `START_HERE.md` - Getting started

## Success Indicators

When everything is working:

✅ All services show "Up" status:
```bash
docker-compose ps
```

✅ Health checks pass:
```bash
curl http://localhost:8004/health
# Returns: {"status":"healthy","model_loaded":true,...}
```

✅ API docs accessible:
```
http://localhost:8004/docs
```

✅ Can make predictions:
- Upload image via API docs
- Receive prediction response

## Still Having Issues?

1. Run the validation script:
   ```bash
   validate-deployment-setup.bat
   ```

2. Check the complete logs:
   ```bash
   docker-compose logs > logs.txt
   ```

3. Review the error messages in the logs

4. Try a clean restart:
   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

5. Ensure your model file is in the correct location:
   ```bash
   dir models\best_chest_xray_model.pth
   ```
