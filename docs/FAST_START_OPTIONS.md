# ⚡ Fast Start Options - Choose Your Speed

You have 3 options to start the system, from fastest to most complete:

## Option 1: Local API (Fastest - 2 minutes) ⚡

**Best for:** Quick testing, development, trying the API immediately

**Time:** ~2 minutes (no Docker build needed!)

```bash
run-api-local.bat
```

**What it does:**
- Installs minimal Python packages locally
- Runs API directly with Python
- Uses your trained model
- No Docker build required!

**Pros:**
- ✅ Fastest option (2 minutes)
- ✅ No Docker build time
- ✅ Easy to debug
- ✅ Hot reload on code changes

**Cons:**
- ❌ Only API service (no MLflow, monitoring, etc.)
- ❌ Requires Python installed locally
- ❌ Not production-ready

**Access:**
- API: http://localhost:8004/docs

---

## Option 2: Quick Docker Start (Fast - 5 minutes) 🚀

**Best for:** Testing with infrastructure, but don't need all services yet

**Time:** ~5 minutes (builds only deployment service)

```bash
docker-quick-start.bat
```

**What it does:**
- Starts PostgreSQL, MinIO, MLflow (pre-built images)
- Builds only deployment service
- Skips training, monitoring, etc. for now

**Pros:**
- ✅ Fast (5 minutes)
- ✅ Has database and storage
- ✅ Has MLflow UI
- ✅ Containerized and isolated

**Cons:**
- ❌ Missing some services initially
- ❌ Still requires Docker build

**Access:**
- API: http://localhost:8004/docs
- MLflow: http://localhost:5000
- MinIO: http://localhost:9001

---

## Option 3: Full Docker Start (Complete - 10-15 minutes) 🏗️

**Best for:** Complete system, production-like environment

**Time:** ~10-15 minutes (builds all 5 services)

```bash
docker-start.bat
```

**What it does:**
- Builds all 5 microservices
- Starts complete MLOps stack
- Full monitoring and tracking

**Pros:**
- ✅ Complete system
- ✅ All services available
- ✅ Production-ready
- ✅ Full monitoring

**Cons:**
- ❌ Longer build time (10-15 min first time)
- ❌ Uses more resources

**Access:**
- API: http://localhost:8004/docs
- MLflow: http://localhost:5000
- MinIO: http://localhost:9001
- Data Pipeline: http://localhost:8001/docs
- Training: http://localhost:8002/docs
- Model Registry: http://localhost:8003/docs
- Monitoring: http://localhost:8005/docs

---

## Comparison Table

| Feature | Local API | Quick Docker | Full Docker |
|---------|-----------|--------------|-------------|
| **Time** | 2 min | 5 min | 10-15 min |
| **API** | ✅ | ✅ | ✅ |
| **MLflow** | ❌ | ✅ | ✅ |
| **Database** | ❌ | ✅ | ✅ |
| **Storage** | ❌ | ✅ | ✅ |
| **Monitoring** | ❌ | ❌ | ✅ |
| **Training** | ❌ | ❌ | ✅ |
| **Hot Reload** | ✅ | ❌ | ❌ |
| **Production Ready** | ❌ | Partial | ✅ |

---

## Recommended Workflow

### For Quick Testing (Right Now!)
```bash
# Start immediately with local API
run-api-local.bat

# Test your model
# Open http://localhost:8004/docs
# Upload an X-ray image
# Get predictions!
```

### For Development
```bash
# Use quick Docker start
docker-quick-start.bat

# Later, build other services as needed
docker-compose build training
docker-compose up -d training
```

### For Production
```bash
# Build everything once
docker-start.bat

# Or deploy to Kubernetes
k8s-deploy.bat
```

---

## Speed Tips

### If You Already Started a Build

**Don't cancel!** Docker caches layers. Even if you restart, it will use cached layers and be faster.

**Check what's cached:**
```bash
docker images
```

**Continue from where it stopped:**
```bash
docker-compose build --no-cache deployment  # Only rebuild deployment
docker-compose up -d
```

### Optimize Future Builds

**1. Keep base images:**
```bash
# Don't delete these
docker images | findstr python
docker images | findstr postgres
docker images | findstr minio
```

**2. Use build cache:**
```bash
# Normal build (uses cache)
docker-compose build

# Only if really needed
docker-compose build --no-cache
```

**3. Build in parallel:**
```bash
# Build multiple services at once
docker-compose build --parallel
```

---

## My Recommendation for You

Since you want to test quickly and already have the model:

### **Start with Option 1 (Local API)** ⚡

```bash
run-api-local.bat
```

**Why:**
- Ready in 2 minutes
- No Docker build wait
- Perfect for testing your trained model
- Can always build Docker later

**Then later, when you have time:**
```bash
# Build Docker in background while you work
docker-compose build

# Or just the essential services
docker-quick-start.bat
```

---

## Troubleshooting

### Local API Issues

**Missing packages:**
```bash
pip install torch torchvision fastapi uvicorn pillow opencv-python-headless python-multipart
```

**Port in use:**
```bash
# Change port in run-api-local.bat
# Change 8004 to 8005 or any free port
```

### Docker Issues

**Build taking too long:**
- Use `docker-quick-start.bat` instead
- Or run local API while Docker builds in background

**Out of memory:**
- Close other applications
- Increase Docker Desktop memory
- Use local API option instead

---

## Summary

**Want to test NOW?** → `run-api-local.bat` (2 min)

**Want Docker but fast?** → `docker-quick-start.bat` (5 min)

**Want everything?** → `docker-start.bat` (10-15 min)

**My advice:** Start with local API, test your model, then build Docker in the background while you explore! 🚀
