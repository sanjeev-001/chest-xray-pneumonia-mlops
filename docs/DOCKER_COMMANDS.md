# Docker Commands Guide - After Fixes

## ✅ Option 1: Docker Build & Run (What You Used Before)

**Step 1: Build the Docker Image**
```cmd
cd "C:\Users\sanje\Downloads\ml ops project"
docker build -t mlopsproject-deployment:latest -f deployment/Dockerfile .
```

**Step 2: Run the Container**
```cmd
docker run -d -p 8004:8004 --name mlops-api -v "%CD%\models:/app/models" mlopsproject-deployment:latest
```

**Step 3: Check if it's running**
```cmd
docker ps
docker logs mlops-api
```

**Step 4: Test the API**
- Open browser: http://localhost:8004/docs
- Or test health: http://localhost:8004/health

**To stop:**
```cmd
docker stop mlops-api
docker rm mlops-api
```

---

## ✅ Option 2: Docker Compose (Recommended - Easier)

**Build and start everything:**
```cmd
cd "C:\Users\sanje\Downloads\ml ops project"
docker-compose build deployment
docker-compose up -d deployment
```

**Or use the quick start batch file:**
```cmd
docker-quick-start.bat
```

**Check status:**
```cmd
docker-compose ps
```

**View logs:**
```cmd
docker-compose logs -f deployment
```

**Stop:**
```cmd
docker-compose stop deployment
```

**Access API:**
- http://localhost:8004/docs
- http://localhost:8004/health

---

## ✅ Option 3: Run Locally Without Docker (Fastest for Testing)

**Just run this batch file:**
```cmd
run-api-local.bat
```

**Or manually:**
```cmd
cd "C:\Users\sanje\Downloads\ml ops project"
python -m uvicorn deployment.api:app --host 0.0.0.0 --port 8004 --reload
```

**Access API:**
- http://localhost:8004/docs
- http://localhost:8004/health

---

## 🔍 Important Notes After Fixes

1. **Port Number**: The API now runs on port **8004** (not 8000)
   - Dockerfile uses port 8004
   - docker-compose uses port 8004
   - Make sure to use port 8004 in URLs

2. **Model File**: Make sure your model file exists:
   ```
   models\best_chest_xray_model.pth
   ```

3. **Test Predictions**: After starting, test with:
   - Known NORMAL image → should predict "NORMAL"
   - Known PNEUMONIA image → should predict "PNEUMONIA"

---

## 🚀 Quick Test Commands

**Test health endpoint:**
```cmd
curl http://localhost:8004/health
```

**Test prediction (if you have curl):**
```cmd
curl -X POST "http://localhost:8004/predict" -H "Content-Type: multipart/form-data" -F "file=@path\to\your\image.jpg"
```

**Or use the test batch file:**
```cmd
test-docker-api.bat
```

---

## 🐛 Troubleshooting

**If Docker build fails:**
```cmd
# Clean build (no cache)
docker build --no-cache -t mlopsproject-deployment:latest -f deployment/Dockerfile .
```

**If container won't start:**
```cmd
# Check logs
docker logs mlops-api

# Check if port is already in use
netstat -ano | findstr :8004
```

**If model not found:**
- Make sure `models\best_chest_xray_model.pth` exists
- Check the volume mount path is correct

---

## 📝 Summary

**For Quick Testing:** Use `run-api-local.bat` (no Docker needed)

**For Docker:** Use your original command but add the run command:
```cmd
docker build -t mlopsproject-deployment:latest -f deployment/Dockerfile .
docker run -d -p 8004:8004 --name mlops-api -v "%CD%\models:/app/models" mlopsproject-deployment:latest
```

**For Full System:** Use `docker-compose up -d deployment`

All options will now work correctly with the fixes applied! ✅

