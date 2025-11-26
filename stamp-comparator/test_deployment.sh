#!/bin/bash

echo "=========================================="
echo "Deployment Validation Script"
echo "=========================================="
echo ""

BACKEND_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:3000"
ERRORS=0

# Test 1: Backend Health Check
echo "Test 1: Backend Health Check..."
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $BACKEND_URL/health)
if [ "$RESPONSE" = "200" ]; then
    echo "✓ Backend is responding"
else
    echo "✗ Backend health check failed (HTTP $RESPONSE)"
    ERRORS=$((ERRORS + 1))
fi

# Test 2: Frontend Accessibility
echo ""
echo "Test 2: Frontend Accessibility..."
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $FRONTEND_URL)
if [ "$RESPONSE" = "200" ]; then
    echo "✓ Frontend is accessible"
else
    echo "✗ Frontend is not accessible (HTTP $RESPONSE)"
    ERRORS=$((ERRORS + 1))
fi

# Test 3: API Documentation
echo ""
echo "Test 3: API Documentation..."
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $BACKEND_URL/docs)
if [ "$RESPONSE" = "200" ]; then
    echo "✓ API docs are accessible"
else
    echo "✗ API docs are not accessible"
    ERRORS=$((ERRORS + 1))
fi

# Test 4: Check Docker Containers
echo ""
echo "Test 4: Docker Containers Status..."
if command -v docker-compose &> /dev/null; then
    BACKEND_STATUS=$(docker-compose ps -q backend 2>/dev/null | xargs docker inspect -f '{{.State.Status}}' 2>/dev/null)
    FRONTEND_STATUS=$(docker-compose ps -q frontend 2>/dev/null | xargs docker inspect -f '{{.State.Status}}' 2>/dev/null)
    
    if [ "$BACKEND_STATUS" = "running" ]; then
        echo "✓ Backend container is running"
    else
        echo "✗ Backend container is not running ($BACKEND_STATUS)"
        ERRORS=$((ERRORS + 1))
    fi
    
    if [ "$FRONTEND_STATUS" = "running" ]; then
        echo "✓ Frontend container is running"
    else
        echo "✗ Frontend container is not running ($FRONTEND_STATUS)"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "⚠ Docker Compose not found, skipping container checks"
fi

# Test 5: Check Required Directories
echo ""
echo "Test 5: Required Directories..."
DIRS=("data" "models" "logs" "data/reference" "data/test")
for DIR in "${DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "✓ Directory exists: $DIR"
    else
        echo "✗ Missing directory: $DIR"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 6: Test Default Config Endpoint
echo ""
echo "Test 6: Default Configuration..."
RESPONSE=$(curl -s $BACKEND_URL/api/config/default)
if echo "$RESPONSE" | grep -q "alignment"; then
    echo "✓ Default config endpoint working"
else
    echo "✗ Default config endpoint failed"
    ERRORS=$((ERRORS + 1))
fi

# Test 7: Check Model Directories
echo ""
echo "Test 7: Model Directories..."
MODEL_DIRS=("models/siamese" "models/cnn_detector" "models/autoencoder")
for DIR in "${MODEL_DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        echo "✓ Model directory exists: $DIR"
    else
        echo "⚠ Model directory missing: $DIR (models optional)"
    fi
done

# Test 8: Check Disk Space
echo ""
echo "Test 8: Disk Space..."
AVAILABLE=$(df -h . | awk 'NR==2 {print $4}')
echo "Available disk space: $AVAILABLE"
AVAILABLE_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -gt 5 ]; then
    echo "✓ Sufficient disk space available"
else
    echo "⚠ Low disk space (less than 5GB available)"
fi

# Test 9: Check Memory
echo ""
echo "Test 9: Memory..."
if command -v free &> /dev/null; then
    AVAILABLE_MEM=$(free -g | awk 'NR==2 {print $7}')
    echo "Available memory: ${AVAILABLE_MEM}GB"
    if [ "$AVAILABLE_MEM" -gt 1 ]; then
        echo "✓ Sufficient memory available"
    else
        echo "⚠ Low memory (less than 1GB available)"
    fi
else
    echo "⚠ Cannot check memory (free command not available)"
fi

# Summary
echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✓ All tests passed!"
    echo "Deployment is successful and ready to use."
    echo ""
    echo "Access URLs:"
    echo "  Frontend: $FRONTEND_URL"
    echo "  Backend API: $BACKEND_URL"
    echo "  API Docs: $BACKEND_URL/docs"
else
    echo "✗ $ERRORS test(s) failed"
    echo ""
    echo "Please check the errors above and review logs:"
    echo "  docker-compose logs backend"
    echo "  docker-compose logs frontend"
fi
echo "=========================================="

exit $ERRORS
