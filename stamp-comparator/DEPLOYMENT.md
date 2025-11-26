# Deployment Guide

This guide covers deploying the Stamp Comparator application in various environments.

## Table of Contents

- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Environment Variables](#environment-variables)
- [SSL/HTTPS Setup](#sslhttps-setup)
- [Monitoring and Logging](#monitoring-and-logging)
- [Backup and Recovery](#backup-and-recovery)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Security Checklist](#security-checklist)

---

## Local Deployment

### Prerequisites

- Python 3.8+
- Node.js 14+
- Tesseract OCR
- (Optional) CUDA for GPU acceleration

### Steps

1. **Clone and setup:**
   ```bash
   git clone <repository>
   cd stamp-comparator
   ./setup.sh  # or setup.bat on Windows
   ```

2. **Start backend:**
   ```bash
   source venv/bin/activate
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. **Start frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

4. **Access application:**
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

---

## Docker Deployment

### Quick Start

```bash
# Production
./deploy.sh production

# Development
./deploy.sh development
```

### Manual Docker Commands

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart specific service
docker-compose restart backend
```

### Using Pre-trained Models

If you have pre-trained models:

```bash
# Copy models to models directory
cp -r /path/to/trained/models/* ./models/

# Models should be organized as:
# models/
#   siamese/siamese_best.pth
#   cnn_detector/cnn_detector_best.pth
#   autoencoder/autoencoder_best.pth
```

---

## Cloud Deployment

### AWS EC2 Deployment

1. **Launch EC2 instance:**
   - Instance type: `t3.medium` or larger (`t3.large` for ML models)
   - OS: Ubuntu 20.04 LTS
   - Storage: 50GB+ EBS volume
   - Security group: Allow ports 80, 443, 22

2. **Install Docker:**
   ```bash
   sudo apt-get update
   sudo apt-get install docker.io docker-compose
   sudo usermod -aG docker $USER
   ```

3. **Deploy application:**
   ```bash
   git clone <repository>
   cd stamp-comparator
   ./deploy.sh production
   ```

4. **Setup domain (optional):**
   - Point domain to EC2 IP
   - Setup SSL with Let's Encrypt (see SSL section)

### Google Cloud Platform

1. **Create Compute Engine instance:**
   ```bash
   gcloud compute instances create stamp-comparator \
       --machine-type=n1-standard-2 \
       --image-family=ubuntu-2004-lts \
       --image-project=ubuntu-os-cloud \
       --boot-disk-size=50GB
   ```

2. **SSH and deploy:**
   ```bash
   gcloud compute ssh stamp-comparator
   # Follow same steps as AWS
   ```

### Azure

1. **Create VM:**
   ```bash
   az vm create \
       --resource-group stamp-comparator-rg \
       --name stamp-comparator-vm \
       --image UbuntuLTS \
       --size Standard_B2s \
       --admin-username azureuser
   ```

2. Deploy as above

### DigitalOcean

1. **Create Droplet:**
   - Choose Ubuntu 20.04
   - Select appropriate size ($20/month minimum for ML)
   - Add SSH key

2. Deploy as above

---

## Environment Variables

### Backend Environment Variables

Create `backend/.env`:

```bash
# Application
LOG_LEVEL=info
DEBUG=false

# Model paths
SIAMESE_MODEL_PATH=models/siamese/siamese_best.pth
CNN_MODEL_PATH=models/cnn_detector/cnn_detector_best.pth
AUTOENCODER_MODEL_PATH=models/autoencoder/autoencoder_best.pth

# Processing
MAX_IMAGE_SIZE=2000
MAX_UPLOAD_SIZE=50MB

# Performance
USE_GPU=true
NUM_WORKERS=4
```

### Frontend Environment Variables

Create `frontend/.env`:

```bash
VITE_API_URL=http://localhost:8000
VITE_MAX_FILE_SIZE=52428800
```

---

## SSL/HTTPS Setup

### Using Let's Encrypt with Nginx

1. **Install Certbot:**
   ```bash
   sudo apt-get install certbot python3-certbot-nginx
   ```

2. **Update nginx configuration:**
   
   Create `nginx-ssl.conf`:
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;
       return 301 https://$server_name$request_uri;
   }

   server {
       listen 443 ssl http2;
       server_name yourdomain.com;

       ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
       
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers HIGH:!aNULL:!MD5;
       
       # Rest of nginx config...
   }
   ```

3. **Obtain certificate:**
   ```bash
   sudo certbot --nginx -d yourdomain.com
   ```

4. **Auto-renewal:**
   ```bash
   sudo certbot renew --dry-run
   ```

---

## Monitoring and Logging

### Docker Logs

```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Application Logs

Logs are stored in `logs/` directory:
- `backend.log` - Backend application logs
- `access.log` - API access logs
- `error.log` - Error logs

### Monitoring with Prometheus (Optional)

Add to `docker-compose.yml`:

```yaml
prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    depends_on:
      - prometheus
```

---

## Backup and Recovery

### Backup Strategy

**What to backup:**
- Trained models (`models/`)
- User data (`data/`)
- Configuration files
- Database (if using one)

**Backup script:**

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/stamp-comparator"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="backup_$DATE.tar.gz"

# Create backup
tar -czf "$BACKUP_DIR/$BACKUP_NAME" \
    models/ \
    data/ \
    backend/.env \
    frontend/.env

# Keep only last 7 backups
cd "$BACKUP_DIR"
ls -t | tail -n +8 | xargs rm -f

echo "Backup created: $BACKUP_NAME"
```

**Automate with cron:**

```bash
# Run daily at 2 AM
0 2 * * * /path/to/backup.sh
```

### Recovery

```bash
# Stop application
docker-compose down

# Extract backup
tar -xzf backup_YYYYMMDD_HHMMSS.tar.gz

# Restart application
docker-compose up -d
```

---

## Performance Tuning

### Backend Optimization

1. **Use GPU for ML models:**
   - Ensure CUDA is available
   - Set `USE_GPU=true` in environment

2. **Adjust worker processes:**
   ```bash
   uvicorn main:app --workers 4 --host 0.0.0.0
   ```

3. **Enable caching (optional):**
   - Use Redis for result caching
   - Add to docker-compose.yml

### Frontend Optimization

- Enable compression in nginx (already in nginx.conf)
- CDN for static assets (optional)
- Lazy loading for images

---

## Troubleshooting

### Common Issues

**Backend won't start:**
```bash
# Check logs
docker-compose logs backend

# Check if port is in use
sudo lsof -i :8000

# Restart service
docker-compose restart backend
```

**Frontend can't reach backend:**
- Check CORS settings in `main.py`
- Verify API_URL in frontend `.env`
- Check network in docker-compose

**Models not loading:**
- Verify model files exist in `models/`
- Check file permissions
- Review backend logs for specific errors

**Out of memory:**
- Reduce batch sizes in config
- Disable some ML models
- Increase Docker memory limit
- Use smaller image sizes

---

## Security Checklist

- [ ] Change default passwords
- [ ] Enable HTTPS/SSL
- [ ] Configure firewall rules
- [ ] Regular security updates
- [ ] Implement rate limiting
- [ ] Use environment variables for secrets
- [ ] Regular backups
- [ ] Monitor logs for suspicious activity

---

## Scaling

For high-traffic deployments:

1. **Load balancing:**
   - Use nginx or HAProxy
   - Deploy multiple backend instances

2. **Separate services:**
   - Move ML models to dedicated servers
   - Use message queue (RabbitMQ/Redis) for async processing

3. **Database:**
   - Add PostgreSQL for storing results
   - Use Redis for caching

---

## Support

For deployment issues:
- Check logs: `docker-compose logs -f`
- Review troubleshooting section
- Open GitHub issue with logs and configuration
