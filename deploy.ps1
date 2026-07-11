# Docker deployment script for PDF RAG service (PowerShell)
# Make sure you're logged in to Docker Hub: docker login

# Set your Docker Hub username
$DOCKER_USERNAME = "vikrant48"
$IMAGE_NAME = "pdf-rag-service-backend"
$TAG = "v1.0.0"

$FULL_IMAGE_NAME = "${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

# Build the Docker image directly with the target tag
Write-Host "Building Docker image: $FULL_IMAGE_NAME..." -ForegroundColor Green
docker build -t $FULL_IMAGE_NAME .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!" -ForegroundColor Red
    exit 1
}

# Push to Docker Hub
Write-Host "Pushing to Docker Hub..." -ForegroundColor Green
docker push $FULL_IMAGE_NAME

if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment complete!" -ForegroundColor Green
    Write-Host "Your image is available at: $FULL_IMAGE_NAME" -ForegroundColor Cyan
    
    # Cleanup dangling images (prevents <none>:<none> buildup)
    Write-Host "Cleaning up dangling images..." -ForegroundColor Yellow
    docker image prune -f
} else {
    Write-Host "Docker push failed!" -ForegroundColor Red
    exit 1
}
