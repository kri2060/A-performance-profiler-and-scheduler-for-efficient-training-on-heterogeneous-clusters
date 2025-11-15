@echo off
REM Docker Multi-Node Training - ADAPTIVE WORKER Node Script (Windows)
REM Automatically detects GPU availability and uses GPU or CPU accordingly

echo ==========================================
echo Starting ADAPTIVE WORKER Node
echo ==========================================

REM Configuration - MUST BE SET BY USER
if "%MASTER_ADDR%"=="" (
    echo ERROR: MASTER_ADDR not set!
    echo Please set MASTER_ADDR to the master node's IP address:
    echo   set MASTER_ADDR=192.168.1.100
    exit /b 1
)

if "%RANK%"=="" (
    echo ERROR: RANK not set!
    echo Please set RANK to this worker's rank (1, 2, 3, ...):
    echo   set RANK=1
    exit /b 1
)

if "%MASTER_PORT%"=="" set MASTER_PORT=29500
if "%WORLD_SIZE%"=="" set WORLD_SIZE=4
if "%EXPERIMENT_NAME%"=="" set EXPERIMENT_NAME=distributed_training
if "%IMAGE_NAME%"=="" set IMAGE_NAME=hetero-cluster-training

REM Detect GPU availability
set GPU_AVAILABLE=false
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    set GPU_AVAILABLE=true
    echo [32m✓ GPU detected - will use GPU acceleration[0m
) else (
    echo [33m✗ No GPU detected - will use CPU[0m
)

echo Master Address: %MASTER_ADDR%
echo Master Port: %MASTER_PORT%
echo Worker Rank: %RANK%
echo World Size: %WORLD_SIZE%
if "%GPU_AVAILABLE%"=="true" (
    echo Device: GPU
) else (
    echo Device: CPU
)
echo ==========================================

REM Build Docker image if it doesn't exist
docker image inspect %IMAGE_NAME% >nul 2>&1
if errorlevel 1 (
    echo Building Docker image...
    docker build -t %IMAGE_NAME% .
)

REM Create shared volume for experiments
if not exist experiments\logs mkdir experiments\logs
if not exist experiments\configs mkdir experiments\configs

REM Run worker container based on GPU availability
if "%GPU_AVAILABLE%"=="true" (
    echo Starting worker in GPU mode...
    docker run --rm -it ^
        --gpus all ^
        --network host ^
        --name hetero-worker-%RANK% ^
        -v "%cd%:/workspace" ^
        -e MASTER_ADDR=%MASTER_ADDR% ^
        -e MASTER_PORT=%MASTER_PORT% ^
        -e RANK=%RANK% ^
        -e WORLD_SIZE=%WORLD_SIZE% ^
        -e EXPERIMENT_NAME=%EXPERIMENT_NAME% ^
        -e CUDA_VISIBLE_DEVICES=0 ^
        %IMAGE_NAME% ^
        bash -c "echo 'Worker node %RANK% ready (GPU). Connecting to master at %MASTER_ADDR%:%MASTER_PORT%...' && python -m src.training.main --model simple_cnn --dataset synthetic_image --batch-size 64 --epochs 10 --enable-profiling --enable-load-balancing --load-balance-policy dynamic --backend gloo --master-addr $MASTER_ADDR --master-port $MASTER_PORT --experiment-name $EXPERIMENT_NAME"
) else (
    echo Starting worker in CPU mode...
    docker run --rm -it ^
        --network host ^
        --name hetero-worker-%RANK% ^
        -v "%cd%:/workspace" ^
        -e MASTER_ADDR=%MASTER_ADDR% ^
        -e MASTER_PORT=%MASTER_PORT% ^
        -e RANK=%RANK% ^
        -e WORLD_SIZE=%WORLD_SIZE% ^
        -e EXPERIMENT_NAME=%EXPERIMENT_NAME% ^
        -e CUDA_VISIBLE_DEVICES= ^
        %IMAGE_NAME% ^
        bash -c "echo 'Worker node %RANK% ready (CPU). Connecting to master at %MASTER_ADDR%:%MASTER_PORT%...' && python -m src.training.main --model simple_cnn --dataset synthetic_image --batch-size 32 --epochs 10 --enable-profiling --enable-load-balancing --load-balance-policy dynamic --backend gloo --master-addr $MASTER_ADDR --master-port $MASTER_PORT --experiment-name $EXPERIMENT_NAME"
)

echo Worker node %RANK% completed!
