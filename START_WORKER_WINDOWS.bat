@echo off
REM Worker start script for native Windows (no WSL2)
REM Run this on Windows with Python installed

echo ==========================================
echo Starting Worker Node (Rank 1) - Windows
echo ==========================================

REM Activate virtual environment (adjust path if needed)
call venv\Scripts\activate.bat

REM Distributed training config
set RANK=1
set WORLD_SIZE=2
set MASTER_ADDR=10.161.199.69
set MASTER_PORT=29500
set LOCAL_RANK=0

REM Network interface for Gloo (Windows will auto-detect)
set GLOO_DEVICE_TRANSPORT=TCP

echo Connecting to master at: %MASTER_ADDR%:%MASTER_PORT%
echo ==========================================

REM Start training with lightweight synthetic dataset for demo
python -m src.training.main ^
  --model simple_cnn ^
  --dataset synthetic_image ^
  --num-samples 1000 ^
  --image-size 32 ^
  --batch-size 32 ^
  --epochs 5 ^
  --lr 0.01 ^
  --backend gloo ^
  --master-addr %MASTER_ADDR% ^
  --enable-profiling ^
  --enable-load-balancing ^
  --load-balance-policy dynamic ^
  --experiment-name demo_gloo_heterogeneous

pause
