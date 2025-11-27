#!/bin/bash
# Sync updated code to worker machine

WORKER_USER="suraj"
WORKER_HOST="<WORKER_IP>"  # Replace with actual worker IP
WORKER_PATH="~/hetero-cluster"

echo "=========================================="
echo "Syncing Code to Worker"
echo "=========================================="

# Option 1: Sync just the distributed_trainer.py file
echo "Syncing distributed_trainer.py..."
scp "src/training/distributed_trainer.py" \
    "${WORKER_USER}@${WORKER_HOST}:${WORKER_PATH}/src/training/"

# Option 2: Sync entire src directory (recommended)
# Uncomment to use:
# echo "Syncing entire src directory..."
# rsync -avz --exclude '__pycache__' --exclude '*.pyc' \
#     src/ "${WORKER_USER}@${WORKER_HOST}:${WORKER_PATH}/src/"

echo "✓ Sync complete!"
echo ""
echo "On worker, verify the file:"
echo "  ssh ${WORKER_USER}@${WORKER_HOST}"
echo "  grep -n 'if torch.cuda.is_available()' ${WORKER_PATH}/src/training/distributed_trainer.py"
echo ""
echo "Expected output: Line 76 with the if statement"
