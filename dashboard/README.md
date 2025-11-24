# Professional Dashboard for Heterogeneous Cluster Trainer

A modern, professional dashboard built with **React + TypeScript + Next.js** for the frontend and **FastAPI** for the backend.

## Tech Stack

### Frontend
- **Framework**: React 18 with Next.js 14
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Components**: Custom components inspired by shadcn/ui
- **Data Fetching**: TanStack Query (React Query)
- **Charts**: Recharts
- **Real-time**: WebSocket support

### Backend
- **Framework**: FastAPI
- **WebSocket**: Real-time metrics streaming
- **API**: RESTful endpoints for data retrieval

## Features

- **Cluster Overview**: Total GPUs, active nodes, utilization metrics
- **GPU Hardware Profiles**: Detailed hardware specs for each GPU
- **Training Metrics**: Loss curves, throughput graphs
- **GPU Monitoring**: Utilization and memory usage charts
- **Time Breakdown**: Analyze where time is spent (data loading, forward, backward, optimizer)
- **Experiment Tracking**: List and compare experiments
- **Real-time Updates**: WebSocket support for live metrics

## Quick Start

### 1. Start Backend (FastAPI)

```bash
cd dashboard/backend

# Install dependencies
pip install -r requirements.txt

# Start server
python main.py
```

Backend runs at: http://localhost:8000

API docs at: http://localhost:8000/docs

### 2. Start Frontend (Next.js)

```bash
cd dashboard/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend runs at: http://localhost:3000

## API Endpoints

### REST Endpoints

- `GET /api/health` - Health check
- `GET /api/nodes` - Get GPU profiles
- `GET /api/system` - Get system information
- `GET /api/cluster/status` - Get cluster overview stats
- `GET /api/experiments` - List all experiments
- `GET /api/jobs` - Get training jobs
- `GET /api/jobs/{job_id}/metrics` - Get metrics for specific job
- `GET /api/metrics/latest` - Get latest metrics
- `GET /api/metrics/history/{experiment}` - Get historical metrics

### WebSocket Endpoints

- `ws://localhost:8000/ws/metrics` - Real-time metrics stream
- `ws://localhost:8000/ws/logs` - Real-time log stream

## Project Structure

```
dashboard/
├── backend/
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
│
└── frontend/
    ├── src/
    │   ├── app/
    │   │   ├── layout.tsx   # Root layout
    │   │   ├── page.tsx     # Main page
    │   │   ├── providers.tsx # React Query provider
    │   │   └── globals.css  # Global styles
    │   │
    │   ├── components/
    │   │   ├── Dashboard.tsx       # Main dashboard
    │   │   ├── ui/
    │   │   │   └── card.tsx        # Card component
    │   │   └── charts/
    │   │       ├── LossChart.tsx
    │   │       ├── ThroughputChart.tsx
    │   │       ├── GPUChart.tsx
    │   │       └── TimeBreakdownChart.tsx
    │   │
    │   ├── hooks/
    │   │   └── useWebSocket.ts     # WebSocket hook
    │   │
    │   ├── lib/
    │   │   └── utils.ts            # Utility functions
    │   │
    │   └── types/
    │       └── index.ts            # TypeScript types
    │
    ├── package.json
    ├── tsconfig.json
    ├── tailwind.config.ts
    ├── postcss.config.js
    └── next.config.js
```

## Screenshots

The dashboard includes:

1. **Stats Overview Cards**: Total GPUs, utilization, memory, throughput
2. **GPU Hardware Cards**: Detailed specs for each GPU
3. **Training Loss Chart**: Line chart showing loss over iterations
4. **Throughput Chart**: Area chart showing samples/sec
5. **GPU Utilization Chart**: Line chart with utilization and memory %
6. **Time Breakdown Chart**: Stacked bar chart of training phases
7. **Experiments Table**: List of all experiments

## Customization

### Adding New Charts

1. Create component in `src/components/charts/`
2. Import and use Recharts components
3. Add to Dashboard.tsx

### Changing Theme

Edit `tailwind.config.ts` and `src/app/globals.css`

### Adding API Endpoints

1. Add route in `backend/main.py`
2. Add TypeScript type in `frontend/src/types/index.ts`
3. Add query in component using `useQuery`

## Development

### Frontend

```bash
# Development
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Lint
npm run lint
```

### Backend

```bash
# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Production Deployment

### Backend

```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Frontend

```bash
# Build
npm run build

# Deploy to Vercel (easiest)
npx vercel

# Or run locally
npm start
```

## Comparison: Streamlit vs React Dashboard

| Feature | Streamlit | React + FastAPI |
|---------|-----------|-----------------|
| Setup Time | 5 minutes | 30 minutes |
| Performance | Good | Excellent |
| Customization | Limited | Full control |
| Professional Look | Basic | Professional |
| Real-time Updates | Auto-refresh | True WebSocket |
| Scalability | Limited | Highly scalable |
| Team Development | Difficult | Easy (typed, tested) |

## Next Steps

1. Add user authentication (JWT)
2. Add experiment comparison view
3. Add model deployment interface
4. Add alerting system
5. Add data export functionality
6. Add dark mode toggle
7. Deploy to cloud (AWS, GCP, etc.)

## License

MIT
