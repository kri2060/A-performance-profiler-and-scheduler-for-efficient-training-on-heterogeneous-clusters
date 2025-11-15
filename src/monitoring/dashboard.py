"""
Streamlit Monitoring Dashboard
Real-time visualization of training metrics
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Hetero Cluster Monitor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_metrics(metrics_dir: str = "experiments/logs"):
    """Load metrics from JSON files"""
    metrics_path = Path(metrics_dir)

    if not metrics_path.exists():
        return {}

    all_metrics = {}

    # Load metrics for each rank
    for metrics_file in metrics_path.glob("rank_*_metrics.json"):
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                rank = int(metrics_file.stem.split('_')[1])
                all_metrics[rank] = data
        except Exception as e:
            st.error(f"Error loading {metrics_file}: {e}")

    return all_metrics


def load_gpu_profiles(config_dir: str = "experiments/configs"):
    """Load GPU profiles"""
    profile_path = Path(config_dir) / "gpu_profiles.json"

    if not profile_path.exists():
        return []

    try:
        with open(profile_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading GPU profiles: {e}")
        return []


def plot_gpu_utilization(metrics_dict):
    """Plot GPU utilization over time"""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("GPU Utilization",)
    )

    for rank, metrics in metrics_dict.items():
        if not metrics:
            continue

        df = pd.DataFrame(metrics)

        fig.add_trace(
            go.Scatter(
                x=df['iteration'],
                y=df['gpu_utilization'],
                mode='lines',
                name=f'Rank {rank}',
                line=dict(width=2)
            )
        )

    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="GPU Utilization (%)", range=[0, 100])
    fig.update_layout(height=400, hovermode='x unified')

    return fig


def plot_memory_usage(metrics_dict):
    """Plot GPU memory usage"""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("GPU Memory Usage",)
    )

    for rank, metrics in metrics_dict.items():
        if not metrics:
            continue

        df = pd.DataFrame(metrics)

        fig.add_trace(
            go.Scatter(
                x=df['iteration'],
                y=df['gpu_memory_percent'],
                mode='lines',
                name=f'Rank {rank}',
                line=dict(width=2)
            )
        )

    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Memory Usage (%)", range=[0, 100])
    fig.update_layout(height=400, hovermode='x unified')

    return fig


def plot_throughput(metrics_dict):
    """Plot training throughput"""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("Training Throughput",)
    )

    for rank, metrics in metrics_dict.items():
        if not metrics:
            continue

        df = pd.DataFrame(metrics)

        fig.add_trace(
            go.Scatter(
                x=df['iteration'],
                y=df['throughput'],
                mode='lines',
                name=f'Rank {rank}',
                line=dict(width=2)
            )
        )

    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Throughput (samples/s)")
    fig.update_layout(height=400, hovermode='x unified')

    return fig


def plot_loss(metrics_dict):
    """Plot training loss"""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("Training Loss",)
    )

    for rank, metrics in metrics_dict.items():
        if not metrics:
            continue

        df = pd.DataFrame(metrics)

        fig.add_trace(
            go.Scatter(
                x=df['iteration'],
                y=df['loss'],
                mode='lines',
                name=f'Rank {rank}',
                line=dict(width=2)
            )
        )

    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Loss")
    fig.update_layout(height=400, hovermode='x unified')

    return fig


def plot_iteration_time_breakdown(metrics_dict):
    """Plot iteration time breakdown"""
    # Calculate averages for latest N iterations
    n = 50
    breakdown_data = []

    for rank, metrics in metrics_dict.items():
        if not metrics:
            continue

        df = pd.DataFrame(metrics)
        if len(df) == 0:
            continue

        recent = df.tail(n)

        breakdown_data.append({
            'Rank': rank,
            'Data Loading': recent['data_loading_time'].mean(),
            'Forward': recent['forward_time'].mean(),
            'Backward': recent['backward_time'].mean(),
            'Optimizer': recent['optimizer_time'].mean(),
        })

    if not breakdown_data:
        return go.Figure()

    df_breakdown = pd.DataFrame(breakdown_data)

    fig = go.Figure()

    for col in ['Data Loading', 'Forward', 'Backward', 'Optimizer']:
        fig.add_trace(go.Bar(
            name=col,
            x=df_breakdown['Rank'],
            y=df_breakdown[col],
        ))

    fig.update_layout(
        barmode='stack',
        title="Iteration Time Breakdown (avg last 50 iters)",
        xaxis_title="Rank",
        yaxis_title="Time (s)",
        height=400
    )

    return fig


def plot_gpu_comparison(gpu_profiles):
    """Plot GPU comparison"""
    if not gpu_profiles:
        return go.Figure()

    df = pd.DataFrame(gpu_profiles)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Compute Score", "Memory (MB)")
    )

    # Compute scores
    fig.add_trace(
        go.Bar(
            x=df['device_id'],
            y=df['compute_score'],
            name='Compute Score',
            marker_color='lightblue'
        ),
        row=1, col=1
    )

    # Memory
    fig.add_trace(
        go.Bar(
            x=df['device_id'],
            y=df['total_memory_mb'],
            name='Memory (MB)',
            marker_color='lightgreen'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text="GPU ID", row=1, col=1)
    fig.update_xaxes(title_text="GPU ID", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)
    fig.update_layout(height=400, showlegend=False)

    return fig


def display_current_stats(metrics_dict):
    """Display current statistics"""
    cols = st.columns(len(metrics_dict) if metrics_dict else 1)

    for idx, (rank, metrics) in enumerate(metrics_dict.items()):
        if not metrics:
            continue

        with cols[idx]:
            latest = metrics[-1] if metrics else {}

            st.metric(
                label=f"Rank {rank} - GPU Util",
                value=f"{latest.get('gpu_utilization', 0):.1f}%"
            )
            st.metric(
                label=f"Rank {rank} - Memory",
                value=f"{latest.get('gpu_memory_percent', 0):.1f}%"
            )
            st.metric(
                label=f"Rank {rank} - Throughput",
                value=f"{latest.get('throughput', 0):.1f} samples/s"
            )


def main():
    """Main dashboard function"""

    st.title("🚀 Heterogeneous Cluster Training Monitor")
    st.markdown("Real-time monitoring of distributed training on heterogeneous GPUs")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Experiment selection
        experiments_base = Path("experiments")
        available_experiments = []
        if experiments_base.exists():
            available_experiments = [d.name for d in experiments_base.iterdir()
                                   if d.is_dir() and (d / "logs").exists()]

        if available_experiments:
            experiment_name = st.selectbox(
                "Select Experiment",
                options=available_experiments,
                index=0
            )
            metrics_dir = f"experiments/{experiment_name}/logs"
            config_dir = f"experiments/{experiment_name}/configs"
        else:
            st.warning("No experiments found. Using manual paths.")
            metrics_dir = st.text_input(
                "Metrics Directory",
                value="experiments/logs"
            )
            config_dir = st.text_input(
                "Config Directory",
                value="experiments/configs"
            )

        refresh_interval = st.slider(
            "Refresh Interval (s)",
            min_value=1,
            max_value=30,
            value=5
        )

        auto_refresh = st.checkbox("Auto Refresh", value=True)

        if st.button("Refresh Now"):
            st.rerun()

    # Load data
    metrics_dict = load_metrics(metrics_dir)
    gpu_profiles = load_gpu_profiles(config_dir)

    if not metrics_dict:
        st.warning("No metrics found. Start training to see data.")
        st.info(f"Looking for metrics in: {metrics_dir}")
        return

    # Display current stats
    st.subheader("Current Statistics")
    display_current_stats(metrics_dict)

    st.markdown("---")

    # GPU Comparison
    if gpu_profiles:
        st.subheader("GPU Hardware Comparison")
        fig = plot_gpu_comparison(gpu_profiles)
        st.plotly_chart(fig, use_container_width=True)

        # Display GPU details
        with st.expander("GPU Details"):
            for gpu in gpu_profiles:
                st.write(f"**GPU {gpu['device_id']}: {gpu['name']}**")
                st.write(f"  - Compute Score: {gpu['compute_score']}")
                st.write(f"  - Memory: {gpu['total_memory_mb']:.0f} MB")
                st.write(f"  - Compute Capability: {gpu['compute_capability']}")

    st.markdown("---")

    # Training metrics
    st.subheader("Training Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(plot_loss(metrics_dict), use_container_width=True)

    with col2:
        st.plotly_chart(plot_throughput(metrics_dict), use_container_width=True)

    st.markdown("---")

    # GPU metrics
    st.subheader("GPU Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(plot_gpu_utilization(metrics_dict), use_container_width=True)

    with col2:
        st.plotly_chart(plot_memory_usage(metrics_dict), use_container_width=True)

    st.markdown("---")

    # Performance analysis
    st.subheader("Performance Analysis")
    st.plotly_chart(plot_iteration_time_breakdown(metrics_dict), use_container_width=True)

    # Bottleneck detection
    with st.expander("Bottleneck Analysis"):
        for rank, metrics in metrics_dict.items():
            if not metrics:
                continue

            df = pd.DataFrame(metrics)
            recent = df.tail(20)

            avg_times = {
                'Data Loading': recent['data_loading_time'].mean(),
                'Forward Pass': recent['forward_time'].mean(),
                'Backward Pass': recent['backward_time'].mean(),
                'Optimizer': recent['optimizer_time'].mean(),
            }

            bottleneck = max(avg_times.items(), key=lambda x: x[1])

            st.write(f"**Rank {rank} Bottleneck:** {bottleneck[0]} ({bottleneck[1]:.3f}s)")

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
