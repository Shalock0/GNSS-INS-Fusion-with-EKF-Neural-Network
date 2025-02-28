import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_trajectory(gt, pred, title="Trajectory Comparison"):
    """使用Matplotlib绘制2D轨迹"""
    plt.figure(figsize=(10, 6))
    plt.plot(gt[:,0], gt[:,1], label='Ground Truth')
    plt.plot(pred[:,0], pred[:,1], label='Predicted', linestyle='--')
    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(f"outputs/results/{title}.png")
    plt.close()

def plot_3d_trajectory(gt, pred):
    """使用Plotly绘制3D轨迹"""
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=gt[:,0], y=gt[:,1], z=gt[:,2],
        mode='lines', name='Ground Truth'
    ))
    fig.add_trace(go.Scatter3d(
        x=pred[:,0], y=pred[:,1], z=pred[:,2],
        mode='lines', name='Predicted'
    ))
    fig.write_html("outputs/results/3d_trajectory.html")