import streamlit as st
import numpy as np
import pandas as pd
import qutip as qt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="SpinDynamics Quantum Simulation", layout="wide")

st.title("SpinDynamics: Quantum Spin Simulation ⚛️")
st.markdown("""
This app simulates the time evolution of a quantum spin system under Zeeman and Exchange interactions.
Based on the [SpinDynamics-Quantum-Simulation](https://github.com/r-karra/SpinDynamics-Quantum-Simulation) project.
""")

# --- Sidebar Configuration ---
st.sidebar.header("Simulation Settings")

# 1. System Parameters
st.sidebar.subheader("System")
num_spins = st.sidebar.number_input("Number of Spins", min_value=1, max_value=2, value=1)
gamma = st.sidebar.number_input("Gyromagnetic Ratio (γ)", value=1.0, format="%.4f")

# 2. Magnetic Field (Zeeman)
st.sidebar.subheader("External Magnetic Field (B)")
bx = st.sidebar.slider("B_x", -5.0, 5.0, 0.0)
by = st.sidebar.slider("B_y", -5.0, 5.0, 0.0)
bz = st.sidebar.slider("B_z", -5.0, 5.0, 1.0)

# 3. Exchange Coupling (if num_spins > 1)
j_coupling = 0.0
if num_spins > 1:
    st.sidebar.subheader("Exchange Interaction (J)")
    j_coupling = st.sidebar.slider("Coupling Strength J", -10.0, 10.0, 1.0)

# 4. Time Parameters
st.sidebar.subheader("Time Evolution")
t_max = st.sidebar.slider("Max Time", 0.1, 50.0, 10.0)
n_points = st.sidebar.slider("Number of Points", 100, 1000, 500)

# --- Simulation Logic ---

def run_simulation():
    # Pauli Operators
    sx = qt.jmat(0.5, 'x')
    sy = qt.jmat(0.5, 'y')
    sz = qt.jmat(0.5, 'z')
    
    # Time array
    tlist = np.linspace(0, t_max, n_points)
    
    if num_spins == 1:
        # Hamiltonian: H = -gamma * (Bx*Sx + By*Sy + Bz*Sz)
        H = -gamma * (bx * sx + by * sy + bz * sz)
        # Initial state: Spin up
        psi0 = qt.basis(2, 0)
        # Observables
        obs = [sx, sy, sz]
        result = qt.mesolve(H, psi0, tlist, c_ops=[], e_ops=obs)
        return tlist, result.expect, ["Sx", "Sy", "Sz"]
    
    else:
        # Two-spin system
        # Hamiltonian H = H1 + H2 + Hex
        # H1 = -gamma * (B . S1), H2 = -gamma * (B . S2)
        # Hex = J * (S1 . S2)
        
        # Identity for tensor product
        id2 = qt.qeye(2)
        
        # Operators for spin 1 and 2
        sx1 = qt.tensor(sx, id2)
        sy1 = qt.tensor(sy, id2)
        sz1 = qt.tensor(sz, id2)
        
        sx2 = qt.tensor(id2, sx)
        sy2 = qt.tensor(id2, sy)
        sz2 = qt.tensor(id2, sz)
        
        # Zeeman terms
        Hz1 = -gamma * (bx * sx1 + by * sy1 + bz * sz1)
        Hz2 = -gamma * (bx * sx2 + by * sy2 + bz * sz2)
        
        # Exchange term: J * (Sx1*Sx2 + Sy1*Sy2 + Sz1*Sz2)
        Hex = j_coupling * (sx1*sx2 + sy1*sy2 + sz1*sz2)
        
        H = Hz1 + Hz2 + Hex
        
        # Initial state: Spin 1 up, Spin 2 down
        psi0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
        
        # Observables (Expectation values for spin 1 and 2)
        obs = [sx1, sy1, sz1, sx2, sy2, sz2]
        result = qt.mesolve(H, psi0, tlist, c_ops=[], e_ops=obs)
        return tlist, result.expect, ["S1x", "S1y", "S1z", "S2x", "S2y", "S2z"]

# --- UI and Visualization ---

tlist, expectations, labels = run_simulation()

# Create DataFrame for plots
df = pd.DataFrame(np.array(expectations).T, columns=labels)
df['Time'] = tlist

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Time Evolution of Spin Components")
    fig_lines = go.Figure()
    for label in labels:
        fig_lines.add_trace(go.Scatter(x=df['Time'], y=df[label], name=label))
    
    fig_lines.update_layout(
        xaxis_title="Time",
        yaxis_title="Expectation Value",
        hovermode="x unified",
        template="plotly_dark"
    )
    st.plotly_chart(fig_lines, use_container_width=True)

with col_right:
    st.subheader("Bloch Sphere Projection")
    # Show Spin 1 on Bloch sphere
    step = st.slider("Time Step index", 0, n_points-1, n_points-1)
    
    current_sx = expectations[0][step]
    current_sy = expectations[1][step]
    current_sz = expectations[2][step]
    
    fig_bloch = go.Figure(data=[
        # The Sphere
        go.Mesh3d(
            x=np.outer(np.cos(np.linspace(0, 2*np.pi, 30)), np.sin(np.linspace(0, np.pi, 30))).flatten(),
            y=np.outer(np.sin(np.linspace(0, 2*np.pi, 30)), np.sin(np.linspace(0, np.pi, 30))).flatten(),
            z=np.outer(np.ones(30), np.cos(np.linspace(0, np.pi, 30))).flatten(),
            opacity=0.1,
            color='cyan'
        ),
        # The Spin Vector
        go.Scatter3d(
            x=[0, current_sx*2], y=[0, current_sy*2], z=[0, current_sz*2],
            mode='lines+markers',
            line=dict(color='red', width=10),
            marker=dict(size=5, color='red'),
            name="Spin Vector"
        )
    ])
    
    # Axes
    fig_bloch.add_trace(go.Scatter3d(x=[-1, 1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='white', width=2), showlegend=False))
    fig_bloch.add_trace(go.Scatter3d(x=[0, 0], y=[-1, 1], z=[0, 0], mode='lines', line=dict(color='white', width=2), showlegend=False))
    fig_bloch.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1, 1], mode='lines', line=dict(color='white', width=2), showlegend=False))

    fig_bloch.update_layout(
        scene=dict(
            xaxis=dict(title='Sx', range=[-1.1, 1.1]),
            yaxis=dict(title='Sy', range=[-1.1, 1.1]),
            zaxis=dict(title='Sz', range=[-1.1, 1.1]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        template="plotly_dark"
    )
    st.plotly_chart(fig_bloch, use_container_width=True)
    st.info(f"Showing state at t = {tlist[step]:.2f}")

st.divider()
st.markdown("### Physics Background")
st.latex(r"H = -\gamma \sum_i \mathbf{B} \cdot \mathbf{S}_i + J \sum_{\langle i,j \rangle} \mathbf{S}_i \cdot \mathbf{S}_j")
st.write("""
The simulation uses QuTiP to solve the Time-Dependent Schrödinger Equation. 
For 1 spin, it demonstrates Larmor precession. 
For 2 spins, the exchange interaction J introduces entanglement and complex dynamics between the spins.
""")
