from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
from src.plots.plot_constants import PlotConstants
import matplotlib
matplotlib.use("Agg")


def save_plot_as_base64(T, length_x, length_y, step, dt):
    # Generate the plot
    plt.figure()
    plt.imshow(T, extent=[0, length_x, 0, length_y], origin='lower', cmap='hot')
    plt.colorbar(label='Temperature (K)')
    plt.title(f'Temperature Distribution at Time = {step * dt:.2f} s')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    # Save the plot to a BytesIO buffer
    from io import BytesIO
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plt.close()

    # Convert to base64
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return img_base64

def plot_temperature_distribution(length_x: float, length_y: float, time_total: float, dt: float):
    n_steps = int(time_total / dt)
    nx, ny = int(length_x / PlotConstants.dx), int(length_y / PlotConstants.dy)  # Number of grid points

    # Create the temperature field
    T = np.full((nx, ny), PlotConstants.T_initial)

    plots = []  # List to store base64 encoded plots

    # Simulation loop
    for step in range(n_steps):
        T_new = T.copy()

        # Calculate temperature at interior points
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                # Heat conduction (finite difference)
                dTdx2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / PlotConstants.dx**2
                dTdy2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / PlotConstants.dy**2
                T_new[i, j] = T[i, j] + dt * PlotConstants.k / (PlotConstants.rho * PlotConstants.cp) * (dTdx2 + dTdy2)

        # Apply boundary conditions (cooling from mold and ambient)
        T_new[:, 0] = PlotConstants.T_mold  # Left boundary (mold)
        T_new[:, -1] = PlotConstants.T_ambient  # Right boundary (ambient)
        T_new[0, :] = PlotConstants.T_ambient  # Top boundary (ambient)
        T_new[-1, :] = PlotConstants.T_ambient  # Bottom boundary (ambient)

        # Update the temperature field
        T = T_new.copy()

        # Save plots at specific time steps
        if step % 100 == 0 or step == n_steps - 1:
            img_base64 = save_plot_as_base64(T, length_x, length_y, step, dt)
            plots.append(img_base64)

    # Return all plots as a list of base64 strings
    return JSONResponse(content={"plots": plots})
