import io
from fastapi.responses import StreamingResponse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
# matplotlib.use("Agg")
from scipy.stats import linregress
from src.plots.plot_constants import PlotConstants
# from plot_constants import PlotConstants

def calculate_bar_temperature(pouring_temp, RPM):
    # Calculating bar temperature at wheel exit and roll entry using the formulas from the paper
    bar_temp_wheel_exit = 530 + 73.29 * (RPM**1.96) + 0.431 * (pouring_temp - 705)
    bar_temp_roll_entry = 503 + 78 * (RPM - 1.96) + 0.5 * (pouring_temp - 705)
    return bar_temp_wheel_exit, bar_temp_roll_entry



# multi-plot
def plot_temperature_distribution(length_x, length_y, time_total, dt):
    n_steps = int(time_total / dt)
    nx, ny = int(length_x / PlotConstants.dx), int(length_y / PlotConstants.dy)  # Number of grid points

    # Create the temperature field
    T = np.full((nx, ny), PlotConstants.T_initial)

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

        # Visualization at specific time steps
        if step % 100 == 0 or step == n_steps - 1:
            plt.imshow(T, extent=[0, length_x, 0, length_y], origin='lower', cmap='hot')
            plt.colorbar(label='Temperature (K)')
            plt.title(f'Temperature Distribution at Time = {step * dt:.2f} s')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.pause(0.1)

    # Final plot
    plt.show()


# single plot
def plot_core_surface_temp(length_x, length_y, time_total, dt):
    n_steps = int(time_total / dt)
    nx, ny = int(length_x / PlotConstants.dx), int(length_y / PlotConstants.dy)  # Number of grid points

    print('we got call to plot_core_surface_temp')
    core_temps = []
    surface_temps = []

     # Create the temperature field
    T = np.full((nx, ny), PlotConstants.T_initial)

    # Simulation loop with data collection
    for step in range(n_steps):
        T_new = T.copy()

        # Calculate temperature at interior points
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dTdx2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / PlotConstants.dx**2
                dTdy2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / PlotConstants.dy**2
                T_new[i, j] = T[i, j] + dt * PlotConstants.k / (PlotConstants.rho * PlotConstants.cp) * (dTdx2 + dTdy2)

        # Apply boundary conditions
        T_new[:, 0] = PlotConstants.T_mold
        T_new[:, -1] = PlotConstants.T_ambient
        T_new[0, :] = PlotConstants.T_ambient
        T_new[-1, :] = PlotConstants.T_ambient

        # Update the temperature field
        T = T_new.copy()

        # Collect core and surface temperatures
        core_temps.append(T[nx // 2, ny // 2])
        surface_temps.append(np.mean(T[:, 0]))

    # Plot core vs surface temperature
    time_steps = np.arange(0, time_total, dt)
    plt.plot(time_steps, core_temps, label="Core Temperature")
    plt.plot(time_steps, surface_temps, label="Surface Temperature")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Core vs. Surface Temperature over Time")
    plt.legend()
    plt.grid()
    # plt.show()

    # # Save the plot to a byte stream
    # buf = io.BytesIO()
    # plt.savefig(buf, format="png")
    # buf.seek(0)
    # plt.close()  # Free up memory

    # # Return the plot as a streaming response
    # return StreamingResponse(buf, media_type="image/png")

    return plt


def plot_temperature_distribution_bar_width(length_x, length_y, time_total, dt):
    n_steps = int(time_total / dt)
    nx, ny = int(length_x / PlotConstants.dx), int(length_y / PlotConstants.dy)  # Number of grid points

    # Create the temperature field
    T = np.full((nx, ny), PlotConstants.T_initial)

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


    # Collect temperature profiles along a horizontal slice (midpoint)
    cross_section_temps = []

    for step in range(n_steps):
        T_new = T.copy()

        # Update temperature
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dTdx2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / PlotConstants.dx**2
                dTdy2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / PlotConstants.dy**2
                T_new[i, j] = T[i, j] + dt * PlotConstants.k / (PlotConstants.rho * PlotConstants.cp) * (dTdx2 + dTdy2)

        # Boundary conditions
        T_new[:, 0] = PlotConstants.T_mold
        T_new[:, -1] = PlotConstants.T_ambient
        T_new[0, :] = PlotConstants.T_ambient
        T_new[-1, :] = PlotConstants.T_ambient

        T = T_new.copy()

        # Record horizontal temperature profile at the midpoint
        if step % 500 == 0 or step == n_steps - 1:
            cross_section_temps.append((step * dt, T[:, ny // 2]))

    # Plot cross-sectional temperature distribution at different times
    for time, temp_profile in cross_section_temps:
        plt.plot(np.linspace(0, length_x, nx), temp_profile, label=f"t = {time:.2f}s")

    plt.xlabel("Position Along Width (m)")
    plt.ylabel("Temperature (K)")
    plt.title("Temperature Distribution Along Bar Width")
    plt.legend()
    plt.grid()
    # plt.show()

    # Save the plot to a byte stream
    # buf = io.BytesIO()
    # plt.savefig(buf, format="png")
    # buf.seek(0)
    # plt.close()  # Free up memory

    # # Return the plot as a streaming response
    # return StreamingResponse(buf, media_type="image/png")

    return plt

def plot_heat_flux(length_x, length_y, time_total, dt):
    n_steps = int(time_total / dt)
    nx, ny = int(length_x / PlotConstants.dx), int(length_y / PlotConstants.dy)  # Number of grid points

    # Create the temperature field
    T = np.full((nx, ny), PlotConstants.T_initial)

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


    heat_flux_mold = []

    for step in range(n_steps):
        T_new = T.copy()

        # Update temperature
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dTdx2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / PlotConstants.dx**2
                dTdy2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / PlotConstants.dy**2
                T_new[i, j] = T[i, j] + dt * PlotConstants.k / (PlotConstants.rho * PlotConstants.cp) * (dTdx2 + dTdy2)

        # Boundary conditions
        T_new[:, 0] = PlotConstants.T_mold
        T_new[:, -1] = PlotConstants.T_ambient
        T_new[0, :] = PlotConstants.T_ambient
        T_new[-1, :] = PlotConstants.T_ambient

        T = T_new.copy()

        # Compute heat flux at the mold surface
        q_mold = -PlotConstants.k * (T[1, 0] - T[0, 0]) / PlotConstants.dx  # Simplified 1st derivative
        heat_flux_mold.append(q_mold)

    # Plot heat flux over time
    plt.plot(np.arange(0, time_total, dt), heat_flux_mold, label="Heat Flux at Mold Surface")
    plt.xlabel("Time (s)")
    plt.ylabel("Heat Flux (W/m²)")
    plt.title("Heat Flux at Mold Surface Over Time")
    plt.grid()
    # plt.show()

    # Save the plot to a byte stream
    # buf = io.BytesIO()
    # plt.savefig(buf, format="png")
    # buf.seek(0)
    # plt.close()  # Free up memory

    # # Return the plot as a streaming response
    # return StreamingResponse(buf, media_type="image/png")

    return plt



def plot_3d_temperature_distribution(length_x, length_y, time_total, dt):
    n_steps = int(time_total / dt)
    nx, ny = int(length_x / PlotConstants.dx), int(length_y / PlotConstants.dy)  # Number of grid points


    # Create the temperature field
    T = np.full((nx, ny), PlotConstants.T_initial)

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

        # Visualization at specific time steps
        if step % 100 == 0 or step == n_steps - 1:
            plt.imshow(T, extent=[0, length_x, 0, length_y], origin='lower', cmap='hot')
            plt.colorbar(label='Temperature (K)')
            plt.title(f'Temperature Distribution at Time = {step * dt:.2f} s')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.pause(0.1)

    # surface vs core temp
    core_temps = []
    surface_temps = []

    # Simulation loop with data collection
    for step in range(n_steps):
        T_new = T.copy()

        # Calculate temperature at interior points
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dTdx2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) /PlotConstants.dx**2
                dTdy2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) /PlotConstants.dy**2
                T_new[i, j] = T[i, j] + dt *PlotConstants.k / (PlotConstants.rho *PlotConstants.cp) * (dTdx2 + dTdy2)

        # Apply boundary conditions
        T_new[:, 0] =PlotConstants.T_mold
        T_new[:, -1] =PlotConstants.T_ambient
        T_new[0, :] =PlotConstants.T_ambient
        T_new[-1, :] =PlotConstants.T_ambient

        # Update the temperature field
        T = T_new.copy()

        # Collect core and surface temperatures
        core_temps.append(T[nx // 2, ny // 2])
        surface_temps.append(np.mean(T[:, 0]))

    # Plot core vs surface temperature
    time_steps = np.arange(0, time_total, dt)
    plt.plot(time_steps, core_temps, label="Core Temperature")
    plt.plot(time_steps, surface_temps, label="Surface Temperature")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Core vs. Surface Temperature over Time")
    plt.legend()
    plt.grid()

    # temp distribution bar width
    # Collect temperature profiles along a horizontal slice (midpoint)
    cross_section_temps = []

    for step in range(n_steps):
        T_new = T.copy()

        # Update temperature
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dTdx2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) /PlotConstants.dx**2
                dTdy2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) /PlotConstants.dy**2
                T_new[i, j] = T[i, j] + dt *PlotConstants.k / (PlotConstants.rho *PlotConstants.cp) * (dTdx2 + dTdy2)

        # Boundary conditions
        T_new[:, 0] =PlotConstants.T_mold
        T_new[:, -1] =PlotConstants.T_ambient
        T_new[0, :] =PlotConstants.T_ambient
        T_new[-1, :] =PlotConstants.T_ambient

        T = T_new.copy()

        # Record horizontal temperature profile at the midpoint
        if step % 500 == 0 or step == n_steps - 1:
            cross_section_temps.append((step * dt, T[:, ny // 2]))

    # Plot cross-sectional temperature distribution at different times
    for time, temp_profile in cross_section_temps:
        plt.plot(np.linspace(0, length_x, nx), temp_profile, label=f"t = {time:.2f}s")

    plt.xlabel("Position Along Width (m)")
    plt.ylabel("Temperature (K)")
    plt.title("Temperature Distribution Along Bar Width")
    plt.legend()
    plt.grid()
    # plt.show()

    # heat flux
    heat_flux_mold = []

    for step in range(n_steps):
        T_new = T.copy()

        # Update temperature
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dTdx2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) /PlotConstants.dx**2
                dTdy2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) /PlotConstants.dy**2
                T_new[i, j] = T[i, j] + dt *PlotConstants.k / (PlotConstants.rho *PlotConstants.cp) * (dTdx2 + dTdy2)

        # Boundary conditions
        T_new[:, 0] =PlotConstants.T_mold
        T_new[:, -1] =PlotConstants.T_ambient
        T_new[0, :] =PlotConstants.T_ambient
        T_new[-1, :] =PlotConstants.T_ambient

        T = T_new.copy()

        # Compute heat flux at the mold surface
        q_mold = -PlotConstants.k * (T[1, 0] - T[0, 0]) /PlotConstants.dx  # Simplified 1st derivative
        heat_flux_mold.append(q_mold)

    # Plot heat flux over time
    plt.plot(np.arange(0, time_total, dt), heat_flux_mold, label="Heat Flux at Mold Surface")
    plt.xlabel("Time (s)")
    plt.ylabel("Heat Flux (W/m²)")
    plt.title("Heat Flux at Mold Surface Over Time")
    plt.grid()
    # plt.show()

    #3d temp dist
    from mpl_toolkits.mplot3d import Axes3D
    # Generate 3D plot for the final temperature distribution
    X, Y = np.meshgrid(np.linspace(0, length_x, nx), np.linspace(0, length_y, ny))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, T.T, cmap="hot")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Temperature (K)")
    ax.set_title("3D Temperature Distribution at Final Time Step")
    # plt.show()

    return plt



def plot_pouring_bar_temperature_non_linear(length_x, length_y, time_total, dt):
    n_steps = int(time_total / dt)
    nx, ny = int(length_x / PlotConstants.dx), int(length_y / PlotConstants.dy)  # Number of grid points


    # Pouring temperatures to simulate (increased density)
    pouring_temps = np.arange(660 + 273.15, 720 + 273.15, 7)  # 660°C to 720°C in Kelvin
    cw_exit_temps = []  # Store cooling water exit temperatures
    roll_entry_temps = []  # Store roll entry temperatures
    for pouring_temp in pouring_temps:
        # Initialize temperature field with float type
        T = np.full((nx, ny), pouring_temp, dtype=np.float64)

        # Simulation loop
        for step in range(n_steps):
            T_new = T.copy()

            # Calculate temperature at interior points
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    dTdx2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / PlotConstants.dx**2
                    dTdy2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / PlotConstants.dy**2
                    T_new[i, j] = T[i, j] + dt * PlotConstants.k / (PlotConstants.rho * PlotConstants.cp) * (dTdx2 + dTdy2)

            # Apply non-linear cooling boundary conditions
            T_new[:, 0] -= dt * PlotConstants.h_conv / (PlotConstants.rho * PlotConstants.cp) * (T[:, 0] - PlotConstants.T_ambient)  # Cooling at mold
            T_new[:, -1] -= dt * (PlotConstants.h_conv * (T[:, -1] - PlotConstants.T_ambient) + PlotConstants.epsilon * PlotConstants.sigma * (T[:, -1]**4 - PlotConstants.T_ambient**4)) / (PlotConstants.rho * PlotConstants.cp)
            T_new[0, :] = PlotConstants.T_ambient  # Top boundary
            T_new[-1, :] = PlotConstants.T_ambient  # Bottom boundary

            # Update the temperature field
            T = T_new.copy()

        # Record temperatures
        cw_exit_temp = np.mean(T[:, 0])  # Cooling water exit temperature
        roll_entry_temp = np.mean(T[:, ny // 2])  # Roll entry temperature
        cw_exit_temps.append(cw_exit_temp)
        roll_entry_temps.append(roll_entry_temp)

    # Convert pouring temperatures and results to Celsius
    pouring_temps_celsius = pouring_temps - 273.15
    cw_exit_temps_celsius = np.array(cw_exit_temps) - 273.15
    roll_entry_temps_celsius = np.array(roll_entry_temps) - 273.15

    # Linear fits
    cw_slope, cw_intercept, _, _, _ = linregress(pouring_temps_celsius, cw_exit_temps_celsius)
    roll_slope, roll_intercept, _, _, _ = linregress(pouring_temps_celsius, roll_entry_temps_celsius)

    # Generate linear trendlines
    cw_linear = cw_slope * pouring_temps_celsius + cw_intercept
    roll_linear = roll_slope * pouring_temps_celsius + roll_intercept

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(pouring_temps_celsius, cw_exit_temps_celsius, 'o-', label="CW Exit Temp")
    plt.plot(pouring_temps_celsius, roll_entry_temps_celsius, 'o-', label="Roll Entry Temp")
    plt.plot(pouring_temps_celsius, cw_linear, '--', label="Linear (CW Exit Temp)")
    plt.plot(pouring_temps_celsius, roll_linear, '--', label="Linear (Roll Entry Temp)")
    plt.xlabel("Pouring Temperature (°C)")
    plt.ylabel("Bar Temperature (°C)")
    plt.title("Pouring Temperature vs Bar Temperatures with Non-linear Effects")
    plt.legend()
    plt.grid()
    # plt.show()

    # Save the plot to a byte stream
    # buf = io.BytesIO()
    # plt.savefig(buf, format="png")
    # buf.seek(0)
    # plt.close()  # Free up memory

    # # Return the plot as a streaming response
    # return StreamingResponse(buf, media_type="image/png")

    return plt

def plot_pouring_bar_temperature_wheel_and_roll():
    # Constants and parameters
    pouring_temps = np.arange(660, 721, 5)  # Pouring temps in Celsius (660°C to 720°C)
    RPM = 1.96  # Example RPM value (can be adjusted)

    # Calculate bar temperatures using the provided formulas
    bar_temp_wheel_exit = 530 + 73.29 * (RPM**1.96) + 0.431 * (pouring_temps - 705)
    bar_temp_roll_entry = 503 + 78 * (RPM - 1.96) + 0.5 * (pouring_temps - 705)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(pouring_temps, bar_temp_wheel_exit, 'o-', label="Bar Temperature at Wheel Exit")
    plt.plot(pouring_temps, bar_temp_roll_entry, 'o-', label="Bar Temperature at Roll Entry")

    # Labels and title
    plt.xlabel("Pouring Temperature (°C)")
    plt.ylabel("Bar Temperature (°C)")
    plt.title("Pouring Temperature vs Bar Temperature (Wheel Exit and Roll Entry)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    # plt.show()

    # Save the plot to a byte stream
    # buf = io.BytesIO()
    # plt.savefig(buf, format="png")
    # buf.seek(0)
    # plt.close()  # Free up memory

    # # Return the plot as a streaming response
    # return StreamingResponse(buf, media_type="image/png")

    return plt


def plot_rpm_surface_temp_wheel_and_roll():
    # Constants and parameters
    pouring_temp = 700  # Example constant pouring temperature in Celsius
    RPM_range = np.arange(1, 11, 0.1)  # RPM range from 1 to 10 with a step of 0.1

    # Calculate bar temperatures using the provided formulas
    bar_temp_wheel_exit = 530 + 73.29 * (RPM_range**1.96) + 0.431 * (pouring_temp - 705)
    bar_temp_roll_entry = 503 + 78 * (RPM_range - 1.96) + 0.5 * (pouring_temp - 705)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(RPM_range, bar_temp_wheel_exit, label="Bar Temperature at Wheel Exit")
    plt.plot(RPM_range, bar_temp_roll_entry, label="Bar Temperature at Roll Entry")

    # Labels and title
    plt.xlabel("RPM")
    plt.ylabel("Surface Temperature (°C)")
    plt.title("RPM vs Surface Temperature (Wheel Exit and Roll Entry)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    # plt.show()

    # Save the plot to a byte stream
    # buf = io.BytesIO()
    # plt.savefig(buf, format="png")
    # buf.seek(0)
    # plt.close()  # Free up memory

    # # Return the plot as a streaming response
    # return StreamingResponse(buf, media_type="image/png")

    return plt



def plot_pouring_temp_vs_bar_temp(rpm:float, pouring_temps:np.array):
    # pouring_temps = np.arange(660, 721, 5)  # Pouring temperature from 660°C to 720°C
    # rpm = float(input("Enter RPM value for the simulation: "))

    bar_temp_wheel_exit = []
    bar_temp_roll_entry = []

    for pouring_temp in pouring_temps:
        wheel_exit, roll_entry = calculate_bar_temperature(pouring_temp, rpm)
        bar_temp_wheel_exit.append(wheel_exit)
        bar_temp_roll_entry.append(roll_entry)

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(pouring_temps, bar_temp_wheel_exit, label="Bar Temperature at Wheel Exit")
    plt.plot(pouring_temps, bar_temp_roll_entry, label="Bar Temperature at Roll Entry")

    plt.xlabel("Pouring Temperature (°C)")
    plt.ylabel("Bar Temperature (°C)")
    plt.title(f"Pouring Temperature vs Bar Temperature (RPM = {rpm})")
    plt.legend()
    plt.grid(True)
    # plt.show()

    return plt


def plot_rpm_vs_surface_temp(pouring_temp:float):
    RPM_range = np.arange(1, 11, 0.1)  # RPM range from 1 to 10
    # pouring_temp = float(input("Enter Pouring Temperature (°C): "))

    bar_temp_wheel_exit = []
    bar_temp_roll_entry = []

    for RPM in RPM_range:
        wheel_exit, roll_entry = calculate_bar_temperature(pouring_temp, RPM)
        bar_temp_wheel_exit.append(wheel_exit)
        bar_temp_roll_entry.append(roll_entry)

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(RPM_range, bar_temp_wheel_exit, label="Bar Temperature at Wheel Exit")
    plt.plot(RPM_range, bar_temp_roll_entry, label="Bar Temperature at Roll Entry")

    plt.xlabel("RPM")
    plt.ylabel("Surface Temperature (°C)")
    plt.title(f"RPM vs Surface Temperature (Pouring Temp = {pouring_temp}°C)")
    plt.legend()
    plt.grid(True)
    # plt.show()

    return plt


def plot_temperature_profile():
    bar_length = np.linspace(0, 100, 100)  # Length of the bar (normalized)
    cooling_rate_surface = np.linspace(1, 0, 100)  # Example cooling rate for the surface
    cooling_rate_core = np.linspace(0, 1, 100)  # Example cooling rate for the core

    temperature_surface = 700 - cooling_rate_surface * 100  # Starting temperature 700°C
    temperature_core = 700 - cooling_rate_core * 200  # Starting temperature 700°C

    plt.figure(figsize=(10, 6))
    plt.plot(bar_length, temperature_surface, label="Surface Temperature")
    plt.plot(bar_length, temperature_core, label="Core Temperature")

    plt.xlabel("Position along Bar Length (%)")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Profile along Bar Length")
    plt.legend()
    plt.grid(True)
    # plt.show()

    return plt


def plot_sensitivity_of_temperature():
    # Sensitivity analysis for casting speed and pouring temperature
    casting_speeds = np.arange(1, 10, 0.5)  # Casting speeds
    pouring_temps = np.arange(660, 721, 5)  # Pouring temperature from 660°C to 720°C
    bar_temp_wheel_exit = np.zeros((len(casting_speeds), len(pouring_temps)))

    for i, casting_speed in enumerate(casting_speeds):
        for j, pouring_temp in enumerate(pouring_temps):
            wheel_exit, _ = calculate_bar_temperature(pouring_temp, casting_speed)
            bar_temp_wheel_exit[i, j] = wheel_exit

    plt.figure(figsize=(10, 6))
    for i, casting_speed in enumerate(casting_speeds):
        plt.plot(pouring_temps, bar_temp_wheel_exit[i, :], label=f"Casting Speed = {casting_speed}")

    plt.xlabel("Pouring Temperature (°C)")
    plt.ylabel("Bar Temperature at Wheel Exit (°C)")
    plt.title("Sensitivity of Bar Temperature to Casting Speed and Pouring Temperature")
    plt.legend()
    plt.grid(True)
    # plt.show()

    return plt


def plot_operating_window(rpm:float):
    # Operating window graph based on temperature constraints
    max_exit_temp = 545  # Maximum temperature at wheel exit
    min_exit_temp = 535  # Minimum temperature at wheel exit
    max_roll_entry_temp = 515  # Maximum temperature at roll entry
    min_roll_entry_temp = 505  # Minimum temperature at roll entry

    pouring_temps = np.arange(660, 721, 5)  # Pouring temperature from 660°C to 720°C
    # rpm = float(input("Enter RPM value for the operating window: "))

    bar_temp_wheel_exit = []
    bar_temp_roll_entry = []

    for pouring_temp in pouring_temps:
        wheel_exit, roll_entry = calculate_bar_temperature(pouring_temp, rpm)
        bar_temp_wheel_exit.append(wheel_exit)
        bar_temp_roll_entry.append(roll_entry)

    # Plotting the operating window
    plt.figure(figsize=(10, 6))
    plt.plot(pouring_temps, bar_temp_wheel_exit, label="Bar Temperature at Wheel Exit")
    plt.plot(pouring_temps, bar_temp_roll_entry, label="Bar Temperature at Roll Entry")
    plt.axhline(y=max_exit_temp, color='r', linestyle='--', label="Max Exit Temp")
    plt.axhline(y=min_exit_temp, color='r', linestyle='--', label="Min Exit Temp")
    plt.axhline(y=max_roll_entry_temp, color='g', linestyle='--', label="Max Roll Entry Temp")
    plt.axhline(y=min_roll_entry_temp, color='g', linestyle='--', label="Min Roll Entry Temp")

    plt.xlabel("Pouring Temperature (°C)")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Operating Window for Caster (RPM = {rpm})")
    plt.legend()
    plt.grid(True)
    # plt.show()

    return plt

