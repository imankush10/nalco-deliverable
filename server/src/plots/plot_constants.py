class PlotConstants:
    # TODO: remove constants: length_x, length_y, time_total, dt

    # Constants and parameters
    # length_x = 0.1  # Length in the x-direction (m)
    # length_y = 0.1  # Length in the y-direction (m)
    dx = 0.005      # Spatial step in x (m)
    dy = 0.005      # Spatial step in y (m)
    # nx, ny = int(length_x / dx), int(length_y / dy)  # Number of grid points

    # time_total = 10.0    # Total simulation time (s)
    # dt = 0.01            # Time step (s)
    # n_steps = int(time_total / dt)

    # Material properties (e.g., aluminum)
    k = 200  # Thermal conductivity (W/m.K)
    rho = 2700  # Density (kg/m^3)
    cp = 900  # Specific heat capacity (J/kg.K)

    # Boundary and initial conditions
    T_initial = 700.0  # Initial temperature (K)
    T_mold = 300.0     # Temperature of the mold (K)
    h_eff = 2200       # Heat transfer coefficient (W/m^2.K)
    T_ambient = 300.0  # Ambient temperature (K)

    # Latent heat properties
    latent_heat = 397000  # J/kg
    T_solidus = 660       # Solidus temperature (K)
    T_liquidus = 700      # Liquidus temperature (K)

    # Non-linear cooling parameters
    h_conv = 15  # Increased convective heat transfer coefficient (W/m^2.K)
    epsilon = 0.8  # Emissivity
    sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m^2.K^4)

