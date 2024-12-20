from src.plots.plots import *
# from plots import *


CORE_SURFACE_TEMPERATURE = "core–surface-temperature"
TEMPERATURE_DISTRIBUTION_BAR_WIDTH = "temperature-distribution-bar-width"
HEAT_FLUX = "heat-flux"
THREE_D_TEMPERATURE_DISTRIBUTION = "3d-temperature-distribution"
POURING_TEMPERATURE_VS_BAR_TEMPERATURE_NON_LINEAR = "pouring-temperature-vs-bar-temperature-non-linear"
POURING_TEMPERATURE_VS_BAR_TEMPERATURE_WHEEL_ROLL = "pouring-temperature-vs-bar-temperature-wheel-roll"
RPM_VS_SURFACE_TEMPERATURE = "rpm-vs-surface-temperature"
TEMPERATURE_PROFILE = "temperature-profile"
PLOT_POURING_TEMP_VS_BAR_TEMP = "plot-pouring-temp-vs-bar-temp"
RPM_VS_SURFACE_TEMP = "rpm-vs-surface-temp"
SENSITIVITY_OF_TEMPERATURE = "sensitivity-of-temperature"
OPERATING_WINDOW = "operating-window"

SINGLE_PLOTS = {
    CORE_SURFACE_TEMPERATURE: plot_core_surface_temp,
    TEMPERATURE_DISTRIBUTION_BAR_WIDTH: plot_temperature_distribution_bar_width,
    HEAT_FLUX: plot_heat_flux,
    THREE_D_TEMPERATURE_DISTRIBUTION: plot_3d_temperature_distribution,
    POURING_TEMPERATURE_VS_BAR_TEMPERATURE_NON_LINEAR: plot_pouring_bar_temperature_non_linear,
    POURING_TEMPERATURE_VS_BAR_TEMPERATURE_WHEEL_ROLL: plot_pouring_bar_temperature_wheel_and_roll,
    RPM_VS_SURFACE_TEMPERATURE: plot_rpm_surface_temp_wheel_and_roll,
    TEMPERATURE_PROFILE: plot_temperature_profile,
    PLOT_POURING_TEMP_VS_BAR_TEMP: plot_pouring_temp_vs_bar_temp,
    RPM_VS_SURFACE_TEMP: plot_rpm_vs_surface_temp,
    SENSITIVITY_OF_TEMPERATURE: plot_sensitivity_of_temperature,
    OPERATING_WINDOW: plot_operating_window,
}

SINGLE_PLOTS_PARAMS = {
    CORE_SURFACE_TEMPERATURE: ['length_x', 'length_y', 'time_total', 'dt'],
    TEMPERATURE_DISTRIBUTION_BAR_WIDTH: ['length_x', 'length_y', 'time_total', 'dt'],
    HEAT_FLUX: ['length_x', 'length_y', 'time_total', 'dt'],
    THREE_D_TEMPERATURE_DISTRIBUTION: ['length_x', 'length_y', 'time_total', 'dt'],
    POURING_TEMPERATURE_VS_BAR_TEMPERATURE_NON_LINEAR: ['length_x', 'length_y', 'time_total', 'dt'],

    POURING_TEMPERATURE_VS_BAR_TEMPERATURE_WHEEL_ROLL: [],
    RPM_VS_SURFACE_TEMPERATURE: [],
    TEMPERATURE_PROFILE: [],
    SENSITIVITY_OF_TEMPERATURE: [],

    PLOT_POURING_TEMP_VS_BAR_TEMP: ['rpm', 'pouring_temps'],
    RPM_VS_SURFACE_TEMP: ['pouring_temp'],
    OPERATING_WINDOW: ['rpm']
}



def call_single_plot(plot_name: str, params: dict):
    if plot_name not in SINGLE_PLOTS:
        raise KeyError(f"Invalid plot name: {plot_name}")

    required_params = SINGLE_PLOTS_PARAMS.get(plot_name, [])
    missing_params = [param for param in required_params if param not in params]

    if missing_params:
        raise ValueError(f"Missing parameters for {plot_name}: {', '.join(missing_params)}")

    try:
        return SINGLE_PLOTS[plot_name](**{k: params[k] for k in required_params})
    except Exception as e:
        print(f"Error calling plot function for {plot_name}: {e}")
        raise



# # ========================================================
# # Get parameters of a function
# # ========================================================
# import inspect
# # Assuming the functions are defined in the same module
# def get_function_parameters(func):
#     signature = inspect.signature(func)
#     return list(signature.parameters.keys())
# # Create a new dictionary with function names and their parameters
# function_parameters = {name: get_function_parameters(func) for name, func in SINGLE_PLOTS.items()}

# temp = {}
# # Print the result
# for name, params in function_parameters.items():
#     # print(f"Function '{name}' has parameters: {params}")
#     temp[name] = params

# print(temp)
# # ========================================================