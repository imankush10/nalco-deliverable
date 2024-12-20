import { create } from "zustand";

export const VALIDATION_RANGES = {
  EMUL_OIL_L_TEMP_PV_VAL0: { min: 28.77317, max: 61.1067 },
  STAND_OIL_L_TEMP_PV_REAL_VAL0: { min: 26.96252, max: 57.50177 },
  GEAR_OIL_L_TEMP_PV_REAL_VAL0: { min: 29.4476, max: 64.43733 },
  EMUL_OIL_L_PR_VAL0: { min: 1, max: 2.70433 },
  ROD_DIA_MM_VAL0: { min: 9.5, max: 12 },
  CAST_WHEEL_RPM_VAL0: { min: 0.4654, max: 3.29 },
  BAR_TEMP_VAL0: { min: 199.89, max: 600 },
  QUENCH_CW_FLOW_ENTRY_VAL0: { min: 100, max: 441.9 },
  GEAR_OIL_L_PR_VAL0: { min: 0.5, max: 2.5124 },
  STANDS_OIL_L_PR_VAL0: { min: 0.11, max: 2.842536 },
  TUNDISH_TEMP_VAL0: { min: 550, max: 1070.3 },
  RM_MOTOR_COOL_WATER__VAL0: { min: 1.044956207, max: 1.529836655 },
  ROLL_MILL_AMPS_VAL0: { min: 200, max: 767.6586914 },
  RM_COOL_WATER_FLOW_VAL0: { min: 193.6369019, max: 221.5656738 },
  EMULSION_LEVEL_ANALO_VAL0: { min: 195.6206055, max: 1380.932373 },
  furnace_temp: { min: 550, max: 1115.69104 },
  "%SI": { min: 0.06, max: 0.13 },
  "%FE": { min: 0.14, max: 0.26 },
  "%TI": { min: 0.001, max: 0.007 },
  "%V": { min: 0.001, max: 0.015 },
  "%MN": { min: 0.003, max: 0.005 },
  OTHIMP: { min: 0.002, max: 0.02 },
  "%AL": { min: 99.618, max: 99.753 },
};

const INITIAL_FORM_DATA = {
  EMUL_OIL_L_TEMP_PV_VAL0: 45.65,
  STAND_OIL_L_TEMP_PV_REAL_VAL0: 40.12,
  GEAR_OIL_L_TEMP_PV_REAL_VAL0: 50.32,
  EMUL_OIL_L_PR_VAL0: 2.1,
  ROD_DIA_MM_VAL0: 10.8,
  CAST_WHEEL_RPM_VAL0: 2.45,
  BAR_TEMP_VAL0: 450.55,
  QUENCH_CW_FLOW_ENTRY_VAL0: 320.65,
  GEAR_OIL_L_PR_VAL0: 1.8,
  STANDS_OIL_L_PR_VAL0: 1.5,
  TUNDISH_TEMP_VAL0: 800.23,
  RM_MOTOR_COOL_WATER__VAL0: 1.35,
  ROLL_MILL_AMPS_VAL0: 500.25,
  RM_COOL_WATER_FLOW_VAL0: 200.45,
  EMULSION_LEVEL_ANALO_VAL0: 900.5,
  furnace_temp: 800.89,
  "%SI": 0.08,
  "%FE": 0.2,
  "%TI": 0.003,
  "%V": 0.008,
  "%MN": 0.004,
  OTHIMP: 0.01,
  "%AL": 99.7,
  Diameter: 9.5,
  Coil: 9,
  castNo: 170,
  "Wt(MT)": 190,
  QUENCH_CW_FLOW_EXIT_VAL0: 100,
};

export const REV_VALIDATION_RANGES = {
  UTS: { min: 6.5, max: 15 },
  Elongation: { min: 13, max: 28 },
  Conductivity: { min: 48, max: 70 },
};

const INITIAL_REVFORM_DATA = {
  UTS: 10.23,
  Elongation: 16.4,
  Conductivity: 61.1,
};

const API_BASE_URL = "http://127.0.0.1:8000";

const apiCall = async (endpoint, options = {}) => {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
};

const useStore = create((set, get) => ({
  formData: INITIAL_FORM_DATA,
  revFormData: INITIAL_REVFORM_DATA,
  result: null,
  csvResult: null,
  revResult: null,
  selectedFile: null,
  showDetails: false,
  isLoading: false,
  currentStep: 2,
  error: false,
  trainingFile: null,
  availableColumns: null,
  validationErrors: [],
  revValidationErrors: [],
  plcCounter: 745237,
  plotImage: null,
  plotError: null,
  plotLoading: false,
  dsbl: false,
  isPLC: false,

  setIsPLC: (valueOrUpdater) => {
    set((state) => ({
      isPLC:
        typeof valueOrUpdater === "function"
          ? valueOrUpdater(state.isPLC)
          : valueOrUpdater,
    }));
  },

  setDisabled: () => set({ disabled: true }),

  setTrainRatio: (ratio) => set({ trainRatio: ratio }),
  setTargetColumn: (column) => set({ targetColumn: column }),
  setOptimizeMetric: (metric) => set({ optimizeMetric: metric }),
  setCurrentStep: (step) => set({ currentStep: step }),

  updateLoading: (value) => {
    set({ isLoading: value });
  },

  toggleShowDetails: () =>
    set((state) => ({ showDetails: !state.showDetails })),

  updateFormData: (key, value) =>
    set((state) => ({
      formData: { ...state.formData, [key]: value },
    })),

  updateRevFormData: (key, value) =>
    set((state) => ({
      revFormData: {
        ...state.revFormData,
        [key]: parseFloat(value.toString()),
      },
    })),

  handleFileSelect: (file) => set({ selectedFile: file }),

  updatePlc: () => {
    set((state) => ({ plcCounter: state.plcCounter + 1 }));
  },

  resetResults: () => {
    set({
      result: null,
      csvResult: null,
      showDetails: false,
      revResult: null,
      validationErrors: [],
      revValidationErrors: [],
      plcCounter: 745237,
      isLoading: false,
    });
  },

  validateFormData: () => {
    const formData = get().formData;
    const errors = [];

    Object.entries(formData).forEach(([key, value]) => {
      const range = VALIDATION_RANGES[key];
      if (range && (value < range.min || value > range.max)) {
        errors.push(
          `${key} value is out of range (${range.min.toFixed(
            2
          )} - ${range.max.toFixed(2)})`
        );
      }
    });

    set({ validationErrors: errors });
    return errors;
  },

  validateRevFormData: () => {
    const revFormData = get().revFormData;
    const errors = [];

    Object.entries(revFormData).forEach(([key, value]) => {
      const range = REV_VALIDATION_RANGES[key];
      if (range && (value < range.min || value > range.max)) {
        errors.push(
          `${key} value is out of range (${range.min.toFixed(
            2
          )} - ${range.max.toFixed(2)})`
        );
      }
    });

    set({ revValidationErrors: errors });
    return errors;
  },

  handleGenerate: async () => {
    const formData = get().formData;
    const validationErrors = get().validateFormData();

    if (validationErrors.length > 0) {
      return false;
    }

    try {
      const result = await apiCall("/predict_all", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: formData }),
      });
      set({ result, validationErrors: [] });
      return true;
    } catch (error) {
      console.error("Error during prediction:", error);
      alert("Failed to generate prediction. Please try again.");
      return false;
    }
  },

  handleReverseGenerate: async () => {
    set({ isLoading: true });
    const validationErrors = get().validateRevFormData();

    if (validationErrors.length > 0) {
      set({ isLoading: false });
      return false;
    }

    try {
      const result = await apiCall("/reverse-predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(get().revFormData),
      });

      set({ revResult: result, isLoading: false });

      return true;
    } catch (error) {
      console.error("Error during reverse prediction:", error);
      alert("Failed to generate reverse prediction. Please try again.");
      return false;
    }
  },

  counter: 0,
  realTimeData: [],

  handlePLC: async () => {
    try {
      console.log("sup");
      const res = await apiCall("/predict-real-time", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          port: import.meta.env.VITE_port,
          slave_id: import.meta.env.VITE_slave_id,
          start_address: import.meta.env.VITE_start_address,
          num_registers: import.meta.env.VITE_num_registers,
        }),
      });

      const { counter, realTimeData } = get();

      const newData = {
        counter,
        predictions: res.predictions,
      };

      set({
        realTimeData: [...realTimeData, newData],
        counter: counter + 10,
      });

      const FEATURES = [
        "EMUL_OIL_L_TEMP_PV_VAL0",
        "STAND_OIL_L_TEMP_PV_REAL_VAL0",
        "GEAR_OIL_L_TEMP_PV_REAL_VAL0",
        "EMUL_OIL_L_PR_VAL0",
        "ROD_DIA_MM_VAL0",
        "QUENCH_CW_FLOW_EXIT_VAL0",
        "CAST_WHEEL_RPM_VAL0",
        "BAR_TEMP_VAL0",
        "QUENCH_CW_FLOW_ENTRY_VAL0",
        "GEAR_OIL_L_PR_VAL0",
        "STANDS_OIL_L_PR_VAL0",
        "TUNDISH_TEMP_VAL0",
        "RM_MOTOR_COOL_WATER__VAL0",
        "ROLL_MILL_AMPS_VAL0",
        "RM_COOL_WATER_FLOW_VAL0",
        "EMULSION_LEVEL_ANALO_VAL0",
        "furnace_temp",
        "%SI",
        "%FE",
        "%TI",
        "%V",
        "%MN",
        "OTHIMP",
        "%AL",
      ];
      const modbusDataDict = Object.fromEntries(
        FEATURES.map((feature, index) => [feature, res.modbus_data[index]])
      );

      console.log(modbusDataDict, res.predictions);

      get().updateLoading(false);
      get().updatePlc();
    } catch (error) {
      set({ isPLC: false });
      console.log(error);
    }
  },

  handleUpload: async () => {
    set({ isLoading: true });
    const file = get().selectedFile;
    if (!file) {
      alert("Please select a file first.");
      return false;
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      const result = await apiCall("/upload-predict", {
        method: "POST",
        body: formData,
      });

      set({ isLoading: false });
      console.log(result);
      set({ csvResult: result });
      return true;
    } catch (error) {
      set({ isLoading: false });
      alert("Error uploading file.");
      console.error("Error during file upload:", error);
      set({ selectedFile: null });
      return false;
    }
  },

  downloadLink: async () => {
    try {
      const downloadUrl = get().csvResult?.download_url;
      if (!downloadUrl) throw new Error("Download URL not found.");

      const response = await fetch(`${API_BASE_URL}${downloadUrl}`);
      const blob = await response.blob();

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = downloadUrl.split("/")[2];
      link.click();
      URL.revokeObjectURL(url);
      return true;
    } catch (error) {
      console.error("Error downloading file:", error);
      alert("Failed to download file. Please try again.");
      return false;
    }
  },

  handleTrainingFileUpload: async () => {
    try {
      const file = get().selectedFile;
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/upload/`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error);

      set({
        trainingFile: file,
        availableColumns: data.columns,
        currentStep: 2,
      });

      return true;
    } catch (error) {
      console.error("Error uploading training file:", error);
      alert("Failed to upload file");
      return false;
    }
  },

  handleSplitData: async () => {
    try {
      const formData = new FormData();
      formData.append("train_ratio", get().trainRatio?.toString() || "");

      const response = await fetch(`${API_BASE_URL}/split/`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error);

      set({ currentStep: 3 });
      return true;
    } catch (error) {
      console.error("Error splitting data:", error);
      alert("Failed to split data");
      return false;
    }
  },

  handleTrainModel: async () => {
    try {
      const formData = new FormData();
      formData.append("target_column", get().targetColumn || "");

      const response = await fetch(`${API_BASE_URL}/train/`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error);

      set({ currentStep: 4 });
      return true;
    } catch (error) {
      console.error("Error training model:", error);
      alert("Failed to train model");
      return false;
    }
  },

  handleTuneModel: async () => {
    try {
      const formData = new FormData();
      formData.append("optimize_metric", get().optimizeMetric || "");

      const response = await fetch(`${API_BASE_URL}/tune/`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error);

      set({ currentStep: 5 });
      return true;
    } catch (error) {
      console.error("Error tuning model:", error);
      alert("Failed to tune model");
      return false;
    }
  },

  handleSaveModel: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/save/`, {
        method: "POST",
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error);

      alert("Model saved successfully!");
      return true;
    } catch (error) {
      console.error("Error saving model:", error);
      alert("Failed to save model");
      return false;
    }
  },

  generatePlot: async (plotType, params) => {
    set({ plotImages: null });
    set({ plotLoading: true, plotError: null });

    try {
      const response = await fetch(`${API_BASE_URL}/plot`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          plot_type: plotType,
          data: params,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to generate plot");
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      console.log(imageUrl);

      set({ plotImage: imageUrl, plotLoading: false });
      return true;
    } catch (error) {
      set({
        plotError: error instanceof Error ? error.message : "An error occurred",
        plotLoading: false,
      });
      return false;
    }
  },

  generateTemperatureDistribution: async (params) => {
    set({ plotImage: null });
    set({ plotLoading: true, plotError: null });
    try {
      const response = await fetch(
        `${API_BASE_URL}/plot-temperature-distribution`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(params),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.error || "Failed to generate temperature distribution"
        );
      }

      const { plots } = await response.json();
      console.log(plots);

      const imageUrls = plots.map((plot) => `data:image/png;base64,${plot}`);

      set({ plotImages: imageUrls, plotLoading: false });
      return true;
    } catch (error) {
      set({
        plotError: error instanceof Error ? error.message : "An error occurred",
        plotLoading: false,
      });
      return false;
    }
  },

  // Helper function to get required parameters for a plot type
  getRequiredParams: (plotType) => {
    const paramMap = {
      CORE_SURFACE_TEMPERATURE: ["length_x", "length_y", "time_total", "dt"],
      TEMPERATURE_DISTRIBUTION_BAR_WIDTH: [
        "length_x",
        "length_y",
        "time_total",
        "dt",
      ],
      HEAT_FLUX: ["length_x", "length_y", "time_total", "dt"],
      THREE_D_TEMPERATURE_DISTRIBUTION: [
        "length_x",
        "length_y",
        "time_total",
        "dt",
      ],
      POURING_TEMPERATURE_VS_BAR_TEMPERATURE_NON_LINEAR: [
        "length_x",
        "length_y",
        "time_total",
        "dt",
      ],
      POURING_TEMPERATURE_VS_BAR_TEMPERATURE_WHEEL_ROLL: [],
      RPM_VS_SURFACE_TEMPERATURE: [],
      TEMPERATURE_PROFILE: [],
      PLOT_POURING_TEMP_VS_BAR_TEMP: ["rpm", "pouring_temps"],
      RPM_VS_SURFACE_TEMP: ["pouring_temp"],
      SENSITIVITY_OF_TEMPERATURE: [],
      OPERATING_WINDOW: ["rpm"],
      MULTIPLE_PLOTTING: ["length_x", "length_y", "time_total", "dt"],
    };

    return paramMap[plotType] || [];
  },

  clearPlot: () => {
    if (get().plotImage) {
      URL.revokeObjectURL(get().plotImage);
    }
    set({ plotImage: null, plotError: null });
  },

  logState: () => console.log(get()),
}));

export default useStore;
