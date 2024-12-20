import React from "react";
import FeatureCard from "../components/FeatureCard";
import useStore from "../store/useStore";

const DefaultPage = () => {
  const labelMap = {
    EMUL_OIL_L_TEMP_PV_VAL0: "Emulsion Temp",
    STAND_OIL_L_TEMP_PV_REAL_VAL0: "Stand Oil Temp",
    GEAR_OIL_L_TEMP_PV_REAL_VAL0: "Gear Oil Temp",
    EMUL_OIL_L_PR_VAL0: "Emulsion Press",
    QUENCH_CW_FLOW_EXIT_VAL0: "Cooling Outflow",
    CAST_WHEEL_RPM_VAL0: "Wheel Speed",
    BAR_TEMP_VAL0: "Bar Temp",
    QUENCH_CW_FLOW_ENTRY_VAL0: "Cooling Inflow",
    GEAR_OIL_L_PR_VAL0: "Gear Press",
    STANDS_OIL_L_PR_VAL0: "Stand Press",
    TUNDISH_TEMP_VAL0: "Tundish Temp",
    RM_MOTOR_COOL_WATER__VAL0: "Motor Cooling",
    ROLL_MILL_AMPS_VAL0: "Mill Current",
    RM_COOL_WATER_FLOW_VAL0: "Mill Cool Flow",
    EMULSION_LEVEL_ANALO_VAL0: "Emulsion Level",
    furnace_temp: "Furnace Temp",
    "%AL": "Al %",
  };

  const { formData, updateFormData } = useStore();
  const {dsbl} = useStore();

  return (
      <div className="py-4 h-full overflow-y-scroll gap-4 px-4 lg:px-8 md:px-8  lg:grid-cols-3 md:grid-cols-2 noscrollbar">
        {(Object.keys(labelMap)).map((key) => (
          <FeatureCard
            title={labelMap[key]}
            value={formData[key]}
            key={key}
            dsbl={dsbl}
            onChange={(value) => updateFormData(key, value)}
          />
        ))}
      </div>
  );
};

export default DefaultPage;
