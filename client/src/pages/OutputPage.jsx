import React from "react";
import ResultCard from "../components/ResultCard";
import { AirVent, Ruler, Zap } from "lucide-react";
import useStore from "../store/useStore";
import Button from "../components/Button";
import { Audio, DNA, Puff, Triangle } from "react-loader-spinner";
import AnimatedProcessInstructions from "../components/AnimatedProcessInstructions.jsx";

const OutputPage = () => {
  const {
    downloadLink,
    result,
    csvResult,
    showDetails,
    toggleShowDetails,
    revResult,
    validationErrors,
    revValidationErrors,
    isLoading,
    plcCounter,
  } = useStore();

  const labelMap = {
    EMUL_OIL_L_TEMP_PV_VAL0: "Emulsion Temp",
    STAND_OIL_L_TEMP_PV_REAL_VAL0: "Stand Oil Temp",
    GEAR_OIL_L_TEMP_PV_REAL_VAL0: "Gear Oil Temp",
    EMUL_OIL_L_PR_VAL0: "Emulsion Press",
    ROD_DIA_MM_VAL0: "Rod Diameter",
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
    "%SI": "Si %",
    "%FE": "Fe %",
    "%TI": "Ti %",
    "%V": "V %",
    "%MN": "Mn %",
    OTHIMP: "Other Impurities",
    "%AL": "Al %",
    UTS: "UTS",
    Elongation: "Elongation",
    Conductivity: "Conductivity",
  };

  return (
    <div className="border-2 border-t-0 rounded-xl h-full pb-16 shadow-xl max-w-[600px] min-w-[550px]">
      <div className="flex items-center gap-2 bg-red-800 rounded-t-xl p-4 text-white">
        <span className="text-xl bg-red-800">Prediction Result</span>
      </div>
      <div className="px-4 h-full overflow-y-scroll noscrollbar">
        {isLoading && (
          <div className="h-full w-full flex items-center justify-center">
            <DNA
              visible={true}
              height="120"
              width="120"
              ariaLabel="dna-loading"
              wrapperStyle={{}}
              wrapperClass="dna-wrapper"
            />
          </div>
        )}
        {validationErrors.length > 0 && (
          <div className="mt-4">
            <h3 className="text-3xl font-bold text-red-800 mb-4">
              Process Validation Errors
            </h3>
            {validationErrors.map((error, index) => {
              const [key, ...messageParts] = error.split(" ");
              const label = labelMap[key] || key;
              const message = messageParts.join(" ");
              return (
                <div
                  key={index}
                  className="border-2 border-red-800 rounded-lg p-4 mb-4"
                >
                  <p className="text-red-800 text-xl">
                    {label} {message}
                  </p>
                </div>
              );
            })}
          </div>
        )}
        {revValidationErrors.length > 0 && (
          <div className="mt-4">
            <h3 className="text-3xl font-bold text-red-800 mb-4">
              Target Validation Errors
            </h3>
            {revValidationErrors.map((error, index) => {
              const [key, ...messageParts] = error.split(" ");
              const label = labelMap[key] || key;
              const message = messageParts.join(" ");
              return (
                <div
                  key={index}
                  className="border-2 border-red-800 rounded-lg p-4 mb-4"
                >
                  <p className="text-red-800 text-xl">
                    {label} {message}
                  </p>
                </div>
              );
            })}
          </div>
        )}
        {result && !validationErrors.length && (
          <div className="flex flex-col gap-4 mt-4">
            <ResultCard
              value={`${result.uts.toFixed(2)} MPa`}
              title="UTS"
              Icon={AirVent}
            />
            <ResultCard
              title="Elongation"
              value={`${result.elongation.toFixed(2)} %`}
              Icon={Ruler}
            />
            <ResultCard
              title="Conductivity"
              value={`${result.conductivity.toFixed(2)}% IACS`}
              Icon={Zap}
            />
          </div>
        )}
        {csvResult && (
          <>
            <div className="flex gap-4 mt-4">
              <Button type="filled" value="Download" onClick={downloadLink} />
              <Button type="filled" value="Peek" onClick={toggleShowDetails} />
            </div>
            {csvResult.metrics && (
              <div className="mt-4">
                <h2 className="text-2xl font-bold mb-4">Metrics</h2>
                <div className="flex flex-col gap-4">
                  {["uts", "conductivity", "elongation"].map((metric) => (
                    <div key={metric} className="p-4 rounded-lg shadow-md">
                      <h3 className="text-xl font-semibold mb-2">
                        {metric.charAt(0).toUpperCase() + metric.slice(1)}
                      </h3>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="border p-2 rounded-lg text-center">
                          <p className="text-lg">MSE</p>
                          <p className="font-bold text-xl">
                            {csvResult.metrics[metric].mse.toFixed(5)}
                          </p>
                        </div>
                        <div className="border p-2 rounded-lg text-center">
                          <p className="text-lg">MAE</p>
                          <p className="font-bold text-xl">
                            {csvResult.metrics[metric].mae.toFixed(5)}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {showDetails && (
              <div className="flex flex-col gap-4 mt-4">
                {csvResult.preview.map((resultObj, index) => (
                  <div key={`test-case-${index}`}>
                    <p className="text-xl font-bold">Test case {index + 1}</p>
                    <div className="grid grid-cols-3 gap-2">
                      <ResultCard
                        value={`${resultObj.uts_predict.toFixed(2)} MPa`}
                        title="UTS"
                        Icon={AirVent}
                      />
                      <ResultCard
                        value={`${resultObj.elongation_predict.toFixed(2)} %`}
                        title="Elongation"
                        Icon={Ruler}
                      />
                      <ResultCard
                        value={`${resultObj.conductivity_predict.toFixed(
                          2
                        )}% IACS`}
                        title="Conductivity"
                        Icon={Zap}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
        {revResult && !revValidationErrors.length && (
          <div className="mt-8">
            <AnimatedProcessInstructions
              revResult={revResult}
              revValidationErrors={revValidationErrors}
              labelMap={labelMap}
            />
          </div>
        )}
        {plcCounter != 745237 && (
          <div className="flex flex-col gap-8">
            <h1 className="text-2xl">PLC connected</h1>
            <div className="flex">
              <Puff
                visible={true}
                height="80"
                width="80"
                color="#4fa94d"
                ariaLabel="puff-loading"
                wrapperStyle={{}}
                wrapperClass=""
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default OutputPage;
