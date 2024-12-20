import React, { useState, useEffect } from "react";

const PRIORITY_MAP = {
  EMUL_OIL_L_TEMP_PV_VAL0: 2,
  STAND_OIL_L_TEMP_PV_REAL_VAL0: 2,
  GEAR_OIL_L_TEMP_PV_REAL_VAL0: 2,
  EMUL_OIL_L_PR_VAL0: 2,
  ROD_DIA_MM_VAL0: 2,
  CAST_WHEEL_RPM_VAL0: 1,
  BAR_TEMP_VAL0: 1,
  QUENCH_CW_FLOW_ENTRY_VAL0: 1,
  GEAR_OIL_L_PR_VAL0: 1,
  STANDS_OIL_L_PR_VAL0: 1,
  TUNDISH_TEMP_VAL0: 3,
  RM_MOTOR_COOL_WATER__VAL0: 2,
  ROLL_MILL_AMPS_VAL0: 1,
  RM_COOL_WATER_FLOW_VAL0: 1,
  EMULSION_LEVEL_ANALO_VAL0: 2,
  furnace_temp: 3,
  "%SI": 3,
  "%FE": 3,
  "%TI": 3,
  "%V": 3,
  "%MN": 3,
  OTHIMP: 3,
  "%AL": 3,
  QUENCH_CW_FLOW_EXIT_VAL0: 1,
};

const ANIMATION_DELAY = 100;

const AnimatedProcessInstructions = ({
  revResult,
  revValidationErrors,
  labelMap,
}) => {
  const [items, setItems] = useState([]);
  const [sorting, setSorting] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [pass, setPass] = useState(0);
  const [noSwapCounter, setNoSwapCounter] = useState(0);
  const [swappedInPass, setSwappedInPass] = useState(false);

  useEffect(() => {
    if (revResult) {
      const initialItems = Object.entries(revResult).map(([key, value]) => ({
        key,
        value,
        priority: PRIORITY_MAP[key] || Number.MAX_SAFE_INTEGER,
        highlighted: false,
      }));
      setItems(initialItems);
      setSorting(true);
      setCurrentIndex(0);
      setPass(0);
      setNoSwapCounter(0);
      setSwappedInPass(false);
    }
  }, [revResult]);

  useEffect(() => {
    if (!sorting || !items.length) return;

    const timer = setTimeout(() => {
      const itemsCopy = [...items];
      const n = itemsCopy.length;
      let swapped = false;

      // Clear previous highlights
      itemsCopy.forEach((item) => (item.highlighted = false));

      // Current pair to compare
      const i = currentIndex % (n - 1);

      // Highlight current pair
      itemsCopy[i].highlighted = true;
      itemsCopy[i + 1].highlighted = true;

      // Compare and swap if needed
      if (itemsCopy[i].priority > itemsCopy[i + 1].priority) {
        [itemsCopy[i], itemsCopy[i + 1]] = [itemsCopy[i + 1], itemsCopy[i]];
        swapped = true;
        setSwappedInPass(true);
      }

      setItems(itemsCopy);

      // Move to the next pair
      if (currentIndex + 1 < n - 1) {
        setCurrentIndex(currentIndex + 1);
      } else {
        // End of pass
        if (!swappedInPass) {
          setNoSwapCounter(noSwapCounter + 1);
        } else {
          setNoSwapCounter(0);
        }
        setSwappedInPass(false);

        // Move to the next pass
        setPass(pass + 1);
        setCurrentIndex(0);

        // Stop sorting if no swaps were made for 2 consecutive passes
        if (noSwapCounter + 1 >= 2) {
          setSorting(false);
        }
      }
    }, ANIMATION_DELAY);

    return () => clearTimeout(timer);
  }, [sorting, currentIndex, items, pass, noSwapCounter, swappedInPass]);

  if (!revResult || revValidationErrors.length) {
    return null;
  }

  return (
    <div className="mt-8">
      <h3 className="text-2xl font-bold text-[#B71C1C] mb-6">
        Process Instructions {sorting && "(Sorting...)"}
      </h3>

      <div className="space-y-4">
        {items.map((item, index) => (
          <div
            key={`${item.key}-${index}`}
            className={`border-2 rounded-lg p-4 flex items-center transition-all duration-300
              ${
                sorting && item.highlighted
                  ? "bg-yellow-200 border-yellow-400"
                  : ""
              }`}
          >
            <div className="bg-[#B71C1C] text-white w-8 h-8 rounded flex items-center justify-center mr-4">
              {index + 1}
            </div>
            <p className="text-xl">
              {`Set ${labelMap[item.key]}`}:{" "}
              <span className="font-bold">{`${item.value.toFixed(6)}`}</span>
            </p>
            <span
              className={`ml-12 font-bold text-2xl bg-transparent cursor-pointer ${
                item.priority == 3 && "text-red-500"
              } ${item.priority == 2 && "text-yellow-500"} ${
                item.priority == 1 && "text-green-500"
              }`}
              onClick={() => {
                item.priority == 1
                  ? alert("Urgent Alert Sent")
                  : item.priority == 2
                  ? alert("Information Alert Sent")
                  : alert("Alert not required");
              }}
            >
              {item.priority}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AnimatedProcessInstructions;
