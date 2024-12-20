import React from "react";
import { Wand2 } from "lucide-react";
import ReverseCard from "../components/ReverseCard";
import useStore from "../store/useStore";

const ReversePage = () => {
  const {
    revFormData,
    resetResults,
    updateRevFormData,
    handleReverseGenerate,
  } = useStore();

  return (
      <div className="py-4 h-full overflow-y-scroll lg:px-20 md:px-10 container-padding">
        <div className="grid md:grid-cols-1 gap-6 lg:grid-cols-2">
          <ReverseCard
            title="UTS"
            value={revFormData.UTS}
            unit="MPa"
            onChange={(value) => updateRevFormData("UTS", value)}
          />
          <ReverseCard
            title="Elongation"
            value={revFormData.Elongation}
            unit="%"
            onChange={(value) => updateRevFormData("Elongation", value)}
          />
          <ReverseCard
            title="Conductivity"
            value={revFormData.Conductivity}
            unit="% IACS"
            onChange={(value) => updateRevFormData("Conductivity", value)}
          />

          <button
            className="bg-[#B71C1C] rounded-xl flex flex-col gap-2 items-center justify-center text-white shadow-xl"
            onClick={async () => {
              resetResults();
              console.log('sup')
              await handleReverseGenerate();
            }}
          >
            <Wand2 size={36} className="bg-transparent" />
            <p className="text-xl bg-transparent text-center">
              GENERATE
            </p>
          </button>
        </div>
      </div>
  );
};

export default ReversePage;