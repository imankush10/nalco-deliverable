import React, { useState } from "react";
import useStore from "../store/useStore";
import Button from "../components/Button";

const TrainPage = () => {
  const {
    handleTrainingFileUpload,
    handleGenerate,
    resetResults,
    updateFormData,
    handleFileSelect,
    handleSplitData,
    handleTrainModel,
    handleTuneModel,
    handleSaveModel,
  } = useStore();

  const [step, setStep] = useState(1);
  const [trainRatio, setTrainRatio] = useState(0.7);
  const [targetColumn, setTargetColumn] = useState("Elongation");
  const [optimizeMetric, setOptimizeMetric] = useState("true");

  const handleNext = async () => {
    if (step === 1) {
      const uploaded = await handleTrainingFileUpload();
      if (!uploaded) return;
    } else if (step == 2) {
      const res = await handleSplitData();
      if(!res) return;
    } else if (step == 3) {
      const res = handleTrainModel();
      if(!res) return;
    } else if (step === 4) {
      await handleTuneModel();
      if(!res) return;
    } else if (step == 5) {
      await handleSaveModel();
      if(!res) return;
    }
    setStep(step + 1);
  };

  const renderStep = () => {
    switch (step) {
      case 1:
        return (
          <div className="flex flex-col gap-4">
            <h2 className="text-2xl">Upload CSV File</h2>
            <input
              type="file"
              accept=".csv, .xlsx"
              onChange={(e)=>{
                console.log(e.target.files[0]);
                handleFileSelect(e.target.files[0])
              }}
            />
            <Button onClick={handleNext} value="Next" type="filled" />
          </div>
        );
      case 2:
        return (
          <div className="flex flex-col gap-4">
            <h2 className="text-2xl">Enter Train Ratio</h2>
            <input
              type="number"
              step="0.01"
              value={trainRatio}
              className="text-2xl border-b"
              onChange={(e) => setTrainRatio(parseFloat(e.target.value))}
              placeholder="0.7"
            />
            <div className="flex gap-4 mt-4">
              <Button
                value="Save Ratio"
                onClick={() => updateFormData("train_ratio", trainRatio)}
              />
              <Button type="filled" onClick={handleNext} value="Next" />
            </div>
          </div>
        );
      case 3:
        return (
          <div className="flex flex-col gap-4">
            <h2 className="text-2xl">Enter Target Column</h2>
            <input
              type="text"
              className="text-xl border-b"
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              placeholder="Elongation"
            />

            <div className="flex gap-8 mt-4">
              <Button
                value="Save Target"
                onClick={() => updateFormData("target_column", targetColumn)}
              />
              <Button onClick={handleNext} value="Next" type="filled" />
            </div>
          </div>
        );
      case 4:
        return (
          <div className="flex flex-col gap-8">
            <h2 className="text-2xl">Enter Hypertune Metric</h2>
            <input
              type="text"
              value={optimizeMetric}
              onChange={(e) => setOptimizeMetric(e.target.value)}
              placeholder="yes/no"
              className="border-b text-xl"
            />
            <Button type="filled" onClick={handleNext} value="Save and Train" />
          </div>
        );
      case 5:
        return (
          <div className="flex flex-col gap-4">
            <h2 className="text-2xl">Model Created and Saved</h2>
            <p className="">Check the console or store for detailed results.</p>
            <Button
              onClick={() => {
                setStep(1);
                resetResults();
              }}
              value="Reset"
              type="filled"
            />
          </div>
        );
      default:
        return null;
    }
  };
  
  return (
    <div className="h-full p-2">
      <div className="py-4 overflow-y-scroll px-4 lg:px-14 container-padding">
        {renderStep()}
      </div>
    </div>
  );
};

export default TrainPage;
