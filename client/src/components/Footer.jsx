import React from "react";
import Button from "./Button";
import useStore from "../store/useStore";
import { useLocation } from "react-router-dom";

const Footer = () => {
  const {
    handleGenerate,
    handleFileSelect,
    handleUpload,
    resetResults,
    selectedFile,
    updateLoading,
  } = useStore();

  const location = useLocation();

  const resultRedirectFunction = async (callback) => {
    resetResults();
    const response = await callback();
    if (!response) return;
  };

  const isPredictPageRoute = location.pathname === "/predict";

  return (
    <footer>
      <div className="flex justify-between w-3/4 mx-auto mt-4">
        {isPredictPageRoute && (
          <>
            <div className="flex gap-4 items-center">
              <Button
                value="Generate"
                type="filled"
                onClick={() => resultRedirectFunction(handleGenerate)}
              />
            </div>
            <div className="flex gap-4 items-center">
              <label
                htmlFor="fileUpload"
                className="border-2 px-4 py-2 rounded-md text-lg font-bold hover:bg-[#B71C1C] shadow-md hover:text-white transition-colors cursor-pointer text-ellipsis overflow-hidden whitespace-nowrap w-36 text-center"
                title={selectedFile?.name || "Upload file"}
              >
                {!selectedFile ? "Upload file" : selectedFile.name}
              </label>
              <Button
                onClick={() => resultRedirectFunction(handleUpload)}
                type="filled"
                value="Upload CSV"
              />
              <input
                type="file"
                hidden
                id="fileUpload"
                accept=".xlsx, .csv"
                onChange={(e) => {
                  if (e.target.files && e.target.files.length > 0)
                    handleFileSelect(e.target.files[0]);
                }}
              />
            </div>
          </>
        )}
      </div>
    </footer>
  );
};

export default Footer;