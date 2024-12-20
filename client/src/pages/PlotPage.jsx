// import { useEffect, useState } from "react";
// import useStore from "../store/useStore";
// import { DNA } from "react-loader-spinner";

// const PlotPage = () => {
//   const {
//     generatePlot,
//     getRequiredParams,
//     plotImages,
//     plotImage,
//     plotError,
//     plotLoading,
//     clearPlot,
//     generateTemperatureDistribution,
//   } = useStore();

//   const plotTypes = [
//     "CORE_SURFACE_TEMPERATURE",
//     "TEMPERATURE_DISTRIBUTION_BAR_WIDTH",
//     "HEAT_FLUX",
//     "THREE_D_TEMPERATURE_DISTRIBUTION",
//     "POURING_TEMPERATURE_VS_BAR_TEMPERATURE_NON_LINEAR",
//     "POURING_TEMPERATURE_VS_BAR_TEMPERATURE_WHEEL_ROLL",
//     "RPM_VS_SURFACE_TEMPERATURE",
//     "TEMPERATURE_PROFILE",
//     "PLOT_POURING_TEMP_VS_BAR_TEMP",
//     "RPM_VS_SURFACE_TEMP",
//     "SENSITIVITY_OF_TEMPERATURE",
//     "OPERATING_WINDOW",
//     "MULTIPLE_PLOTTING"
//   ];

//   const [selectedPlotType, setSelectedPlotType] = useState(plotTypes[0]);
//   const [plotParams, setPlotParams] = useState({
//     length_x: 0.1,
//     length_y: 0.1,
//     time_total: 10.0,
//     dt: 0.01,
//     rpm: 1.0,
//     pouring_temp: 700,
//     pouring_temps: [600, 700, 800],
//   });
//   const [error, setError] = useState(null);

//   useEffect(() => {
//     return () => {
//       clearPlot();
//     };
//   }, [clearPlot]);

//   const handleInputChange = (e) => {
//     const { name, value } = e.target;
//     setPlotParams((prev) => ({
//       ...prev,
//       [name]:
//         name === "pouring_temps"
//           ? value.split(",").map((val) => parseFloat(val.trim()))
//           : parseFloat(value),
//     }));
//   };

//   const handlePlotTypeChange = (e) => {
//     setSelectedPlotType(e.target.value);
//     setError(null); // Clear any previous errors
//   };

//   const validateParams = (params, required) => {
//     for (const param of required) {
//       if (params[param] === undefined || params[param] === null || 
//           (typeof params[param] === 'number' && isNaN(params[param]))) {
//         throw new Error(`Missing or invalid parameter: ${param}`);
//       }
//     }
//   };

//   const handleGeneratePlot = async () => {
//     try {
//       setError(null);
//       const requiredParams = getRequiredParams(selectedPlotType);
//       const filteredParams = Object.fromEntries(
//         Object.entries(plotParams).filter(([key]) =>
//           requiredParams.includes(key)
//         )
//       );

//       // Validate parameters before making the API call
//       validateParams(filteredParams, requiredParams);

//       if (selectedPlotType === "MULTIPLE_PLOTTING") {
//         await generateTemperatureDistribution(filteredParams);
//       } else {
//         await generatePlot(selectedPlotType, filteredParams);
//       }
//     } catch (error) {
//       console.error("Plot generation error:", error);
//       setError(error.message || "Failed to generate plot. Please check your parameters.");
//     }
//   };

//   const renderInputFields = () => {
//     const requiredParams = getRequiredParams(selectedPlotType);

//     return (
//       <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//         {requiredParams.map((param) => (
//           <div key={param} className="space-y-2">
//             <label className="block text-sm font-medium text-gray-700">
//               {param.replace(/_/g, " ").toUpperCase()}
//             </label>
//             <input
//               type={param === "pouring_temps" ? "text" : "number"}
//               name={param}
//               value={plotParams[param]}
//               onChange={handleInputChange}
//               className="w-full p-2 border rounded-md shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
//               step={param === "dt" ? "0.01" : "0.1"}
//               min={0}
//               placeholder={param === "pouring_temps" ? "e.g., 600,700,800" : ""}
//               required
//             />
//           </div>
//         ))}
//       </div>
//     );
//   };

//   return (
//     <div className="max-w-4xl mx-auto p-6">
//       <div className="rounded-lg shadow-lg p-6 space-y-6">
//         <h1 className="text-2xl font-bold text-gray-900">Plot Generator</h1>

//         <div className="space-y-4">
//           <div>
//             <label className="block text-sm font-medium text-gray-700 mb-2">
//               Select Plot Type
//             </label>
//             <select
//               value={selectedPlotType}
//               onChange={handlePlotTypeChange}
//               className="w-full p-2 border rounded-md shadow-sm focus:ring-2 focus:ring-red-500 focus:border-red-500"
//             >
//               {plotTypes.map((plotType) => (
//                 <option key={plotType} value={plotType}>
//                   {plotType.replace(/_/g, " ")}
//                 </option>
//               ))}
//             </select>
//           </div>

//           {renderInputFields()}

//           {error && (
//             <div className="text-red-600 bg-red-50 p-3 rounded-md">
//               {error}
//             </div>
//           )}

//           <button
//             onClick={handleGeneratePlot}
//             disabled={plotLoading}
//             className="w-full md:w-auto px-6 py-2 bg-red-700 text-white rounded-md hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
//           >
//             {plotLoading ? "Generating..." : "Generate Plot"}
//           </button>
//         </div>

//         <div className="mt-6">
//           {plotLoading && (
//             <div className="text-center py-4 text-gray-600 flex justify-center">
//               <DNA
//                 visible={true}
//                 height="120"
//                 width="120"
//                 ariaLabel="dna-loading"
//                 wrapperStyle={{}}
//                 wrapperClass="dna-wrapper"
//               />
//             </div>
//           )}

//           {plotError && (
//             <div className="text-red-600 bg-red-50 p-3 rounded-md mt-4">
//               {plotError}
//             </div>
//           )}

//           {plotImage && (
//             <div className="mt-4 grid grid-cols-1 gap-4">
//               <div className="border rounded-lg p-4">
//                 <img
//                   src={plotImage}
//                   className="max-w-full h-auto rounded-md"
//                   alt="Generated Plot"
//                 />
//               </div>
//             </div>
//           )}

//           {plotImages && plotImages.length > 0 && (
//             <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
//               {plotImages.map((imageUrl, index) => (
//                 <div key={index} className="border rounded-lg p-4">
//                   <img
//                     src={imageUrl}
//                     alt={`Generated Plot ${index + 1}`}
//                     className="max-w-full h-auto rounded-md"
//                   />
//                 </div>
//               ))}
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default PlotPage;