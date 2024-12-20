import React from "react";

const Button = ({ type = "", value, onClick }) => {
  return (
    <button
      onClick={onClick}
      className={
        type === "filled"
          ? "bg-[#B71C1C] text-white px-4 py-2 rounded-md text-lg font-bold shadow-md"
          : "border-2 px-4 py-2 rounded-md text-lg font-bold hover:bg-[#B71C1C] hover:text-white transition-colors shadow-md"
      }
    >
      {value}
    </button>
  );
};

export default Button;
