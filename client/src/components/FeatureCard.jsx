import React, { useRef } from "react";
import { Edit } from "lucide-react";
import Button from "./Button";
import useStore from "../store/useStore";

const FeatureCard = ({
  title,
  value,
  onChange,
  dsbl,
}) => {
  const inputRef = useRef(null);

  const handleChange = (e) => {
    const newValue = Number(e.target.value);
    if (!isNaN(newValue)) {
      onChange(newValue);
    }
  };

  const handleEditClick = () => {
    inputRef.current?.focus();
  };

  return (
    <div className="flex flex-col justify-between border-2 rounded-xl py-1 px-4 hover:shadow-lg transition-shadow gap-2">
      <button
        onClick={handleEditClick}
        className="p-2 hover:bg-gray-100 rounded-full transition-colors self-end"
      ></button>
      <div >
        <p className="text-xl text-center text-gray-600 mb-2">{dsbl==false && title}</p>
        <input
          type="number"
          value={value}
          onChange={handleChange}
          className="text-2xl text-center bg-transparent w-full focus:outline-none focus:border-b text-red-700 font-bold"
          ref={inputRef}
        />
      </div>
    </div>
  );
};

export default FeatureCard;
