import React, { useRef } from "react";
import { Edit } from "lucide-react";

const ReverseCard = ({
  title,
  value,
  unit,
  onChange,
}) => {
  const inputRef = useRef(null);

  const handleEditClick = () => {
    inputRef.current?.focus();
  };

  return (
    <div className="flex flex-col p-4 py-6 gap-2 items-center justify-center text-center border-2 shadow-xl rounded-xl">
      <Edit
        size={24}
        className="self-end bg-transparent cursor-pointer"
        onClick={handleEditClick}
      />
      <span className="text-xl bg-transparent">{title.toUpperCase()}</span>
      <div className="flex">
        <input
          type="number"
          placeholder={title}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="text-3xl text-center bg-transparent w-full focus:outline-none focus:border-b text-red-700"
          ref={inputRef}
        />
        <span className="text-xl">{unit}</span>
      </div>
    </div>
  );
};

export default ReverseCard;