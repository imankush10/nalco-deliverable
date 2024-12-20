import React from "react";

const ResultCard = ({ title, value, Icon }) => {
  return (
    <div className="flex flex-col p-4 gap-4 items-center justify-center border-2 shadow-xl rounded-xl">
      <Icon
        size={36}
        className="bg-transparent cursor-pointer"
      />
      <span className="text-xl bg-transparent">{title}</span>
      <span className="text-xl text-center bg-transparent w-full focus:outline-none focus:border-b">
        {value}
      </span>
    </div>
  );
};

export default ResultCard;