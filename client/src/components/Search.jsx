import React from "react";
import { SearchIcon } from "lucide-react";

const Search = ({ className = "" }) => {
  return (
    <div
      className={`${className} flex gap-4 text-2xl border-2 border-black rounded-xl px-2 pr-10 py-1 items-center`}
    >
      <SearchIcon size={28} className="cursor-pointer" />
      <input type="text" placeholder="Search" className="focus:outline-none placeholder:text-black" />
    </div>
  );
};

export default Search;
