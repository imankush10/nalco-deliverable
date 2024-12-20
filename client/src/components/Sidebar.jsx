import React from "react";
import { NavLink, useLocation } from "react-router-dom";
import { BarChart, RefreshCw, GitBranch, Play, Zap, Vault } from "lucide-react";
import useStore from "../store/useStore";
import { useState } from "react";

const Sidebar = () => {
  const { resetResults } = useStore();
  const location = useLocation();
  const [port, setPort] = useState();

  return (
    <nav>
      <div className="bg-red-800 min-w-[200px] h-[calc(100vh-17rem)] py-10 text-white px-4 flex flex-col max-w-[350px] text-2xl gap-4 rounded-xl mx-8 my-2">
        <NavLink
          to="/home"
          className={({ isActive }) =>
            isActive
              ? "flex gap-4 items-center p-2 rounded-md bg-red-700"
              : "flex gap-4 items-center p-2 rounded-md bg-red-800"
          }
          onClick={() => resetResults()}
        >
          <Zap size={20} className="bg-transparent" />
          <span className="bg-transparent">Home</span>
        </NavLink>
        <NavLink
          to="/predict"
          className={({ isActive }) =>
            isActive
              ? "flex gap-4 items-center p-2 rounded-md bg-red-700"
              : "flex gap-4 items-center p-2 rounded-md bg-red-800"
          }
          onClick={() => resetResults()}
        >
          <BarChart size={20} className="bg-transparent" />
          <span className="bg-transparent">Predict</span>
        </NavLink>
        <NavLink
          to="/reverse"
          className={({ isActive }) =>
            isActive
              ? "flex gap-4 items-center p-2 rounded-md bg-red-700"
              : "flex gap-4 items-center p-2 rounded-md bg-red-800"
          }
          onClick={() => resetResults()}
        >
          <RefreshCw size={20} className="bg-transparent" />
          <span className="bg-transparent">Reverse</span>
        </NavLink>
        <NavLink
          to="/dashboard"
          className={({ isActive }) =>
            isActive
              ? "flex gap-4 items-center p-2 rounded-md bg-red-700"
              : "flex gap-4 items-center p-2 rounded-md bg-red-800"
          }
          onClick={() => resetResults()}
        >
          <Play size={20} className="bg-transparent" />
          <span className="bg-transparent">PLC Trends</span>
        </NavLink>
        <NavLink
          to="/default-values"
          className={({ isActive }) =>
            isActive
              ? "flex gap-4 items-center p-2 rounded-md bg-red-700"
              : "flex gap-4 items-center p-2 rounded-md bg-red-800"
          }
          onClick={() => resetResults()}
        >
          <Vault size={20} className="bg-transparent" />
          <span className="bg-transparent">Set default</span>
        </NavLink>
      </div>
      {/* {location.pathname == "/home" && (
        <div className="h-48">
          <div className="flex flex-col gap-2 border-2 rounded-md p-1 m-auto w-52 py-4 px-2">
            <label className="text-xl">Enter Port</label>
            <input
              type="number"
              placeholder="1443"
              className="text-2xl border-b"
            />
          </div>
          <NavLink
          to="/predict"
            className="bg-red-700 px-3 rounded-lg text-2xl text-white ml-8"
            onClick={() => {
              alert("Port started", port);
            }}
          >
            Start
          </NavLink>
        </div>
      )} */}

      {/* <NavLink
        to="/plot"
        className={({ isActive }) =>
          isActive
            ? "flex gap-2 mx-8 rounded-xl items-center text-xl my-4 fire-button w-fit py-4 px-2 bg-neutral-800 text-white"
            : "flex gap-2 mx-8 rounded-xl items-center text-xl my-4 fire-button w-fit py-4 px-2"
        }
        onClick={() => resetResults()}
      >
        <Zap size={20} className="bg-transparent" />
        <span className="bg-transparent font-bold">Hyper-Optimize</span>
      </NavLink> */}
    </nav>
  );
};

export default Sidebar;
