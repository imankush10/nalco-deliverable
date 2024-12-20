import React from "react";
import { useLocation } from "react-router-dom";
import Footer from "./Footer";
import OutputPage from "../pages/OutputPage";
import Sidebar from "./Sidebar";

const Layout = ({ children }) => {
  const location = useLocation();

  return (
    <div className="h-screen">
      <div className="h-[calc(100vh-5rem)] flex overflow-hidden p-4 gap-4">
        <Sidebar />
        <div
          className={`flex-1 min-w-[300px] h-full overflow-y-scroll noscrollbar rounded-xl border-2 border-black py-2 ${
            location.pathname === "/plot" ? "max-w-[980px]" : "max-w-full"
          }`}
        >
          <main className="p-2">{children}</main>
        </div>
        {location.pathname !== "/plot" &&
          location.pathname !== "/dashboard" &&
          location.pathname !== "/home" &&
          location.pathname !== "/default-values" && <OutputPage />}
      </div>
      <Footer />
    </div>
  );
};

export default Layout;
