import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const HomePage = () => {
  const [data, setData] = useState([]);

  useEffect(() => {
    const sampleData = [
      { name: "Jan", UTS: 50, Conductivity: 100, Elongation: 80 },
      { name: "Feb", UTS: 60, Conductivity: 110, Elongation: 85 },
      { name: "Mar", UTS: 70, Conductivity: 120, Elongation: 90 },
      { name: "Apr", UTS: 80, Conductivity: 130, Elongation: 95 },
      { name: "May", UTS: 90, Conductivity: 140, Elongation: 100 },
      { name: "Jun", UTS: 100, Conductivity: 150, Elongation: 105 },
      { name: "Jul", UTS: 110, Conductivity: 160, Elongation: 110 },
      { name: "Aug", UTS: 120, Conductivity: 170, Elongation: 115 },
      { name: "Sep", UTS: 130, Conductivity: 180, Elongation: 120 },
      { name: "Oct", UTS: 140, Conductivity: 190, Elongation: 125 },
      { name: "Nov", UTS: 150, Conductivity: 200, Elongation: 130 },
      { name: "Dec", UTS: 160, Conductivity: 210, Elongation: 135 },
    ];
    setData(sampleData);
  }, []);

  return (
    <div className="w-full max-w-6xl mx-auto p-6">
      <header className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Welcome, Ananya</h1>
        <div className="flex items-center space-x-4">
          <div className="w-10 h-10 bg-gray-200 rounded-full cursor-pointer flex items-center justify-center text-2xl">
            A
          </div>
        </div>
      </header>

      <main className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="shadow-md rounded-lg p-6">
          <h2 className="text-lg font-medium mb-2">UTS</h2>
          <div className="text-4xl font-bold">8 MP</div>
          <div className="text-green-500">+18% +2.8k this week</div>
        </div>
        <div className="shadow-md rounded-lg p-6">
          <h2 className="text-lg font-medium mb-2">Elongation</h2>
          <div className="text-4xl font-bold">18</div>
          <div className="text-green-500">+10% +7.8k this week</div>
        </div>
        <div className="shadow-md rounded-lg p-6">
          <h2 className="text-lg font-medium mb-2">Conductivity</h2>
          <div className="text-4xl font-bold">61</div>
          <div className="text-red-500">+18% +1.2k this week</div>
        </div>
      </main>

      <div className="shadow-md rounded-lg p-6">
        <h2 className="text-lg font-medium mb-4">Daily Trends</h2>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={data}>
            <XAxis dataKey="name" />
            <YAxis />
            <CartesianGrid strokeDasharray="3 3" />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="UTS" stroke="#ffa500" />
            <Line type="monotone" dataKey="Conductivity" stroke="#4287f5" />
            <Line type="monotone" dataKey="Elongation" stroke="#e53e3e" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default HomePage;
