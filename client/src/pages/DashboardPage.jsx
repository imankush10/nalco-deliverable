import { useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import Button from "../components/Button";
import useStore from "../store/useStore";
import { useState } from "react";

const SemiCircleGauge = ({ value, max, title, color }) => {
  const percentage = Math.min(100, (value / max) * 100);
  const radius = 100;
  const strokeWidth = 20;
  const normalizedRadius = radius - strokeWidth / 2;
  const circumference = 2 * Math.PI * normalizedRadius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div style={{ textAlign: 'center' }}>
      <h2>{title}</h2>
      <svg
        width={radius * 2}
        height={radius}
        viewBox={`0 0 ${radius * 2} ${radius}`}
      >
        {/* Outer Circle */}
        <circle
          stroke="#e6e6e6"
          strokeWidth={strokeWidth}
          fill="none"
          cx={radius}
          cy={radius}
          r={normalizedRadius}
        />
        {/* Semi Circle Gauge */}
        <circle
          stroke={color}
          strokeWidth={strokeWidth}
          fill="none"
          cx={radius}
          cy={radius}
          r={normalizedRadius}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          transform={`rotate(-90 ${radius} ${radius})`}
          strokeLinecap="round"
        />
        {/* Inner Circle to make it look like a gauge */}
        <circle
          stroke="#e6e6e6"
          strokeWidth={strokeWidth}
          fill="none"
          cx={radius}
          cy={radius}
          r={normalizedRadius - 10}
        />
        {/* Text Value */}
        <text
          x="50%"
          y="50%"
          dy=".3em"
          textAnchor="middle"
          fill={color}
          fontSize="24"
        >
          {value?.toFixed(3) || "0.000"}
        </text>
      </svg>
    </div>
  );
};


function DashboardPage() {
  const {
    realTimeData,
    handlePLC,
    resetResults,
    updateLoading,
  } = useStore();
  
  const [isPLC, setIsPLC] = useState(false);
  const [timeRange, setTimeRange] = useState(60); // Default to the last 60 seconds

  useEffect(() => {
    let interval = null;
    if (isPLC) {
      interval = setInterval(handlePLC, 10000);
    }
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isPLC, handlePLC]);

  const handlePLCToggle = () => {
    updateLoading(true);
    if (isPLC) {
      resetResults();
    } else {
      handlePLC();
    }
    setIsPLC((prev) => !prev);
  };

  const now = realTimeData.length ? realTimeData[realTimeData.length - 1].counter : 0;
  const filteredData = realTimeData.filter((entry) => entry.counter >= now - timeRange);

  const transformedData = filteredData.map((entry) => ({
    time: entry.counter,
    uts: entry.predictions.uts,
    conductivity: entry.predictions.conductivity,
    elongation: entry.predictions.elongation,
    originalUts: entry.predictions.uts,
    originalConductivity: entry.predictions.conductivity,
    originalElongation: entry.predictions.elongation,
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="p-4 border rounded shadow bg-white">
          <p className="text-sm font-bold">Time: {label}s</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(3)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const latestData = realTimeData[realTimeData.length - 1]?.predictions || {};

  return (
    <div className="p-4 space-y-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold">Real-time Monitoring</h2>
        <Button
          value={isPLC ? "Stop PLC" : "Start PLC"}
          type={isPLC ? "filled" : ""}
          onClick={handlePLCToggle}
        />
      </div>

      <div className="rounded-lg shadow-md p-6">
        <div className="mb-4">
          <h2 className="text-xl font-bold mb-4">Material Properties Time Series</h2>
          <div className="flex justify-between items-center mb-4">
            <label className="text-sm font-medium">Time Range (seconds):</label>
            <input
              type="range"
              min="10"
              max="60"
              step="10"
              value={timeRange}
              onChange={(e) => setTimeRange(Number(e.target.value))}
              className="w-full mx-4"
            />
            <span className="text-sm">{timeRange}s</span>
          </div>
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={transformedData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="time"
                  label={{ value: "Time (s)", position: "bottom" }}
                />
                <YAxis
                  label={{
                    value: "Value",
                    angle: -90,
                    position: "left",
                  }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
                <Line
                  type="monotone"
                  dataKey="elongation"
                  stroke="#4ade80"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  name="Elongation"
                />
                <Line
                  type="monotone"
                  dataKey="conductivity"
                  stroke="#f43f5e"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  name="Conductivity"
                />
                <Line
                  type="monotone"
                  dataKey="uts"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  name="UTS"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="rounded-lg shadow-md p-6">
        <h2 className="text-xl font-bold mb-4">Current Values</h2>
        <div className="flex justify-around">
          <SemiCircleGauge
            value={latestData.elongation}
            max={100}
            title="Elongation"
            color="#4ade80"
          />
          <SemiCircleGauge
            value={latestData.conductivity}
            max={20}
            title="Conductivity"
            color="#f43f5e"
          />
          <SemiCircleGauge
            value={latestData.uts}
            max={20}
            title="UTS"
            color="#3b82f6"
          />
        </div>
      </div>
    </div>
  );
}

export default DashboardPage;
