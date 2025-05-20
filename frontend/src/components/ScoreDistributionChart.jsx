import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell
} from "recharts";

// This is a new component that can be added to show the distribution of scores
const ScoreDistributionChart = ({ scores }) => {
  const data = Object.entries(scores).map(([key, value]) => ({
    name: key.charAt(0).toUpperCase() + key.slice(1),
    score: Math.round(value * 100),
    color: getScoreColor(value),
  }));

  // Sort data by score value (descending)
  data.sort((a, b) => b.score - a.score);

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold mb-3">Quality Score Distribution</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 100]} />
          <YAxis dataKey="name" type="category" width={100} />
          <Tooltip
            formatter={(value) => [`${value}%`, "Score"]}
            labelFormatter={(value) => `${value} Quality`}
          />
          <Legend />
          <Bar
            dataKey="score"
            name="Quality Score"
            radius={[0, 4, 4, 0]}
            label={{ position: "right", formatter: (value) => `${value}%` }}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="grid grid-cols-3 gap-4 mt-4">
        {data.map((item) => (
          <div key={item.name} className="flex items-center">
            <div
              className="w-4 h-4 mr-2 rounded-full"
              style={{ backgroundColor: item.color }}
            ></div>
            <div className="text-sm">
              <span className="font-medium">{item.name}:</span> {item.score}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Helper function to get color based on score value
const getScoreColor = (score) => {
  if (score >= 0.8) return "#4CAF50"; // Green
  if (score >= 0.6) return "#8BC34A"; // Light Green
  if (score >= 0.4) return "#FFC107"; // Amber
  if (score >= 0.2) return "#FF9800"; // Orange
  return "#F44336"; // Red
};

export default ScoreDistributionChart;
