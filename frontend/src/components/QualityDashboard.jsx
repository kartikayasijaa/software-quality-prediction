import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { SunIcon, MoonIcon } from "./Icons";
import ScoreDistributionChart from "./ScoreDistributionChart";
import ScoreExplanation from "./ScoreExplanation";

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884d8"];
const RADAR_COLORS = {
  maintainability: "#0088FE",
  reliability: "#00C49F",
  scalability: "#FFBB28",
  security: "#FF8042",
  efficiency: "#8884d8",
};

// Sample data - this would be dynamically fetched from your backend
const sampleRepositoryData = {
  name: "Sample Repository",
  overall_score: 0.78,
  scores: {
    maintainability: 0.82,
    reliability: 0.75,
    scalability: 0.68,
    security: 0.9,
    efficiency: 0.72,
  },
  history: [
    {
      date: "2025-01",
      maintainability: 0.7,
      reliability: 0.65,
      scalability: 0.6,
      security: 0.8,
      efficiency: 0.68,
    },
    {
      date: "2025-02",
      maintainability: 0.72,
      reliability: 0.68,
      scalability: 0.62,
      security: 0.82,
      efficiency: 0.69,
    },
    {
      date: "2025-03",
      maintainability: 0.75,
      reliability: 0.7,
      scalability: 0.64,
      security: 0.85,
      efficiency: 0.7,
    },
    {
      date: "2025-04",
      maintainability: 0.78,
      reliability: 0.72,
      scalability: 0.66,
      security: 0.87,
      efficiency: 0.71,
    },
    {
      date: "2025-05",
      maintainability: 0.82,
      reliability: 0.75,
      scalability: 0.68,
      security: 0.9,
      efficiency: 0.72,
    },
  ],
  key_metrics: {
    maintainability: [
      { name: "Code Complexity", value: 0.65 },
      { name: "Documentation", value: 0.9 },
      { name: "Technical Debt", value: 0.78 },
      { name: "Code Structure", value: 0.85 },
    ],
    reliability: [
      { name: "Test Coverage", value: 0.72 },
      { name: "Exception Handling", value: 0.68 },
      { name: "Error Rate", value: 0.8 },
      { name: "Stability", value: 0.75 },
    ],
    scalability: [
      { name: "Performance", value: 0.7 },
      { name: "Resource Usage", value: 0.65 },
      { name: "Architecture", value: 0.68 },
      { name: "Load Handling", value: 0.62 },
    ],
    security: [
      { name: "Vulnerability Score", value: 0.92 },
      { name: "Secure Practices", value: 0.88 },
      { name: "Dependency Security", value: 0.85 },
      { name: "Input Validation", value: 0.9 },
    ],
    efficiency: [
      { name: "Algorithm Complexity", value: 0.75 },
      { name: "Memory Usage", value: 0.7 },
      { name: "Execution Time", value: 0.72 },
      { name: "Resource Efficiency", value: 0.68 },
    ],
  },
  recommendations: [
    {
      category: "scalability",
      description: "Reduce circular dependencies to improve scalability",
      severity: "high",
      impact: 0.15,
    },
    {
      category: "reliability",
      description: "Increase test coverage for critical components",
      severity: "medium",
      impact: 0.1,
    },
    {
      category: "efficiency",
      description: "Optimize database queries in UserService",
      severity: "medium",
      impact: 0.08,
    },
    {
      category: "maintainability",
      description: "Refactor complex methods in AuthController",
      severity: "high",
      impact: 0.12,
    },
    {
      category: "security",
      description: "Update vulnerable dependencies",
      severity: "critical",
      impact: 0.2,
    },
  ],
  feature_importance: {
    maintainability: [
      { name: "Code Complexity", value: 28 },
      { name: "Documentation", value: 22 },
      { name: "Technical Debt", value: 20 },
      { name: "Circular Dependencies", value: 15 },
      { name: "Code Structure", value: 15 },
    ],
    reliability: [
      { name: "Test Coverage", value: 30 },
      { name: "Exception Handling", value: 25 },
      { name: "Error Handling", value: 20 },
      { name: "Code Stability", value: 15 },
      { name: "Test Quality", value: 10 },
    ],
    scalability: [
      { name: "Architecture", value: 35 },
      { name: "Dependency Structure", value: 25 },
      { name: "Resource Usage", value: 20 },
      { name: "Performance", value: 20 },
    ],
    security: [
      { name: "Dependency Security", value: 30 },
      { name: "Input Validation", value: 25 },
      { name: "Authentication", value: 20 },
      { name: "Data Protection", value: 15 },
      { name: "Error Exposure", value: 10 },
    ],
    efficiency: [
      { name: "Algorithm Complexity", value: 35 },
      { name: "Memory Usage", value: 25 },
      { name: "Execution Time", value: 25 },
      { name: "Resource Efficiency", value: 15 },
    ],
  },
};

const QualityDashboard = () => {
  const [repositoryData, setRepositoryData] = useState(sampleRepositoryData);
  const [selectedMetric, setSelectedMetric] = useState("maintainability");
  const [loading, setLoading] = useState(false);
  const [repoPath, setRepoPath] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [darkMode, setDarkMode] = useState(true);
  const fileInputRef = useRef(null);

  useEffect(() => {
    // Set initial dark mode
    if (darkMode) {
      document.body.classList.add("dark-mode");
      document.body.style.backgroundColor = "#1a1a1a";
      document.body.style.color = "#e2e8f0";
    }
  }, []);

  const toggleTheme = () => {
    setDarkMode(!darkMode);
    // Update body class for global styles if needed
    if (darkMode) {
      document.body.classList.remove("dark-mode");
      document.body.style.backgroundColor = "#f9fafb";
      document.body.style.color = "#111827";
    } else {
      document.body.classList.add("dark-mode");
      document.body.style.backgroundColor = "#1a1a1a";
      document.body.style.color = "#e2e8f0";
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file || !file.name.endsWith(".zip")) {
      setError("Please select a zip file");
      return;
    }

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("/api/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      if (response.data.success) {
        setRepoPath(response.data.repo_path);
        // Automatically analyze after successful upload
        analyzeRepository(response.data.repo_path);
      }
    } catch (err) {
      setError(err.response?.data?.error || "Failed to upload repository");
    } finally {
      setUploading(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const analyzeRepository = async (path = null) => {
    const repoToAnalyze = path || repoPath;

    if (!repoToAnalyze) {
      setError("Please enter a repository path or upload a zip file");
      return;
    }

    setAnalyzing(true);
    setError(null);

    try {
      const response = await axios.post("/api/analyze", {
        repo_path: repoToAnalyze,
      });
      setRepositoryData(response.data);
    } catch (err) {
      setError(err.response?.data?.error || "Failed to analyze repository");
    } finally {
      setAnalyzing(false);
    }
  };

  // Convert scores object to array for radar chart
  const radarData = Object.entries(repositoryData.scores).map(
    ([key, value]) => ({
      subject: key.charAt(0).toUpperCase() + key.slice(1),
      value: value,
      fullMark: 1,
    })
  );

  // Format recommendations by severity
  const getSeverityColor = (severity) => {
    switch (severity) {
      case "critical":
        return "#d32f2f";
      case "high":
        return "#f57c00";
      case "medium":
        return "#fbc02d";
      case "low":
        return "#388e3c";
      default:
        return "#757575";
    }
  };

  return (
    <div
      className={`flex flex-col w-full max-w-6xl mx-auto p-4 ${
        darkMode ? "bg-dark-bg text-dark-text" : "bg-gray-50 text-gray-800"
      }`}
    >
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Software Quality Prediction</h1>
        <button
          onClick={toggleTheme}
          className={`p-2 rounded-full ${
            darkMode
              ? "bg-gray-700 text-yellow-300 hover:bg-gray-600"
              : "bg-blue-100 text-blue-800 hover:bg-blue-200"
          }`}
          aria-label={darkMode ? "Switch to light mode" : "Switch to dark mode"}
        >
          {darkMode ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>

      {/* Repository Input */}
      <div
        className={`${
          darkMode ? "bg-dark-card border border-dark-border" : "bg-white"
        } p-6 rounded-lg shadow-md mb-6`}
      >
        <h2 className="text-xl font-bold mb-4">Analyze Repository</h2>
        <div className="mb-4">
          <div className="flex flex-col md:flex-row gap-4 mb-4">
            <input
              type="text"
              placeholder="Enter repository path"
              className={`flex-1 p-2 border rounded ${
                darkMode
                  ? "border-dark-border bg-dark-bg text-dark-text"
                  : "border-gray-300"
              }`}
              value={repoPath}
              onChange={(e) => setRepoPath(e.target.value)}
            />
            <button
              className={`px-4 py-2 rounded bg-blue-600 text-white ${
                analyzing || uploading
                  ? "opacity-50 cursor-not-allowed"
                  : "hover:bg-blue-700"
              }`}
              onClick={() => analyzeRepository()}
              disabled={analyzing || uploading}
            >
              {analyzing ? "Analyzing..." : "Analyze"}
            </button>
          </div>

          <div className="mt-4">
            <p
              className={`${
                darkMode ? "text-dark-text" : "text-gray-700"
              } mb-2`}
            >
              Or upload a repository ZIP file:
            </p>
            <div className="flex flex-col md:flex-row gap-4">
              <input
                type="file"
                ref={fileInputRef}
                accept=".zip"
                className={`flex-1 p-2 border rounded ${
                  darkMode
                    ? "border-dark-border bg-dark-bg text-dark-text"
                    : "border-gray-300"
                }`}
                onChange={handleFileUpload}
                disabled={analyzing || uploading}
              />
              <button
                className={`px-4 py-2 rounded bg-green-600 text-white ${
                  analyzing || uploading
                    ? "opacity-50 cursor-not-allowed"
                    : "hover:bg-green-700"
                }`}
                onClick={() => fileInputRef.current?.click()}
                disabled={analyzing || uploading}
              >
                {uploading ? "Uploading..." : "Upload & Analyze"}
              </button>
            </div>
          </div>
        </div>
        {error && <p className="text-red-500 mt-2">{error}</p>}
      </div>

      {loading || analyzing || uploading ? (
        <div
          className={`flex justify-center items-center h-64 p-6 rounded-lg shadow-md ${
            darkMode ? "bg-dark-card border border-dark-border" : "bg-white"
          }`}
        >
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-xl">
              {uploading
                ? "Uploading repository..."
                : analyzing
                ? "Analyzing repository..."
                : "Loading data..."}
            </p>
          </div>
        </div>
      ) : (
        <>
          {/* Repository Name */}
          <div
            className={`p-6 rounded-lg shadow-md mb-6 ${
              darkMode ? "bg-dark-card border border-dark-border" : "bg-white"
            }`}
          >
            <h2 className="text-2xl font-bold">{repositoryData.name}</h2>
          </div>

          {/* Overall Score */}
          <div
            className={`p-6 rounded-lg shadow-md mb-6 ${
              darkMode ? "bg-dark-card border border-dark-border" : "bg-white"
            }`}
          >
            <h2 className="text-xl font-bold mb-4">Overall Quality Score</h2>
            <div className="flex flex-col md:flex-row items-center">
              <div className="relative w-40 h-40 mb-6 md:mb-0">
                <svg viewBox="0 0 100 100" className="w-full h-full">
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke={darkMode ? "#3f3f3f" : "#e6e6e6"}
                    strokeWidth="10"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke={`hsl(${
                      repositoryData.overall_score * 120
                    }, 80%, 50%)`}
                    strokeWidth="10"
                    strokeDasharray={`${
                      repositoryData.overall_score * 283
                    } 283`}
                    strokeLinecap="round"
                    transform="rotate(-90 50 50)"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-3xl font-bold">
                    {Math.round(repositoryData.overall_score * 100)}
                  </span>
                </div>
              </div>
              <div className="md:ml-6 flex-1">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(repositoryData.scores).map(([key, value]) => (
                    <div key={key} className="flex flex-col">
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium capitalize">
                          {key}
                        </span>
                        <span className="text-sm font-medium">
                          {Math.round(value * 100)}
                        </span>
                      </div>
                      <div
                        className={`w-full rounded-full h-2 ${
                          darkMode ? "bg-dark-border" : "bg-gray-200"
                        }`}
                      >
                        <div
                          className="h-2 rounded-full"
                          style={{
                            width: `${value * 100}%`,
                            backgroundColor: RADAR_COLORS[key],
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Add the Score Distribution Chart here */}
                <div className="mt-6">
                  <ScoreDistributionChart scores={repositoryData.scores} />
                </div>
              </div>
            </div>
          </div>

          {/* Quality Dimensions Overview */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* Radar Chart */}
            <div
              className={`p-6 rounded-lg shadow-md ${
                darkMode ? "bg-dark-card border border-dark-border" : "bg-white"
              }`}
            >
              <h2 className="text-xl font-bold mb-4">Quality Dimensions</h2>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart outerRadius={90} data={radarData}>
                  <PolarGrid stroke={darkMode ? "#3f3f3f" : "#e6e6e6"} />
                  <PolarAngleAxis
                    dataKey="subject"
                    stroke={darkMode ? "#e2e8f0" : "#000000"}
                  />
                  <PolarRadiusAxis
                    domain={[0, 1]}
                    stroke={darkMode ? "#e2e8f0" : "#000000"}
                  />
                  <Radar
                    name="Quality"
                    dataKey="value"
                    stroke="#8884d8"
                    fill="#8884d8"
                    fillOpacity={0.6}
                  />
                  <Tooltip
                    contentStyle={
                      darkMode
                        ? {
                            backgroundColor: "#2d2d2d",
                            borderColor: "#3f3f3f",
                            color: "#e2e8f0",
                          }
                        : {
                            backgroundColor: "white",
                            borderColor: "#cccccc",
                            color: "black",
                          }
                    }
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* History Chart */}
            <div
              className={`p-6 rounded-lg shadow-md ${
                darkMode ? "bg-dark-card border border-dark-border" : "bg-white"
              }`}
            >
              <h2 className="text-xl font-bold mb-4">Quality Trends</h2>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={repositoryData.history}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke={darkMode ? "#3f3f3f" : "#e6e6e6"}
                  />
                  <XAxis
                    dataKey="date"
                    stroke={darkMode ? "#e2e8f0" : "#000000"}
                  />
                  <YAxis
                    domain={[0, 1]}
                    stroke={darkMode ? "#e2e8f0" : "#000000"}
                  />
                  <Tooltip
                    contentStyle={
                      darkMode
                        ? {
                            backgroundColor: "#2d2d2d",
                            borderColor: "#3f3f3f",
                            color: "#e2e8f0",
                          }
                        : {
                            backgroundColor: "white",
                            borderColor: "#cccccc",
                            color: "black",
                          }
                    }
                  />
                  <Legend />
                  {Object.keys(repositoryData.scores).map((key, index) => (
                    <Line
                      key={key}
                      type="monotone"
                      dataKey={key}
                      stroke={COLORS[index % COLORS.length]}
                      activeDot={{ r: 8 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Detailed Analysis Section */}
          <div
            className={`p-6 rounded-lg shadow-md mb-6 ${
              darkMode ? "bg-dark-card border border-dark-border" : "bg-white"
            }`}
          >
            <h2 className="text-xl font-bold mb-4">Detailed Analysis</h2>
            <div className="flex flex-wrap mb-4">
              {Object.keys(repositoryData.scores).map((metric) => (
                <button
                  key={metric}
                  className={`px-4 py-2 mr-2 mb-2 rounded-lg ${
                    selectedMetric === metric
                      ? "bg-blue-600 text-white"
                      : darkMode
                      ? "bg-dark-border text-dark-text hover:bg-gray-700"
                      : "bg-gray-200 text-gray-700 hover:bg-gray-300"
                  }`}
                  onClick={() => setSelectedMetric(metric)}
                >
                  {metric.charAt(0).toUpperCase() + metric.slice(1)}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Key Metrics */}
              <div>
                <h3 className="text-lg font-semibold mb-3 capitalize">
                  {selectedMetric} Metrics
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={repositoryData.key_metrics[selectedMetric]}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke={darkMode ? "#3f3f3f" : "#e6e6e6"}
                    />
                    <XAxis
                      dataKey="name"
                      stroke={darkMode ? "#e2e8f0" : "#000000"}
                    />
                    <YAxis
                      domain={[0, 1]}
                      stroke={darkMode ? "#e2e8f0" : "#000000"}
                    />
                    <Tooltip
                      contentStyle={
                        darkMode
                          ? {
                              backgroundColor: "#2d2d2d",
                              borderColor: "#3f3f3f",
                              color: "#e2e8f0",
                            }
                          : {
                              backgroundColor: "white",
                              borderColor: "#cccccc",
                              color: "black",
                            }
                      }
                    />
                    <Bar dataKey="value" fill={RADAR_COLORS[selectedMetric]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Feature Importance */}
              {repositoryData.feature_importance[selectedMetric] && (
                <div>
                  <h3 className="text-lg font-semibold mb-3">
                    Feature Importance
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={repositoryData.feature_importance[selectedMetric]}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => {
                          const value = repositoryData.feature_importance[
                            selectedMetric
                          ].reduce((sum, entry) => sum + entry.value, 0);
                          return `${name} ${Math.round(percent * 100)}%`;
                        }}
                      >
                        {repositoryData.feature_importance[selectedMetric].map(
                          (entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              fill={COLORS[index % COLORS.length]}
                            />
                          )
                        )}
                      </Pie>
                      <Tooltip
                        contentStyle={
                          darkMode
                            ? {
                                backgroundColor: "#2d2d2d",
                                borderColor: "#3f3f3f",
                                color: "#e2e8f0",
                              }
                            : {
                                backgroundColor: "white",
                                borderColor: "#cccccc",
                                color: "black",
                              }
                        }
                      />
                    </PieChart>
                  </ResponsiveContainer>

                  {/* Add Score Explanation component here */}
                  {repositoryData.features && (
                    <ScoreExplanation
                      dimension={selectedMetric}
                      score={repositoryData.scores[selectedMetric]}
                      features={repositoryData.features}
                    />
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Recommendations */}
          <div
            className={`p-6 rounded-lg shadow-md ${
              darkMode ? "bg-dark-card border border-dark-border" : "bg-white"
            }`}
          >
            <h2 className="text-xl font-bold mb-4">Recommendations</h2>
            <div className="space-y-4">
              {repositoryData.recommendations
                .sort((a, b) => b.impact - a.impact)
                .map((rec, index) => (
                  <div
                    key={index}
                    className={`border-l-4 p-4 flex flex-col md:flex-row items-start md:items-center ${
                      darkMode ? "bg-dark-bg" : "bg-gray-50"
                    }`}
                    style={{ borderColor: getSeverityColor(rec.severity) }}
                  >
                    <div className="flex-1">
                      <div className="flex items-center flex-wrap">
                        <span
                          className="inline-block px-2 py-1 text-xs rounded text-white mr-2 mb-2 md:mb-0 capitalize"
                          style={{
                            backgroundColor: getSeverityColor(rec.severity),
                          }}
                        >
                          {rec.severity}
                        </span>
                        <span className="text-sm font-medium capitalize mr-2">
                          {rec.category}
                        </span>
                      </div>
                      <p className="mt-1">{rec.description}</p>
                    </div>
                    <div className="text-right mt-2 md:mt-0">
                      <div
                        className={`text-sm ${
                          darkMode ? "text-gray-400" : "text-gray-500"
                        }`}
                      >
                        Impact
                      </div>
                      <div className="text-lg font-bold">
                        {Math.round(rec.impact * 100)}%
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </>
      )}

      {/* Footer */}
      <div className="mt-8 text-center text-gray-500 text-sm">
        <p>Software Quality Prediction Tool &copy; 2025</p>
      </div>
    </div>
  );
};

export default QualityDashboard;
