import React from "react";

const ScoreExplanation = ({ dimension, score, features }) => {
  // Determine primary factors that influenced the score
  const getInfluencingFactors = (dimension, features) => {
    switch (dimension) {
      case "maintainability":
        return [
          {
            name: "Code Complexity",
            value: features.complexity,
            impact: features.complexity > 70 ? "negative" : "positive",
            explanation:
              features.complexity > 70
                ? "High complexity makes code difficult to maintain"
                : "Moderate complexity indicates good maintainability",
          },
          {
            name: "Documentation",
            value: features.comment_ratio,
            format: "percent",
            impact: features.comment_ratio < 0.15 ? "negative" : "positive",
            explanation:
              features.comment_ratio < 0.15
                ? "Insufficient code documentation"
                : "Good documentation practices detected",
          },
          {
            name: "Circular Dependencies",
            value: features.circular_dependencies,
            impact:
              features.circular_dependencies > 5 ? "negative" : "positive",
            explanation:
              features.circular_dependencies > 5
                ? "Circular dependencies complicate maintenance"
                : "Good architectural structure with minimal circular references",
          },
        ];

      case "reliability":
        return [
          {
            name: "Test Coverage",
            value: features.test_to_code_ratio,
            format: "percent",
            impact: features.test_to_code_ratio < 0.3 ? "negative" : "positive",
            explanation:
              features.test_to_code_ratio < 0.3
                ? "Low test coverage increases risk of bugs"
                : "Good test coverage increases reliability",
          },
          {
            name: "Error Handling",
            value: features.lint_errors,
            impact: features.lint_errors > 30 ? "negative" : "positive",
            explanation:
              features.lint_errors > 30
                ? "Many lint errors suggest potential runtime issues"
                : "Few lint errors indicate good coding practices",
          },
          {
            name: "Test Frameworks",
            value: features.test_frameworks_count,
            impact:
              features.test_frameworks_count < 1 ? "negative" : "positive",
            explanation:
              features.test_frameworks_count < 1
                ? "No test frameworks detected"
                : "Testing framework(s) properly implemented",
          },
        ];

      case "scalability":
        return [
          {
            name: "Dependency Structure",
            value: features.avg_dependencies,
            impact: features.avg_dependencies > 15 ? "negative" : "positive",
            explanation:
              features.avg_dependencies > 15
                ? "High average dependencies impact scalability"
                : "Well-managed dependency structure",
          },
          {
            name: "Circular Dependencies",
            value: features.circular_dependencies,
            impact:
              features.circular_dependencies > 3 ? "negative" : "positive",
            explanation:
              features.circular_dependencies > 3
                ? "Circular references limit scalability"
                : "Clean architecture supports scaling",
          },
          {
            name: "Component Coupling",
            value: features.avg_fan_out,
            impact: features.avg_fan_out > 10 ? "negative" : "positive",
            explanation:
              features.avg_fan_out > 10
                ? "High coupling between components"
                : "Components are appropriately decoupled",
          },
        ];

      case "security":
        return [
          {
            name: "Dependency Management",
            value: features.has_dependency_management ? "Yes" : "No",
            format: "text",
            impact: !features.has_dependency_management
              ? "negative"
              : "positive",
            explanation: !features.has_dependency_management
              ? "No dependency management system detected"
              : "Proper dependency management found",
          },
          {
            name: "Code Quality Issues",
            value: features.lint_errors,
            impact: features.lint_errors > 25 ? "negative" : "positive",
            explanation:
              features.lint_errors > 25
                ? "Lint errors may indicate security vulnerabilities"
                : "Few code quality issues detected",
          },
          {
            name: "Test Coverage",
            value: features.test_to_code_ratio,
            format: "percent",
            impact: features.test_to_code_ratio < 0.2 ? "negative" : "positive",
            explanation:
              features.test_to_code_ratio < 0.2
                ? "Low test coverage may miss security issues"
                : "Testing helps ensure secure code",
          },
        ];

      case "efficiency":
        return [
          {
            name: "Code Complexity",
            value: features.complexity,
            impact: features.complexity > 60 ? "negative" : "positive",
            explanation:
              features.complexity > 60
                ? "Complex code often has efficiency issues"
                : "Reasonable complexity indicates efficient code",
          },
          {
            name: "Function Complexity",
            value: features.avg_function_complexity,
            impact:
              features.avg_function_complexity > 15 ? "negative" : "positive",
            explanation:
              features.avg_function_complexity > 15
                ? "Complex functions may impact performance"
                : "Functions have good complexity metrics",
          },
          {
            name: "Code Size",
            value: features.loc,
            impact: features.loc > 5000 ? "warning" : "positive",
            explanation:
              features.loc > 5000
                ? "Large codebase - review for efficiency"
                : "Codebase size is reasonable",
          },
        ];

      default:
        return [];
    }
  };

  const factors = getInfluencingFactors(dimension, features);
  const scorePercentage = Math.round(score * 100);

  // Determine the overall assessment based on score
  const getAssessment = (score) => {
    if (score >= 0.8) return "Excellent";
    if (score >= 0.7) return "Very Good";
    if (score >= 0.6) return "Good";
    if (score >= 0.5) return "Satisfactory";
    if (score >= 0.4) return "Fair";
    if (score >= 0.3) return "Needs Improvement";
    return "Poor";
  };

  // Format value based on type
  const formatValue = (value, format) => {
    if (format === "percent") {
      return `${Math.round(value * 100)}%`;
    }
    if (format === "text") {
      return value;
    }
    return value;
  };

  return (
    <div className="mt-4">
      <h3 className="text-lg font-semibold mb-2">
        {dimension.charAt(0).toUpperCase() + dimension.slice(1)} Score
        Explanation
      </h3>

      <div className="p-4 rounded-lg bg-gray-100 dark:bg-gray-800">
        <div className="flex items-center mb-3">
          <div
            className="w-12 h-12 rounded-full flex items-center justify-center text-white font-bold mr-3"
            style={{
              backgroundColor:
                scorePercentage >= 70
                  ? "#4CAF50"
                  : scorePercentage >= 50
                  ? "#FFC107"
                  : "#F44336",
            }}
          >
            {scorePercentage}%
          </div>
          <div>
            <div className="font-semibold">{getAssessment(score)}</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Overall {dimension} assessment
            </div>
          </div>
        </div>

        <div className="text-sm mb-4">
          {scorePercentage >= 70
            ? `This codebase demonstrates strong ${dimension} characteristics.`
            : scorePercentage >= 50
            ? `This codebase has adequate ${dimension}, but there's room for improvement.`
            : `This codebase needs significant improvements in ${dimension}.`}
        </div>

        <div className="text-sm font-medium mb-2">
          Key factors influencing this score:
        </div>
        <div className="space-y-3">
          {factors.map((factor, index) => (
            <div key={index} className="flex items-start">
              <div
                className={`mt-1 w-3 h-3 rounded-full mr-2 flex-shrink-0 ${
                  factor.impact === "positive"
                    ? "bg-green-500"
                    : factor.impact === "warning"
                    ? "bg-yellow-500"
                    : "bg-red-500"
                }`}
              ></div>
              <div>
                <div className="flex items-center">
                  <span className="font-medium">{factor.name}:</span>
                  <span
                    className={`ml-1 ${
                      factor.impact === "positive"
                        ? "text-green-600 dark:text-green-400"
                        : factor.impact === "warning"
                        ? "text-yellow-600 dark:text-yellow-400"
                        : "text-red-600 dark:text-red-400"
                    }`}
                  >
                    {formatValue(factor.value, factor.format)}
                  </span>
                </div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  {factor.explanation}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ScoreExplanation;
