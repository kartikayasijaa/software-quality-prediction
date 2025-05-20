/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        dark: {
          bg: "#1a1a1a",
          card: "#2d2d2d",
          text: "#e2e8f0",
          border: "#3f3f3f",
          primary: "#3b82f6",
          secondary: "#10b981",
        },
      },
    },
  },
  plugins: [],
};
