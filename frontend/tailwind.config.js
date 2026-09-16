/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    container: {
      center: true,
      padding: "1rem",
    },
    extend: {
      colors: {
        // --- Soft Clinical + Quantum theme -----------------------------
        background: "#F8FAFC",       // soft off-white / very light gray
        foreground: "#44403C",       // warm gray text
        primary: {
          DEFAULT: "#0F766E",        // deep teal / emerald
          foreground: "#F8FAFC",
          50: "#ECFDFA",
          100: "#CCFBF1",
          500: "#0F766E",
          600: "#0D6660",
          700: "#0B534E",
        },
        accent: {
          DEFAULT: "#8B5CF6",        // soft violet (quantum touch)
          foreground: "#F8FAFC",
          50: "#F5F3FF",
          100: "#EDE9FE",
          500: "#8B5CF6",
          600: "#7C3AED",
        },
        muted: {
          DEFAULT: "#F1F5F9",
          foreground: "#78716C",
        },
        card: {
          DEFAULT: "#FFFFFF",
          foreground: "#44403C",
        },
        border: "#E7E5E4",
        warmgray: {
          50: "#FAFAF9",
          100: "#F5F5F4",
          400: "#A8A29E",
          500: "#78716C",
          700: "#44403C",
          900: "#1C1917",
        },
        risk: {
          low: "#0F766E",
          medium: "#D97706",
          high: "#DC2626",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      borderRadius: {
        lg: "1rem",
        xl: "1.25rem",
        "2xl": "1.5rem",
      },
      boxShadow: {
        soft: "0 2px 20px -4px rgba(15, 118, 110, 0.08)",
        card: "0 4px 24px -8px rgba(68, 64, 60, 0.10)",
        glow: "0 0 40px -8px rgba(139, 92, 246, 0.25)",
      },
      backgroundImage: {
        "quantum-grid":
          "radial-gradient(circle at 1px 1px, rgba(139,92,246,0.12) 1px, transparent 0)",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-8px)" },
        },
        pulseSoft: {
          "0%, 100%": { opacity: 1 },
          "50%": { opacity: 0.6 },
        },
      },
      animation: {
        float: "float 6s ease-in-out infinite",
        pulseSoft: "pulseSoft 2.5s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
