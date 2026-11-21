import React from "react";
import { Link, NavLink, Route, Routes, useParams } from "react-router-dom";
import UploadPage from "./pages/UploadPage.jsx";
import TrainingPage from "./pages/TrainingPage.jsx";
import ReportPage from "./pages/ReportPage.jsx";
import PredictPage from "./pages/PredictPage.jsx";
import MonitoringPage from "./pages/MonitoringPage.jsx";
import OverviewPage from "./pages/OverviewPage.jsx";
import { ApiStatus, HealthProvider, OfflineBanner } from "./components/status.jsx";

function SessionNav() {
  const { id } = useParams();
  const tabs = [
    ["overview", "Overview"],
    ["train", "Training"],
    ["report", "Report"],
    ["predict", "Predict"],
    ["monitor", "Monitoring"],
  ];
  return (
    <nav aria-label="Session" className="flex max-w-full gap-1 overflow-x-auto">
      {tabs.map(([path, label]) => (
        <NavLink
          key={path}
          to={`/s/${id}/${path}`}
          className={({ isActive }) =>
            `whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium ${
              isActive ? "bg-indigo-50 text-indigo-700" : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
            }`
          }
        >
          {label}
        </NavLink>
      ))}
    </nav>
  );
}

function Shell({ children, session = false }) {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3">
          <Link to="/" className="flex items-center gap-2">
            <img src="/favicon.svg" alt="" className="h-7 w-7" />
            <span className="text-lg font-semibold tracking-tight">ModelDoctor</span>
            <span className="hidden text-sm text-slate-500 sm:inline">· tells you when not to believe your model</span>
          </Link>
          <div className="flex max-w-full flex-wrap items-center gap-2">
            {session && <SessionNav />}
            <ApiStatus />
          </div>
        </div>
      </header>
      <OfflineBanner />
      <main className="mx-auto max-w-7xl px-4 py-6">{children}</main>
    </div>
  );
}

export default function App() {
  return (
    <HealthProvider>
      <Routes>
        <Route path="/" element={<Shell><UploadPage /></Shell>} />
        <Route path="/s/:id/overview" element={<Shell session><OverviewPage /></Shell>} />
        <Route path="/s/:id/train" element={<Shell session><TrainingPage /></Shell>} />
        <Route path="/s/:id/report" element={<Shell session><ReportPage /></Shell>} />
        <Route path="/s/:id/predict" element={<Shell session><PredictPage /></Shell>} />
        <Route path="/s/:id/monitor" element={<Shell session><MonitoringPage /></Shell>} />
        <Route
          path="*"
          element={
            <Shell>
              <div className="py-20 text-center">
                <h1 className="text-xl font-semibold">Page not found</h1>
                <Link className="mt-2 inline-block text-indigo-600 underline" to="/">Back to upload</Link>
              </div>
            </Shell>
          }
        />
      </Routes>
    </HealthProvider>
  );
}
