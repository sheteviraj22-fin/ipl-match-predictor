"use client";

import { useState } from "react";

const API = "http://localhost:8000";

type SeasonStats = {
  accuracy: number;
  matches: number;
  correct: number;
};

type BacktestResult = {
  accuracy: number;
  total_matches: number;
  correct: number;
  per_season: Record<string, SeasonStats>;
  best_season: string | null;
  worst_season: string | null;
  best_season_accuracy: number;
  worst_season_accuracy: number;
  longest_correct_streak: number;
  longest_incorrect_streak: number;
};

export default function BacktestPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BacktestResult | null>(null);

  async function runBacktest() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API}/backtest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Request failed");
      }

      const data: BacktestResult = await res.json();
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  const seasons = result
    ? Object.entries(result.per_season).sort(([a], [b]) => a.localeCompare(b))
    : [];

  return (
    <main style={{ background: "#0B0F14", minHeight: "100vh", color: "#fff", padding: "40px 32px", fontFamily: "monospace" }}>
      <h1 style={{ fontSize: "20px", fontWeight: 700, marginBottom: "8px", color: "#F59E0B" }}>
        IPL Model — Backtest
      </h1>
      <p style={{ fontSize: "13px", color: "#6B7280", marginBottom: "32px" }}>
        Walk-forward evaluation. No lookahead bias.
      </p>

      <button
        onClick={runBacktest}
        disabled={loading}
        style={{
          background: loading ? "#374151" : "#F59E0B",
          color: loading ? "#9CA3AF" : "#000",
          border: "none",
          padding: "10px 24px",
          fontSize: "14px",
          fontWeight: 600,
          cursor: loading ? "not-allowed" : "pointer",
          fontFamily: "monospace",
        }}
      >
        {loading ? "Running..." : "Run Backtest"}
      </button>

      {loading && (
        <p style={{ marginTop: "20px", color: "#6B7280", fontSize: "13px" }}>
          Computing walk-forward predictions — this may take a minute.
        </p>
      )}

      {error && (
        <p style={{ marginTop: "20px", color: "#EF4444", fontSize: "13px" }}>
          Error: {error}
        </p>
      )}

      {result && (
        <>
          {/* Metrics */}
          <section style={{ marginTop: "36px" }}>
            <h2 style={{ fontSize: "14px", color: "#F59E0B", marginBottom: "16px", textTransform: "uppercase", letterSpacing: "1px" }}>
              Summary
            </h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: "16px" }}>
              {[
                ["Accuracy", `${(result.accuracy * 100).toFixed(1)}%`],
                ["Total Matches", result.total_matches],
                ["Correct Predictions", result.correct],
                ["Longest Win Streak", result.longest_correct_streak],
                ["Longest Loss Streak", result.longest_incorrect_streak],
                ["Best Season", result.best_season ?? "—"],
                ["Worst Season", result.worst_season ?? "—"],
                ["Best Season Accuracy", `${(result.best_season_accuracy * 100).toFixed(1)}%`],
                ["Worst Season Accuracy", `${(result.worst_season_accuracy * 100).toFixed(1)}%`],
              ].map(([label, value]) => (
                <div key={label as string}>
                  <div style={{ fontSize: "11px", color: "#6B7280", marginBottom: "4px" }}>{label}</div>
                  <div style={{ fontSize: "16px", fontWeight: 600 }}>{value}</div>
                </div>
              ))}
            </div>
          </section>

          {/* Per-season table */}
          <section style={{ marginTop: "40px" }}>
            <h2 style={{ fontSize: "14px", color: "#F59E0B", marginBottom: "16px", textTransform: "uppercase", letterSpacing: "1px" }}>
              Per Season
            </h2>
            <table style={{ borderCollapse: "collapse", fontSize: "13px", width: "100%", maxWidth: "600px" }}>
              <thead>
                <tr style={{ color: "#6B7280", textAlign: "left" }}>
                  {["Season", "Matches", "Correct", "Accuracy"].map((h) => (
                    <th key={h} style={{ padding: "6px 16px 6px 0", fontWeight: 400, borderBottom: "1px solid #1F2937" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {seasons.map(([season, stats]) => (
                  <tr key={season} style={{ borderBottom: "1px solid #111827" }}>
                    <td style={{ padding: "8px 16px 8px 0", color: season === result.best_season ? "#F59E0B" : "#fff" }}>{season}</td>
                    <td style={{ padding: "8px 16px 8px 0" }}>{stats.matches}</td>
                    <td style={{ padding: "8px 16px 8px 0" }}>{stats.correct}</td>
                    <td style={{ padding: "8px 16px 8px 0" }}>{(stats.accuracy * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ fontSize: "11px", color: "#374151", marginTop: "8px" }}>
              Best season highlighted in amber.
            </p>
          </section>
        </>
      )}
    </main>
  );
}
