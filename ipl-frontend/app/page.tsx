"use client";

import { useState, useEffect, useRef } from "react";

// ─── DATA ────────────────────────────────────────────────────────────────────

const SQUADS: Record<string, string[]> = {
  CSK: ["A Kamboj","A Mhatre","AJ Hosein","Aman Hakim Khan","D Brevis","Gurjapneet Singh","J Overton","KK Ahmed","Kartik Sharma","MJ Henry","MS Dhoni","MW Short","Mukesh Choudhary","Noor Ahmad","PR Veer","RD Chahar","RD Gaikwad","S Dube","S Gopal","SH Johnson","SN Khan","SV Samson","Urvil Patel"],
  DC:  ["AR Patel","Abishek Porel","Ashutosh Sharma","Auqib Nabi","DA Miller","K Yadav","KA Jamieson","KK Nair","KL Rahul","Kuldeep Yadav","L Ngidi","M Tiwari","MA Starc","Mukesh Kumar","N Rana","P Nissanka","PP Shaw","Sameer Rizvi","T Natarajan","T Stubbs","V Nigam"],
  GT:  ["Anuj Rawat","Arshad Khan","B Sai Sudharsan","GD Phillips","Gurnoor Brar","I Sharma","J Yadav","JC Buttler","JO Holder","K Rabada","Kumar Kushagra","L Wood","M Prasidh Krishna","M Shahrukh Khan","Mohammed Siraj","R Sai Kishore","R Tewatia","Rashid Khan","Shubman Gill","Washington Sundar"],
  KKR: ["A Raghuvanshi","AM Rahane","B Muzarabani","C Green","CV Varun","FH Allen","Kartik Tyagi","M Pathirana","MK Pandey","N Saini","PH Solanki","R Powell","R Ravindra","RA Tripathi","RK Singh","Ramandeep Singh","SP Narine","TL Seifert","Umran Malik","VG Arora"],
  LSG: ["A Badoni","A Nortje","AK Markram","Abdul Samad","Akash Singh","Arjun Tendulkar","Avesh Khan","GF Linde","Himmat Singh","JP Inglis","M Siddharth","MP Breetzke","MR Marsh","Mohammed Shami","Mohsin Khan","N Pooran","Prince Yadav","RR Pant","Shahbaz Ahmed"],
  MI:  ["AM Ghazanfar","Ashwani Kumar","C Bosch","DL Chahar","HH Pandya","JJ Bumrah","M Markande","MJ Santner","Naman Dhir","Q de Kock","R Minz","RA Bawa","RD Rickelton","RG Sharma","SA Yadav","SE Rutherford","SN Thakur","TA Boult","Tilak Varma","WG Jacks"],
  PBKS: ["Arshdeep Singh","Azmatullah Omarzai","C Connolly","DP Vijaykumar","Harpreet Brar","LH Ferguson","M Jansen","MJ Owen","MP Stoinis","Musheer Khan","N Wadhera","P Dubey","P Simran Singh","Priyansh Arya","SS Iyer","Shashank Singh","Suryansh Shedge","Vijaykumar Vyshak","Vishnu Vinod","XC Bartlett","YS Chahal","Yash Thakur"],
  RCB: ["Abhinandan Singh","B Kumar","D Padikkal","JA Duffy","JG Bethell","JM Sharma","JR Hazlewood","N Thushara","PD Salt","R Shepherd","RM Patidar","Rasikh Salam","Suyash Sharma","Swapnil Singh","TH David","V Kohli","VR Iyer","Yash Dayal"],
  RR:  ["AF Milne","D Ferreira","Dhruv Jurel","JC Archer","KR Sen","KT Maphaka","LG Pretorius","N Burger","R Bishnoi","R Parag","RA Jadeja","Ravi Bishnoi","SB Dubey","SO Hetmyer","Sandeep Sharma","TU Deshpande","V Puthur","V Suryavanshi","YBK Jaiswal","Yudhvir Singh"],
  SRH: ["Abhishek Sharma","E Malinga","G Coetzee","H Klaasen","Harsh Dubey","Ishan Kishan","LS Livingstone","Nithish Kumar Reddy","PJ Cummins","PP Hinge","S Arora","Sakib Hussain","Shivam Mavi","TM Head","Zeeshan Ansari"],
};

// IPL group structure (2025)
const GROUP_A = ["CSK", "KKR", "RR", "RCB", "PBKS"];
const GROUP_B = ["MI", "SRH", "GT", "DC", "LSG"];
const TEAMS = [...GROUP_A, ...GROUP_B];

// Single identifying color per team (used for labels and bars)
const TEAM_COLOR: Record<string, string> = {
  CSK: "#F5A623", MI: "#1D6BCC", RCB: "#EC1C24", KKR: "#7C3AED",
  GT: "#4A7BB5",  LSG: "#0EA5E9", RR: "#EC4899", SRH: "#F97316",
  DC: "#2563EB",  PBKS: "#E11D48",
};

const VENUES = [
  "Wankhede Stadium, Mumbai","M. A. Chidambaram Stadium, Chennai",
  "Eden Gardens, Kolkata","Narendra Modi Stadium, Ahmedabad",
  "M. Chinnaswamy Stadium, Bengaluru","Rajiv Gandhi International Stadium, Hyderabad",
  "Sawai Mansingh Stadium, Jaipur","Punjab Cricket Association Stadium, Mohali",
  "Arun Jaitley Stadium, Delhi","BRSABV Ekana Cricket Stadium, Lucknow",
];

// ─── FIXTURE BUILDER (IPL schedule structure) ────────────────────────────────

function buildFixtures(defaultVenue: string) {
  const fixtures: { match_id: string; teamA: string; teamB: string; venue: string; toss_winner: null }[] = [];
  let mid = 1;

  // Within group: each pair plays once
  for (const grp of [GROUP_A, GROUP_B]) {
    for (let i = 0; i < grp.length; i++)
      for (let j = i + 1; j < grp.length; j++)
        fixtures.push({ match_id: `${mid++}`, teamA: grp[i], teamB: grp[j], venue: defaultVenue, toss_winner: null });
  }

  // Cross group: each pair plays twice (home + away)
  for (const a of GROUP_A)
    for (const b of GROUP_B) {
      fixtures.push({ match_id: `${mid++}`, teamA: a, teamB: b, venue: defaultVenue, toss_winner: null });
      fixtures.push({ match_id: `${mid++}`, teamA: b, teamB: a, venue: defaultVenue, toss_winner: null });
    }

  return fixtures;
}

// ─── API ─────────────────────────────────────────────────────────────────────

const BASE =
process.env.NEXT_PUBLIC_API_URL ||
"http://localhost:8000";

async function apiFetch(path: string, body?: unknown) {
  const res = await fetch(`${BASE}${path}`, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ─── MOCK DATA ────────────────────────────────────────────────────────────────


// ─── TYPES ───────────────────────────────────────────────────────────────────

interface Prediction {
  teamA: string; teamB: string;
  win_probability_A: number; win_probability_B: number;
  base_diff: number; venue_adjustment: number; toss_adjustment: number;
}
interface TeamStrength {
  batting_unit: number; bowling_unit: number; allrounder_balance: number; total_strength: number;
  squad_size: number; squad_matched: number;
}
interface SimResult {
  team: string; title_prob: number; playoff_prob: number; avg_points: number; avg_wins: number;
}

// ─── ANIMATED NUMBER ─────────────────────────────────────────────────────────

function AnimNum({ value, decimals = 1 }: { value: number; decimals?: number }) {
  const [display, setDisplay] = useState(0);
  const rafRef = useRef<number>(0);
  useEffect(() => {
    const start = display;
    const end = value;
    const t0 = performance.now();
    const tick = (now: number) => {
      const t = Math.min((now - t0) / 700, 1);
      const ease = 1 - (1 - t) ** 3;
      setDisplay(start + (end - start) * ease);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [value]); // eslint-disable-line
  return <>{display.toFixed(decimals)}</>;
}

// ─── COMPONENTS ──────────────────────────────────────────────────────────────

function Spinner({ dark }: { dark?: boolean }) {
  return (
    <span style={{
      display: "inline-block", width: 13, height: 13, flexShrink: 0,
      border: `2px solid ${dark ? "rgba(0,0,0,0.2)" : "#1E2836"}`,
      borderTopColor: dark ? "#000" : "#F59E0B",
      borderRadius: "50%", animation: "spin 0.6s linear infinite",
    }} />
  );
}

function Card({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{ background: "#0F1923", border: "1px solid #1E2836", borderRadius: 10, padding: 20, ...style }}>
      {children}
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 14 }}>
      {children}
    </div>
  );
}

// ─── PROBABILITY BAR ─────────────────────────────────────────────────────────

function ProbBar({ pA, pB, teamA, teamB }: { pA: number; pB: number; teamA: string; teamB: string }) {
  const pctA = Math.round(pA * 100);
  const pctB = 100 - pctA;
  const cA = TEAM_COLOR[teamA] || "#F59E0B";
  const cB = TEAM_COLOR[teamB] || "#6366F1";
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, fontSize: 13 }}>
        <span style={{ fontWeight: 700, color: cA }}>{teamA} · {pctA}%</span>
        <span style={{ fontWeight: 700, color: cB }}>{pctB}% · {teamB}</span>
      </div>
      <div style={{ height: 10, borderRadius: 5, background: "#1E2836", overflow: "hidden", display: "flex" }}>
        <div style={{ width: `${pctA}%`, background: cA, transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)" }} />
        <div style={{ flex: 1, background: cB }} />
      </div>
      <div style={{ marginTop: 6, display: "flex", justifyContent: "space-between", fontSize: 10, color: "#334155" }}>
        <span>win probability</span>
        <span>{pctA > pctB ? teamA : teamB} favoured</span>
      </div>
    </div>
  );
}

// ─── STRENGTH ROW ─────────────────────────────────────────────────────────────

function StrengthRow({ label, vA, vB, cA, cB }: { label: string; vA: number; vB: number; cA: string; cB: string }) {
  const total = (vA + vB) || 1;
  const pctA = (vA / total) * 100;
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#475569", marginBottom: 5, textTransform: "uppercase", letterSpacing: "0.08em" }}>
        <span>{vA.toFixed(3)}</span>
        <span>{label}</span>
        <span>{vB.toFixed(3)}</span>
      </div>
      <div style={{ height: 6, borderRadius: 3, background: "#1E2836", overflow: "hidden", display: "flex" }}>
        <div style={{ width: `${pctA}%`, background: cA, borderRadius: "3px 0 0 3px", transition: "width 0.9s cubic-bezier(0.4,0,0.2,1)" }} />
        <div style={{ flex: 1, background: cB, borderRadius: "0 3px 3px 0" }} />
      </div>
    </div>
  );
}

// ─── SIM BAR ─────────────────────────────────────────────────────────────────

function SimBar({ result, maxProb }: { result: SimResult; maxProb: number }) {
  const c = TEAM_COLOR[result.team] || "#F59E0B";
  const w = maxProb > 0 ? (result.title_prob / maxProb) * 100 : 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
      <div style={{ width: 40, fontSize: 11, fontWeight: 700, color: c, letterSpacing: "0.04em" }}>{result.team}</div>
      <div style={{ flex: 1, height: 20, background: "#111720", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${w}%`, height: "100%", background: c, opacity: 0.85, transition: "width 0.8s cubic-bezier(0.4,0,0.2,1)" }} />
      </div>
      <div style={{ width: 44, textAlign: "right", fontSize: 12, fontWeight: 600, color: "#E2E8F0" }}>
        {(result.title_prob * 100).toFixed(1)}%
      </div>
      <div style={{ width: 36, textAlign: "right", fontSize: 10, color: "#334155" }}>
        {(result.playoff_prob * 100).toFixed(0)}%↑
      </div>
    </div>
  );
}

// ─── TEAM SELECT ─────────────────────────────────────────────────────────────

function TeamSelect({ label, value, onChange, exclude }: { label: string; value: string; onChange: (t: string) => void; exclude?: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);
  const c = TEAM_COLOR[value];
  return (
    <div ref={ref} style={{ position: "relative", flex: 1 }}>
      <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: "100%", padding: "9px 14px", borderRadius: 8, cursor: "pointer",
          background: "#111720", border: `1.5px solid ${open ? "#F59E0B" : "#1E2836"}`,
          color: c || "#F1F5F9", fontSize: 14, fontWeight: 700,
          display: "flex", alignItems: "center", justifyContent: "space-between",
          transition: "border-color 0.15s",
        }}
      >
        <span>{value || "—"}</span>
        <span style={{ fontSize: 9, color: "#475569", transform: open ? "rotate(180deg)" : "", transition: "transform 0.15s" }}>▼</span>
      </button>
      {open && (
        <div style={{
          position: "absolute", top: "calc(100% + 4px)", left: 0, right: 0, zIndex: 60,
          background: "#16202E", border: "1px solid #1E2836", borderRadius: 8,
          overflow: "hidden", boxShadow: "0 8px 32px rgba(0,0,0,0.7)",
        }}>
          {TEAMS.filter(t => t !== exclude).map(t => (
            <button key={t} onClick={() => { onChange(t); setOpen(false); }}
              style={{
                width: "100%", padding: "9px 14px", background: t === value ? "#1E2836" : "transparent",
                border: "none", color: t === value ? TEAM_COLOR[t] : "#94A3B8",
                fontSize: 12, fontWeight: t === value ? 700 : 400, cursor: "pointer",
                textAlign: "left", display: "flex", alignItems: "center", gap: 8,
                transition: "background 0.1s",
              }}
              onMouseEnter={e => { if (t !== value) (e.currentTarget as HTMLElement).style.background = "#1a2535"; }}
              onMouseLeave={e => { if (t !== value) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
            >
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: TEAM_COLOR[t], flexShrink: 0 }} />
              {t}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── PLAYER CHIPS ─────────────────────────────────────────────────────────────

function PlayerChips({ team, selected, onToggle }: { team: string; selected: string[]; onToggle: (p: string) => void }) {
  const [query, setQuery] = useState("");
  const players = SQUADS[team] || [];
  const filtered = players.filter(p => p.toLowerCase().includes(query.toLowerCase()));
  const c = TEAM_COLOR[team] || "#F59E0B";
  return (
    <div>
      <input
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder={`Filter ${team} players…`}
        style={{
          width: "100%", padding: "7px 11px", marginBottom: 10, borderRadius: 6,
          background: "#0B0F14", border: "1px solid #1E2836", color: "#F1F5F9",
          fontSize: 11, outline: "none", boxSizing: "border-box", transition: "border-color 0.15s",
        }}
        onFocus={e => (e.target.style.borderColor = c)}
        onBlur={e => (e.target.style.borderColor = "#1E2836")}
      />
      <div style={{ display: "flex", flexWrap: "wrap", gap: 5, maxHeight: 140, overflowY: "auto" }}>
        {filtered.map(p => {
          const active = selected.includes(p);
          return (
            <button key={p} onClick={() => onToggle(p)}
              style={{
                padding: "3px 10px", borderRadius: 20, fontSize: 11, cursor: "pointer",
                background: active ? c : "#1E2836", color: active ? "#0B0F14" : "#64748B",
                border: "none", fontWeight: active ? 700 : 400, transition: "all 0.1s",
              }}
            >{p}</button>
          );
        })}
      </div>
      <div style={{ marginTop: 7, fontSize: 10, color: "#334155" }}>{selected.length}/{players.length} selected</div>
    </div>
  );
}

// ─── ADJUSTMENT CELL ─────────────────────────────────────────────────────────

function AdjCell({ label, value }: { label: string; value: number }) {
  const sign = value >= 0 ? "+" : "";
  const color = value > 0.005 ? "#22C55E" : value < -0.005 ? "#EF4444" : "#475569";
  return (
    <div style={{ padding: "12px 14px", background: "#111720", borderRadius: 8, border: "1px solid #1E2836" }}>
      <div style={{ fontSize: 10, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 17, fontWeight: 700, color }}>{sign}{(value * 100).toFixed(2)}</div>
    </div>
  );
}

// ─── SORTABLE TABLE ───────────────────────────────────────────────────────────

type SortKey = "title_prob" | "playoff_prob" | "avg_wins" | "avg_points";

function SimTable({ data }: { data: SimResult[] }) {
  const [sortBy, setSortBy] = useState<SortKey>("title_prob");
  const [asc, setAsc] = useState(false);
  const sorted = [...data].sort((a, b) => (asc ? 1 : -1) * (a[sortBy] - b[sortBy]));

  const TH = ({ k, label }: { k: SortKey; label: string }) => (
    <th onClick={() => { sortBy === k ? setAsc(a => !a) : (setSortBy(k), setAsc(false)); }}
      style={{ padding: "8px 12px", textAlign: "right", fontSize: 10, color: sortBy === k ? "#F59E0B" : "#475569", letterSpacing: "0.1em", textTransform: "uppercase", cursor: "pointer", userSelect: "none", whiteSpace: "nowrap" }}>
      {label} {sortBy === k ? (asc ? "↑" : "↓") : ""}
    </th>
  );

  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
      <thead>
        <tr style={{ borderBottom: "1px solid #1E2836" }}>
          <th style={{ padding: "8px 12px", textAlign: "left", fontSize: 10, color: "#475569", letterSpacing: "0.1em", textTransform: "uppercase" }}>Team</th>
          <TH k="title_prob" label="Title %" />
          <TH k="playoff_prob" label="Playoff %" />
          <TH k="avg_wins" label="Avg Wins" />
          <TH k="avg_points" label="Avg Pts" />
        </tr>
      </thead>
      <tbody>
        {sorted.map((r, i) => {
          const c = TEAM_COLOR[r.team] || "#F59E0B";
          return (
            <tr key={r.team} style={{ borderBottom: "1px solid #0F1923", transition: "background 0.1s" }}
              onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = "#111720"}
              onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = "transparent"}>
              <td style={{ padding: "10px 12px" }}>
                <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 10, color: "#334155", width: 18 }}>#{i + 1}</span>
                  <span style={{ width: 6, height: 6, borderRadius: "50%", background: c, flexShrink: 0 }} />
                  <span style={{ fontWeight: 700, color: c }}>{r.team}</span>
                </span>
              </td>
              <td style={{ padding: "10px 12px", textAlign: "right", fontWeight: 700, color: "#F59E0B" }}>{(r.title_prob * 100).toFixed(1)}%</td>
              <td style={{ padding: "10px 12px", textAlign: "right", color: "#94A3B8" }}>{(r.playoff_prob * 100).toFixed(1)}%</td>
              <td style={{ padding: "10px 12px", textAlign: "right", color: "#94A3B8" }}>{r.avg_wins?.toFixed(1) ?? "—"}</td>
              <td style={{ padding: "10px 12px", textAlign: "right", color: "#94A3B8" }}>{r.avg_points?.toFixed(1) ?? "—"}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ─── MAIN ────────────────────────────────────────────────────────────────────

export default function IPLDashboard() {
  const [tab, setTab] = useState<"builder" | "prediction" | "simulator">("builder");

  // Match builder state
  const [teamA, setTeamA] = useState("MI");
  const [teamB, setTeamB] = useState("RCB");
  const [squadA, setSquadA] = useState(SQUADS["MI"].slice(0, 11));
  const [squadB, setSquadB] = useState(SQUADS["RCB"].slice(0, 11));
  const [venue, setVenue] = useState(VENUES[0]);
  const [tossWinner, setTossWinner] = useState<string>("");

  // Results state
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [strengthA, setStrengthA] = useState<TeamStrength | null>(null);
  const [strengthB, setStrengthB] = useState<TeamStrength | null>(null);
  const [simResults, setSimResults] = useState<SimResult[] | null>(null);

  // Loading / API
  const [predLoading, setPredLoading] = useState(false);
  const [simLoading, setSimLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    apiFetch("/health").then(() => setApiOnline(true)).catch(() => setApiOnline(false));
  }, []);

  useEffect(() => { setSquadA(SQUADS[teamA]?.slice(0, 11) || []); }, [teamA]);
  useEffect(() => { setSquadB(SQUADS[teamB]?.slice(0, 11) || []); }, [teamB]);

  const toggleA = (p: string) => setSquadA(s => s.includes(p) ? s.filter(x => x !== p) : [...s, p]);
  const toggleB = (p: string) => setSquadB(s => s.includes(p) ? s.filter(x => x !== p) : [...s, p]);

  const handlePredict = async () => {
    setPredLoading(true);
    try {
      const [pred, sA, sB] = await Promise.all([
        apiFetch(
          "/predict-match",
          {
            teamA,
            teamB,
            squads: {
              [teamA]: squadA,
              [teamB]: squadB
            },
            venue,
            toss_winner: tossWinner || null
          }
        ),

        apiFetch(
          "/team-strength",
          {
            team: teamA,
            squad: squadA
          }
        ),

        apiFetch(
          "/team-strength",
          {
            team: teamB,
            squad: squadB
          }
        )
      ]);

      setPrediction(pred);
      setStrengthA(sA);
      setStrengthB(sB);
      setTab("prediction");
    } catch (e) { console.error(e); }
    setPredLoading(false);
  };

  const handleSimulate = async () => {
    setSimLoading(true);
    try {
      if (apiOnline) {
        const fixtures = buildFixtures(venue);
        const allSquads: Record<string, string[]> = {};
        TEAMS.forEach(t => { allSquads[t] = SQUADS[t].slice(0, 11); });
        const results = await apiFetch("/simulate-season", { fixtures, squads: allSquads, simulations: 10000 });
        setSimResults(results);
      }
      setTab("simulator");
    } catch (e) { console.error(e); }
    setSimLoading(false);
  };

  const cA = TEAM_COLOR[teamA] || "#F59E0B";
  const cB = TEAM_COLOR[teamB] || "#6366F1";
  const maxProb = simResults ? Math.max(...simResults.map(r => r.title_prob)) : 1;

  return (
    <div style={{ minHeight: "100vh", background: "#0B0F14", color: "#F1F5F9", fontFamily: "'DM Sans', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        button, input, select { font-family: inherit; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #1E2836; border-radius: 2px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .fade-in { animation: fadeIn 0.25s ease forwards; }
      `}</style>

      {/* ── TOP BAR ── */}
      <div style={{ borderBottom: "1px solid #1E2836", padding: "0 24px", height: 52, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 15, fontWeight: 700, letterSpacing: "-0.02em" }}>IPL Engine</span>
          <span style={{ fontSize: 10, color: "#334155", letterSpacing: "0.1em" }}>2025 · Monte Carlo</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span style={{
            width: 6, height: 6, borderRadius: "50%", flexShrink: 0,
            background: apiOnline === null ? "#475569" : apiOnline ? "#22C55E" : "#F59E0B",
          }} />
          <span style={{ fontSize: 11, color: "#475569" }}>
            {apiOnline === null ? "connecting…" : apiOnline ? "API live" : "demo mode"}
          </span>
        </div>
      </div>

      {/* ── TAB BAR ── */}
      <div style={{ borderBottom: "1px solid #1E2836", padding: "0 24px", display: "flex", gap: 0 }}>
        {(["builder", "prediction", "simulator"] as const).map(t => {
          const labels = { builder: "Match Builder", prediction: "Prediction", simulator: "Season Sim" };
          const active = tab === t;
          return (
            <button key={t} onClick={() => setTab(t)}
              style={{
                padding: "12px 18px", background: "transparent", border: "none",
                borderBottom: `2px solid ${active ? "#F59E0B" : "transparent"}`,
                color: active ? "#F59E0B" : "#475569", fontSize: 12,
                fontWeight: active ? 600 : 400, cursor: "pointer",
                transition: "all 0.15s", letterSpacing: "0.02em",
              }}
            >{labels[t]}</button>
          );
        })}
      </div>

      {/* ── CONTENT ── */}
      <div style={{ maxWidth: 1080, margin: "0 auto", padding: "20px 24px 60px" }}>

        {/* ── MATCH BUILDER ── */}
        {tab === "builder" && (
          <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: 14 }}>

            <Card>
              <SectionLabel>Teams</SectionLabel>
              <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
                <TeamSelect label="Team A" value={teamA} onChange={t => { setTeamA(t); setPrediction(null); }} exclude={teamB} />
                <div style={{ paddingBottom: 11, color: "#334155", fontSize: 13, fontWeight: 600, flexShrink: 0 }}>vs</div>
                <TeamSelect label="Team B" value={teamB} onChange={t => { setTeamB(t); setPrediction(null); }} exclude={teamA} />
              </div>
            </Card>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              <Card style={{ borderColor: `${cA}28` }}>
                <SectionLabel>{teamA} squad</SectionLabel>
                <PlayerChips team={teamA} selected={squadA} onToggle={toggleA} />
              </Card>
              <Card style={{ borderColor: `${cB}28` }}>
                <SectionLabel>{teamB} squad</SectionLabel>
                <PlayerChips team={teamB} selected={squadB} onToggle={toggleB} />
              </Card>
            </div>

            <Card>
              <SectionLabel>Context</SectionLabel>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <div>
                  <div style={{ fontSize: 10, color: "#475569", marginBottom: 6 }}>Venue</div>
                  <select
                    value={venue} onChange={e => setVenue(e.target.value)}
                    style={{ width: "100%", padding: "9px 12px", background: "#111720", border: "1px solid #1E2836", borderRadius: 8, color: "#F1F5F9", fontSize: 12, cursor: "pointer", outline: "none" }}
                  >
                    {VENUES.map(v => <option key={v} value={v} style={{ background: "#111720" }}>{v}</option>)}
                  </select>
                </div>
                <div>
                  <div style={{ fontSize: 10, color: "#475569", marginBottom: 6 }}>Toss Winner</div>
                  <div style={{ display: "flex", gap: 6 }}>
                    {[teamA, teamB, ""].map((t, i) => (
                      <button key={i} onClick={() => setTossWinner(t)}
                        style={{
                          flex: 1, padding: "9px 4px", borderRadius: 8, fontSize: 11, cursor: "pointer",
                          background: "transparent",
                          border: `1px solid ${tossWinner === t ? "#F59E0B" : "#1E2836"}`,
                          color: tossWinner === t ? "#F59E0B" : "#475569",
                          transition: "all 0.12s",
                        }}
                      >{t || "—"}</button>
                    ))}
                  </div>
                </div>
              </div>
            </Card>

            <div style={{ display: "flex", gap: 10 }}>
              <button onClick={handlePredict} disabled={predLoading}
                style={{
                  flex: 1, padding: "11px 0", borderRadius: 8, fontSize: 13, fontWeight: 600,
                  background: predLoading ? "rgba(245,158,11,0.4)" : "#F59E0B", border: "none",
                  color: "#0B0F14", cursor: predLoading ? "not-allowed" : "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                  transition: "background 0.15s",
                }}
              >
                {predLoading ? <><Spinner dark /> Analyzing…</> : "Predict Match"}
              </button>
              <button onClick={handleSimulate} disabled={simLoading}
                style={{
                  flex: 1, padding: "11px 0", borderRadius: 8, fontSize: 13, fontWeight: 600,
                  background: "transparent", border: `1px solid ${simLoading ? "#334155" : "#1E2836"}`,
                  color: simLoading ? "#334155" : "#94A3B8", cursor: simLoading ? "not-allowed" : "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                  transition: "all 0.15s",
                }}
                onMouseEnter={e => { if (!simLoading) { (e.currentTarget as HTMLElement).style.borderColor = "#F59E0B"; (e.currentTarget as HTMLElement).style.color = "#F59E0B"; } }}
                onMouseLeave={e => { if (!simLoading) { (e.currentTarget as HTMLElement).style.borderColor = "#1E2836"; (e.currentTarget as HTMLElement).style.color = "#94A3B8"; } }}
              >
                {simLoading ? <><Spinner /> Running…</> : "Season Simulator"}
              </button>
            </div>
          </div>
        )}

        {/* ── PREDICTION ── */}
        {tab === "prediction" && (
          <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {!prediction ? (
              <div style={{ padding: "60px 0", textAlign: "center" }}>
                <div style={{ color: "#334155", fontSize: 12, marginBottom: 12 }}>No prediction yet.</div>
                <button onClick={() => setTab("builder")}
                  style={{ padding: "8px 18px", borderRadius: 6, background: "transparent", border: "1px solid #1E2836", color: "#64748B", fontSize: 12, cursor: "pointer" }}
                >→ Match Builder</button>
              </div>
            ) : (
              <>
                {/* Win probability */}
                <Card>
                  <SectionLabel>{teamA} vs {teamB} · win probability</SectionLabel>
                  <ProbBar pA={prediction.win_probability_A} pB={prediction.win_probability_B} teamA={teamA} teamB={teamB} />
                  <div style={{ marginTop: 14, padding: "10px 14px", background: "#111720", borderRadius: 8, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 11, color: "#475569" }}>Predicted winner</span>
                    <span style={{ fontWeight: 700, color: prediction.win_probability_A > 0.5 ? cA : cB, fontSize: 13 }}>
                      {prediction.win_probability_A > 0.5 ? teamA : teamB}
                      <span style={{ fontWeight: 400, fontSize: 11, color: "#475569", marginLeft: 8 }}>
                        (<AnimNum value={Math.max(prediction.win_probability_A, prediction.win_probability_B) * 100} />%)
                      </span>
                    </span>
                  </div>
                </Card>

                {/* Squad strength */}
                {strengthA && strengthB && (
                  <Card>
                    <SectionLabel>Squad strength — {teamA} left · {teamB} right</SectionLabel>
                    <StrengthRow label="Batting" vA={strengthA.batting_unit} vB={strengthB.batting_unit} cA={cA} cB={cB} />
                    <StrengthRow label="Bowling" vA={strengthA.bowling_unit} vB={strengthB.bowling_unit} cA={cA} cB={cB} />
                    <StrengthRow label="All-round" vA={strengthA.allrounder_balance} vB={strengthB.allrounder_balance} cA={cA} cB={cB} />
                    <StrengthRow label="Total" vA={strengthA.total_strength} vB={strengthB.total_strength} cA={cA} cB={cB} />
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#334155", marginTop: 4 }}>
                      <span>Matched {strengthA.squad_matched}/{strengthA.squad_size} players</span>
                      <span>{strengthB.squad_matched}/{strengthB.squad_size} matched</span>
                    </div>
                  </Card>
                )}

                {/* Score breakdown */}
                <Card>
                  <SectionLabel>Score breakdown</SectionLabel>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
                    <AdjCell label="Base Diff" value={prediction.base_diff} />
                    <AdjCell label="Venue" value={prediction.venue_adjustment} />
                    <AdjCell label="Toss" value={prediction.toss_adjustment} />
                  </div>
                  <div style={{ marginTop: 10, padding: "9px 14px", background: "#111720", borderRadius: 8, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 10, color: "#475569", textTransform: "uppercase", letterSpacing: "0.1em" }}>Final score</span>
                    <span style={{ fontSize: 13, fontWeight: 700, color: "#E2E8F0" }}>
                      {((prediction.base_diff + prediction.venue_adjustment + prediction.toss_adjustment) * 100).toFixed(2)}
                    </span>
                  </div>
                </Card>
              </>
            )}
          </div>
        )}

        {/* ── SEASON SIMULATOR ── */}
        {tab === "simulator" && (
          <div className="fade-in" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {!simResults ? (
              <div style={{ padding: "60px 0", textAlign: "center" }}>
                <div style={{ color: "#334155", fontSize: 12, marginBottom: 14 }}>10,000 Monte Carlo simulations across all 70 IPL fixtures.</div>
                <button onClick={handleSimulate} disabled={simLoading}
                  style={{
                    padding: "10px 24px", borderRadius: 8, background: simLoading ? "rgba(245,158,11,0.4)" : "#F59E0B",
                    border: "none", color: "#0B0F14", fontSize: 13, fontWeight: 600,
                    cursor: simLoading ? "not-allowed" : "pointer",
                    display: "inline-flex", alignItems: "center", gap: 8,
                  }}
                >
                  {simLoading ? <><Spinner dark /> Running…</> : "Run Season Simulation"}
                </button>
              </div>
            ) : (
              <>
                <Card>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                    <SectionLabel>Title probability · 10,000 simulations</SectionLabel>
                    <button onClick={handleSimulate} disabled={simLoading}
                      style={{ padding: "5px 12px", borderRadius: 6, background: "transparent", border: "1px solid #1E2836", color: "#475569", fontSize: 11, cursor: simLoading ? "not-allowed" : "pointer", display: "flex", alignItems: "center", gap: 6 }}>
                      {simLoading ? <><Spinner /> running</> : "↻ re-run"}
                    </button>
                  </div>
                  {simResults.map(r => <SimBar key={r.team} result={r} maxProb={maxProb} />)}
                  <div style={{ marginTop: 10, fontSize: 10, color: "#1E2836" }}>
                    ↑ = playoff qualification %
                  </div>
                </Card>

                <Card>
                  <SectionLabel>Full standings — click headers to sort</SectionLabel>
                  <SimTable data={simResults} />
                </Card>
              </>
            )}
          </div>
        )}

      </div>
    </div>
  );
}