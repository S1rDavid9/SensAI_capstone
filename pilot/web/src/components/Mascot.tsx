// Sensei Ryu: the friendly dragon who guides learners (ryū means dragon; in
// East Asian tradition dragons stand for wisdom). Drawn as SVG so it works
// offline and scales on any screen. Moods change only the eyes and mouth, so
// children always recognise the same character.

export type Mood = "hello" | "thinking" | "cheering" | "gentle";

const SKIN = "#2bb3a3";
const SKIN_DARK = "#1f8f82";
const BELLY = "#fff1c7";
const WING = "#ff9f5a";
const BAND = "#e8453c";

export default function Mascot({ mood = "hello", size = 96 }: { mood?: Mood; size?: number }) {
  const happyEyes = mood === "hello" || mood === "cheering";
  return (
    <svg
      className={`mascot mascot-${mood}`}
      width={size}
      height={size}
      viewBox="0 0 100 100"
      role="img"
      aria-label="Sensei Ryu the dragon"
    >
      {/* tail */}
      <path d="M70 80 Q92 82 90 64 Q89 56 82 58" stroke={SKIN_DARK} strokeWidth="7" fill="none" strokeLinecap="round" />
      <path d="M80 54 l6 2 l-3 6 z" fill={WING} />
      {/* wings */}
      <path className="wing left" d="M28 62 Q10 50 14 36 Q22 46 32 50 Z" fill={WING} />
      <path className="wing right" d="M72 62 Q90 50 86 36 Q78 46 68 50 Z" fill={WING} />
      {/* body */}
      <ellipse cx="50" cy="72" rx="24" ry="21" fill={SKIN} />
      <ellipse cx="50" cy="76" rx="14" ry="14" fill={BELLY} />
      <path d="M40 72 h20 M39 78 h22 M41 84 h18" stroke="#f1d58a" strokeWidth="1.5" />
      {/* feet */}
      <ellipse cx="38" cy="92" rx="7" ry="4" fill={SKIN_DARK} />
      <ellipse cx="62" cy="92" rx="7" ry="4" fill={SKIN_DARK} />
      {/* spikes */}
      <path d="M38 18 l4 -8 l4 8 Z M47 15 l3 -9 l3 9 Z M55 18 l4 -8 l4 8 Z" fill={WING} />
      {/* horns */}
      <path d="M32 24 Q26 12 30 8 Q34 16 38 22 Z" fill="#ffd34d" />
      <path d="M68 24 Q74 12 70 8 Q66 16 62 22 Z" fill="#ffd34d" />
      {/* head */}
      <circle cx="50" cy="40" r="24" fill={SKIN} />
      {/* sensei headband */}
      <path d="M27 31 Q50 23 73 31 L73 37 Q50 29 27 37 Z" fill={BAND} />
      <path d="M73 33 q8 -2 11 3 M73 35 q7 3 9 9" stroke={BAND} strokeWidth="3" fill="none" strokeLinecap="round" />
      <circle cx="50" cy="30" r="2.6" fill="#fff" />
      {/* snout */}
      <ellipse cx="50" cy="52" rx="13" ry="8" fill="#5fd0c1" />
      <circle cx="46" cy="50" r="1.4" fill={SKIN_DARK} />
      <circle cx="54" cy="50" r="1.4" fill={SKIN_DARK} />
      {/* eyes */}
      {happyEyes ? (
        <>
          <path d="M36 43 Q41 37 46 43" stroke="#163b36" strokeWidth="3.2" fill="none" strokeLinecap="round" />
          <path d="M54 43 Q59 37 64 43" stroke="#163b36" strokeWidth="3.2" fill="none" strokeLinecap="round" />
        </>
      ) : (
        <>
          <circle cx="41" cy="42" r="6" fill="#fff" />
          <circle cx="59" cy="42" r="6" fill="#fff" />
          <circle cx={mood === "thinking" ? 42.5 : 41} cy={mood === "thinking" ? 40 : 43} r="3.2" fill="#163b36" />
          <circle cx={mood === "thinking" ? 60.5 : 59} cy={mood === "thinking" ? 40 : 43} r="3.2" fill="#163b36" />
        </>
      )}
      {/* mouth */}
      {mood === "cheering" ? (
        <path d="M43 56 Q50 63 57 56 Z" fill="#163b36" />
      ) : mood === "thinking" ? (
        <path d="M46 57 h7" stroke="#163b36" strokeWidth="2" strokeLinecap="round" />
      ) : (
        <path d="M44 56 Q50 60 56 56" stroke="#163b36" strokeWidth="2" fill="none" strokeLinecap="round" />
      )}
      {/* cheeks */}
      <circle cx="33" cy="50" r="3" fill="#ff7a9a" opacity="0.85" />
      <circle cx="67" cy="50" r="3" fill="#ff7a9a" opacity="0.85" />
    </svg>
  );
}
