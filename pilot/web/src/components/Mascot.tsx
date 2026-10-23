// Sena the owl: the friendly guide on the learner screen. Drawn as SVG so it
// works offline and scales on any screen. Moods change the eyes, beak and
// wings only; the shapes stay the same so children recognise her.

export type Mood = "hello" | "thinking" | "cheering" | "gentle";

export default function Mascot({ mood = "hello", size = 96 }: { mood?: Mood; size?: number }) {
  const eyeY = mood === "thinking" ? 40 : 42;
  const happy = mood === "cheering" || mood === "hello";
  return (
    <svg
      className={`mascot mascot-${mood}`}
      width={size}
      height={size}
      viewBox="0 0 100 100"
      role="img"
      aria-label="Sena the owl"
    >
      {/* wings */}
      <ellipse className="wing left" cx="20" cy="62" rx="12" ry="20" fill="#7c5cd6" />
      <ellipse className="wing right" cx="80" cy="62" rx="12" ry="20" fill="#7c5cd6" />
      {/* body */}
      <ellipse cx="50" cy="58" rx="32" ry="34" fill="#9b7ff0" />
      <ellipse cx="50" cy="68" rx="20" ry="20" fill="#e9e1ff" />
      {/* ear tufts */}
      <path d="M26 30 L32 14 L40 28 Z" fill="#7c5cd6" />
      <path d="M74 30 L68 14 L60 28 Z" fill="#7c5cd6" />
      {/* eyes */}
      <circle cx="37" cy={eyeY} r="11" fill="#fff" />
      <circle cx="63" cy={eyeY} r="11" fill="#fff" />
      {happy ? (
        <>
          <path d={`M31 ${eyeY + 1} Q37 ${eyeY - 6} 43 ${eyeY + 1}`} stroke="#2b2350" strokeWidth="3.5" fill="none" strokeLinecap="round" />
          <path d={`M57 ${eyeY + 1} Q63 ${eyeY - 6} 69 ${eyeY + 1}`} stroke="#2b2350" strokeWidth="3.5" fill="none" strokeLinecap="round" />
        </>
      ) : (
        <>
          <circle cx={mood === "thinking" ? 40 : 37} cy={eyeY + 1} r="5" fill="#2b2350" />
          <circle cx={mood === "thinking" ? 66 : 63} cy={eyeY + 1} r="5" fill="#2b2350" />
        </>
      )}
      {/* beak */}
      <path d="M45 52 L55 52 L50 60 Z" fill="#f4a23a" />
      {/* feet */}
      <path d="M40 91 l-4 5 M40 91 l0 6 M40 91 l4 5 M60 91 l-4 5 M60 91 l0 6 M60 91 l4 5"
        stroke="#f4a23a" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}
