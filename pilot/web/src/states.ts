// The behavioural states SensAI works with. Observers code these; the
// provisional rules (and later the Stage 1 classifier) predict them.

export const OBSERVER_CODES = [
  { code: "on_task", label: "On task", hint: "Working steadily, attending to the question", color: "var(--on-task)" },
  { code: "rushing", label: "Rushing", hint: "Answering very fast, guessing, not reading", color: "var(--rushing)" },
  { code: "struggling", label: "Struggling", hint: "Slow, getting answers wrong, working hard at it", color: "var(--struggling)" },
  { code: "stuck", label: "Stuck", hint: "Long pause, not trying, staring at the question", color: "var(--stuck)" },
  { code: "off_task", label: "Off task", hint: "Talking, looking away, doing something else", color: "var(--off-task)" },
  { code: "cant_see", label: "Can't see", hint: "Learner not visible or screen blocked", color: "var(--muted)" },
] as const;

export const STATE_LABEL: Record<string, string> = {
  on_task: "On task",
  rushing: "Rushing",
  struggling: "Struggling",
  stuck: "Stuck",
  off_task: "Off task",
  cant_see: "Can't see",
};

export const CHECKIN_CHOICES = [
  { choice: "easy", face: "😎", label: "Too easy" },
  { choice: "good", face: "🙂", label: "Going well" },
  { choice: "hard", face: "😓", label: "It's hard" },
  { choice: "lost", face: "😕", label: "I'm lost" },
] as const;
