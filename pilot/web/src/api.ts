// Small fetch wrapper plus a server clock, so timestamps from the learner's
// laptop and the observers' phones can be lined up afterwards.

let clockOffsetMs = 0;

export async function syncClock(): Promise<void> {
  const samples: { offset: number; rtt: number }[] = [];
  for (let i = 0; i < 3; i++) {
    const t0 = Date.now();
    const res = await fetch("/api/time");
    const t1 = Date.now();
    const { server_ms } = (await res.json()) as { server_ms: number };
    samples.push({ offset: server_ms - (t0 + t1) / 2, rtt: t1 - t0 });
  }
  samples.sort((a, b) => a.rtt - b.rtt);
  clockOffsetMs = Math.round(samples[0].offset);
}

/** Current time on the server's clock, estimated locally. */
export function serverNow(): number {
  return Date.now() + clockOffsetMs;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T>(path: string, body?: unknown, headers: Record<string, string> = {}): Promise<T> {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") message = data.detail;
    } catch {
      /* keep the generic message */
    }
    throw new ApiError(res.status, message);
  }
  return res.json() as Promise<T>;
}

export function researcherHeaders(key: string) {
  return { "X-Researcher-Key": key };
}

export function load(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function save(key: string, value: string | null) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* private mode: just don't remember */
  }
}
