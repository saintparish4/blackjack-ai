// Pure coaching logic: no DOM, no WebAssembly. Shared by the page (app.mjs)
// and the Node smoke test, so what the tests check is what visitors see.

export const ACTION_LABELS = {
  HIT: "Hit",
  STAND: "Stand",
  DOUBLE: "Double",
  SPLIT: "Split",
  SURRENDER: "Surrender",
};

export const ACTION_KEYS = { HIT: "h", STAND: "s", DOUBLE: "d", SPLIT: "p", SURRENDER: "r" };

const SUITS = { S: "♠", H: "♥", D: "♦", C: "♣" };

/** "10H" -> { rank: "10", suit: "♥", red: true } */
export function parseCard(code) {
  const suit = code.slice(-1);
  return { rank: code.slice(0, -1), suit: SUITS[suit] ?? suit, red: suit === "H" || suit === "D" };
}

/** Signed amount in bets, e.g. "+1.5", "−1", "0". */
export function formatUnits(x, digits = 2) {
  const rounded = Number(x.toFixed(digits));
  if (rounded === 0) return "0";
  const body = Math.abs(rounded).toString();
  return (rounded > 0 ? "+" : "−") + body;
}

/**
 * Grade one decision against the agent's own estimates.
 *
 * `decision` is the state the Table reported before the move: legal actions,
 * Q-values, the agent's choice, the book's play. The estimated cost is the gap
 * between the agent's value for its move and for yours, in bets. It's the
 * agent's estimate, not the truth, and it's only offered when every legal
 * action here was actually trained.
 */
export function judge(decision, yourAction) {
  const { ai, book, q, trained } = decision;
  const match = yourAction === ai;
  const cost = trained && !match ? Math.max(0, q[ai] - q[yourAction]) : 0;

  let verdict;
  if (match) verdict = "match";
  else if (!trained) verdict = "unknown";
  else if (cost < 0.02) verdict = "close";
  else verdict = "mistake";

  return {
    you: yourAction,
    ai,
    book,
    match,
    matchesBook: yourAction === book,
    cost: trained ? cost : null,
    verdict,
    // The agent and the book can disagree on marginal cells; say so plainly
    // rather than pretend the agent is the last word.
    aiDisagreesWithBook: ai !== book,
  };
}

/** One-line headline for a judged decision. */
export function headline(j) {
  const ai = ACTION_LABELS[j.ai];
  switch (j.verdict) {
    case "match":
      return `Same as the AI: ${ai}.`;
    case "close":
      return `The AI would ${ai.toLowerCase()}, but it's a close call (about ${j.cost.toFixed(3)} of a bet).`;
    case "mistake":
      return `The AI would ${ai.toLowerCase()}. By its estimate that cost you about ${j.cost.toFixed(2)} of a bet.`;
    default:
      return `The AI would ${ai.toLowerCase()}, but it never tried every option here, so it can't price the difference.`;
  }
}

/** Running totals for a session. */
export function newSession() {
  return { hands: 0, net: 0, decisions: 0, matches: 0, costGivenUp: 0, history: [] };
}

export function recordDecision(session, decision, judged) {
  session.decisions += 1;
  if (judged.match) session.matches += 1;
  if (judged.cost) session.costGivenUp += judged.cost;
  session.history.unshift({ situation: decision.situation, ...judged });
  if (session.history.length > 50) session.history.pop();
}

export function recordRound(session, state) {
  session.hands += 1;
  session.net += state.roundReward;
}

/** Sorted Q-values for display, best first, with bar widths in [0, 1]. */
export function qBars(decision) {
  const entries = Object.entries(decision.q).map(([action, value]) => ({ action, value }));
  entries.sort((a, b) => b.value - a.value);
  // Scale against the widest swing a single decision can have: ±2 when
  // doubling, ±1 otherwise.
  const span = Math.max(1, ...entries.map((e) => Math.abs(e.value)));
  return entries.map((e) => ({ ...e, width: Math.abs(e.value) / span, positive: e.value >= 0 }));
}
