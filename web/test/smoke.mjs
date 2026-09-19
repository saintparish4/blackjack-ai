// Node smoke test for the browser demo: loads the real WebAssembly module and
// the published model, plays thousands of rounds through the same API the page
// uses, and checks the coaching logic the page shows.
//
//   web/build.sh && node web/test/smoke.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import createBlackjack from "../wasm/blackjack.mjs";
import { formatUnits, judge, newSession, parseCard, qBars, recordDecision, recordRound } from "../js/coach.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const Module = await createBlackjack();
Module.FS.writeFile("/agent.qtable", readFileSync(join(here, "../model/agent.qtable")));
Module.FS.writeFile("/agent.meta", readFileSync(join(here, "../model/agent.meta"), "utf8"));

let checks = 0;
const check = (cond, msg) => { assert.ok(cond, msg); checks += 1; };

// ---- Loading ----
const empty = new Module.Table(1);
check(!empty.loadModel("/missing"), "a missing model reports failure instead of aborting");
const table = new Module.Table(12345);
check(table.loadModel("/agent"), "published model loads");
const idle = JSON.parse(table.state());
check(idle.dealt === false && idle.modelLoaded === true, "fresh table is undealt");

// ---- Watch the AI: the page's aiAct loop over many rounds ----
const ROUNDS = 20000;
let total = 0;
let decisions = 0;
let bookAgreement = 0;
let naturals = 0;
let splits = 0;
for (let r = 0; r < ROUNDS; r++) {
  let s = JSON.parse(table.deal());
  if (s.done) naturals += 1;
  let guard = 0;
  while (!s.done) {
    check(s.legal.length >= 2, "a live decision offers at least HIT and STAND");
    check(s.legal.includes(s.ai), `AI move ${s.ai} is legal`);
    check(s.legal.includes(s.book), `book move ${s.book} is legal`);
    check(Object.keys(s.q).length === s.legal.length, "one Q-value per legal action");
    check(s.dealer.hidden && s.dealer.cards.length === 1, "hole card stays hidden during play");
    if (s.ai === s.book) bookAgreement += 1;
    if (s.ai === "SPLIT") splits += 1;
    decisions += 1;
    s = JSON.parse(table.aiAct());
    check(++guard < 30, "a round always finishes");
  }
  check(!s.dealer.hidden && s.dealer.cards.length >= 2, "hole card revealed at showdown");
  check(s.hands.every((h) => typeof h.reward === "number"), "every hand settles");
  const sum = s.hands.reduce((a, h) => a + h.reward, 0);
  check(Math.abs(sum - s.roundReward) < 1e-9, "round reward is the sum of hand rewards");
  total += s.roundReward;
}
const ev = total / ROUNDS;
// The C++ evaluator measures this model at about -0.01 per round. Over 20k
// rounds the standard error is about 0.008, so +/-0.035 is over 4 sigma.
check(ev > -0.045 && ev < 0.025, `EV per round in the browser build is ${ev.toFixed(4)}`);
check(bookAgreement / decisions > 0.9, `AI agrees with the book on ${(100 * bookAgreement / decisions).toFixed(1)}% of live decisions`);
check(naturals > 0 && splits > 0, "naturals and splits both occur");

// ---- Play mode: illegal actions are ignored, legal ones apply ----
let s = JSON.parse(table.deal());
while (s.done) s = JSON.parse(table.deal());
const unchanged = (json) => { assert.deepEqual(JSON.parse(json), s); return true; };
check(unchanged(table.act("FLY")), "unknown action is ignored");
if (!s.legal.includes("SPLIT")) check(unchanged(table.act("SPLIT")), "illegal SPLIT is ignored");
const after = JSON.parse(table.act("STAND"));
check(after.done || after.current === 1, "STAND ends the hand");

// ---- Same seed, same shoe ----
const a = new Module.Table(777), b = new Module.Table(777);
a.loadModel("/agent"); b.loadModel("/agent");
for (let i = 0; i < 50; i++) check(a.deal() === b.deal(), "a seed replays the same shoe");

// ---- Coaching logic ----
const decision = { ai: "STAND", book: "STAND", trained: true, situation: "hard 16 vs 6",
  q: { HIT: -0.35, STAND: -0.15, DOUBLE: -0.9 } };
const same = judge(decision, "STAND");
check(same.match && same.verdict === "match" && same.cost === 0, "matching the AI costs nothing");
const off = judge(decision, "HIT");
check(off.verdict === "mistake" && Math.abs(off.cost - 0.2) < 1e-12, "cost is the Q gap, in bets");
const close = judge({ ...decision, q: { HIT: -0.16, STAND: -0.15, DOUBLE: -0.9 } }, "HIT");
check(close.verdict === "close", "a sub-0.02 gap is a close call");
check(judge({ ...decision, trained: false }, "HIT").cost === null, "untrained states aren't priced");
const ss = newSession();
recordDecision(ss, decision, off);
recordRound(ss, { roundReward: -1 });
check(ss.decisions === 1 && ss.matches === 0 && ss.hands === 1 && ss.net === -1, "session totals");
check(parseCard("10H").rank === "10" && parseCard("10H").red, "card codes parse");
check(formatUnits(1.5) === "+1.5" && formatUnits(-1) === "−1" && formatUnits(0) === "0", "unit formatting");
check(qBars(decision)[0].action === "STAND", "Q bars sort best first");

console.log(`web smoke test: ${checks} checks passed · ${ROUNDS} rounds · EV ${ev.toFixed(4)}/round · ` +
  `AI agrees with the book on ${(100 * bookAgreement / decisions).toFixed(1)}% of ${decisions} decisions`);
