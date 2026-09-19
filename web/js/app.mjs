import createBlackjack from "../wasm/blackjack.mjs";
import {
  ACTION_KEYS, ACTION_LABELS, formatUnits, headline, judge, newSession, parseCard, qBars,
  recordDecision, recordRound,
} from "./coach.mjs";

const $ = (id) => document.getElementById(id);

const MODE_HELP = {
  play: "Play your hand. After each move the AI tells you what it would have done, and what the difference cost by its estimate.",
  coach: "The AI shows its move and its reasoning before you decide. Follow it or overrule it.",
  watch: "Deal, then step through the AI's decisions or switch on auto-play. Every move shows the Q-values it came from.",
};
const CHALLENGE_HANDS = 10;

const ui = {
  mode: "play",
  sessions: { play: newSession(), coach: newSession(), watch: newSession() },
  state: null,
  lastJudged: null,
  autoTimer: null,
};

// ---- Boot ----------------------------------------------------------------

const params = new URLSearchParams(location.search);
const seed = params.has("seed")
  ? Number(params.get("seed")) >>> 0
  : crypto.getRandomValues(new Uint32Array(1))[0];

const Module = await createBlackjack();
const [qtable, meta] = await Promise.all([
  fetch("model/agent.qtable").then((r) => r.arrayBuffer()),
  fetch("model/agent.meta").then((r) => r.text()),
]);
Module.FS.writeFile("/agent.qtable", new Uint8Array(qtable));
Module.FS.writeFile("/agent.meta", meta);
const table = new Module.Table(seed);
if (!table.loadModel("/agent")) {
  $("loading").textContent = "Couldn't load the trained model.";
  throw new Error("model failed to load");
}
$("loading").hidden = true;
ui.state = JSON.parse(table.state());

// ---- Actions -------------------------------------------------------------

function deal() {
  ui.lastJudged = null; // last hand's verdict would read as advice on this one
  update(table.deal());
}

function playerAction(action) {
  const before = ui.state;
  if (before.done || !before.legal.includes(action)) return;
  const judged = judge(before, action);
  recordDecision(session(), before, judged);
  ui.lastJudged = { decision: before, judged };
  update(table.act(action));
}

function aiStep() {
  const before = ui.state;
  if (!before.dealt || before.done) return deal();
  recordDecision(session(), before, judge(before, before.ai));
  update(table.aiAct());
}

/** Take the Table's new state. Callers only deal, or act on a live round,
 *  so a finished state here is always a round that just ended (a natural
 *  settles on the deal itself) and is counted exactly once. */
function update(json) {
  ui.state = JSON.parse(json);
  if (ui.state.dealt && ui.state.done) recordRound(session(), ui.state);
  render();
}

function session() {
  return ui.sessions[ui.mode];
}

function setMode(mode) {
  ui.mode = mode;
  stopAuto();
  document.querySelectorAll(".modes button").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  ui.lastJudged = null;
  render();
}

// ---- Rendering -----------------------------------------------------------

function cardEl(code) {
  const { rank, suit, red } = parseCard(code);
  const el = document.createElement("div");
  el.className = "card-face" + (red ? " red" : "");
  el.setAttribute("aria-label", `${rank} of ${suit}`);
  el.innerHTML = `<span class="rank">${rank}${suit}</span><span class="pip">${suit}</span><span class="rank bottom">${rank}${suit}</span>`;
  return el;
}

function totalText(total, soft) {
  return soft && total <= 21 ? `soft ${total}` : String(total);
}

function renderTable() {
  const s = ui.state;
  const dealerCards = $("dealer-cards");
  const hands = $("hands");
  dealerCards.replaceChildren();
  hands.replaceChildren();
  $("dealer-total").textContent = "";
  $("banner").textContent = "";

  if (!s.dealt) {
    $("banner").textContent = "Press Deal to start.";
    return;
  }

  s.dealer.cards.forEach((c) => dealerCards.append(cardEl(c)));
  if (s.dealer.hidden) {
    const back = document.createElement("div");
    back.className = "card-back";
    back.setAttribute("aria-label", "face-down card");
    dealerCards.append(back);
    $("dealer-total").textContent = `(showing ${totalText(s.dealer.total, s.dealer.soft)})`;
  } else {
    $("dealer-total").textContent = `(${totalText(s.dealer.total, s.dealer.soft)})`;
  }

  s.hands.forEach((h, i) => {
    const box = document.createElement("div");
    box.className = "hand" + (!s.done && i === s.current && s.hands.length > 1 ? " current" : "");
    const label = document.createElement("div");
    label.className = "seat-label";
    label.textContent = `${s.hands.length > 1 ? `Hand ${i + 1}` : "You"} (${totalText(h.total, h.soft)})${h.doubled ? " · doubled" : ""}`;
    const cards = document.createElement("div");
    cards.className = "cards";
    h.cards.forEach((c) => cards.append(cardEl(c)));
    box.append(label, cards);
    if (s.done && h.outcome) {
      const r = document.createElement("div");
      r.className = "result " + (h.reward > 0 ? "win" : h.reward < 0 ? "loss" : "push");
      r.textContent = `${h.outcome} · ${formatUnits(h.reward)}`;
      box.append(r);
    }
    hands.append(box);
  });

  if (s.done) {
    const net = formatUnits(s.roundReward);
    $("banner").textContent =
      s.roundReward > 0 ? `You win ${net}` : s.roundReward < 0 ? `Dealer wins (${net})` : "Push";
    if (ui.mode === "watch") $("banner").textContent = `AI: ${net} this hand`;
  }
}

function renderControls() {
  const s = ui.state;
  const watch = ui.mode === "watch";
  const actions = $("actions");
  actions.replaceChildren();
  if (!watch) {
    for (const action of ["HIT", "STAND", "DOUBLE", "SPLIT"]) {
      const b = document.createElement("button");
      b.innerHTML = `${ACTION_LABELS[action]} <kbd>${ACTION_KEYS[action].toUpperCase()}</kbd>`;
      b.disabled = !s.dealt || s.done || !s.legal.includes(action);
      if (ui.mode === "coach" && !b.disabled && s.ai === action) b.classList.add("suggested");
      b.addEventListener("click", () => playerAction(action));
      actions.append(b);
    }
  }
  $("deal").disabled = s.dealt && !s.done;
  $("deal").hidden = watch && s.dealt && !s.done;
  $("step").hidden = !watch;
  $("step").disabled = false;
  $("auto-wrap").hidden = !watch;
}

function aiViewHtml(d) {
  const bars = qBars(d)
    .map((b, i) => `<li class="qbar${i === 0 ? " best" : ""}">
        <span class="name">${ACTION_LABELS[b.action]}</span>
        <span class="track"><span class="fill ${b.positive ? "pos" : "neg"}" style="width:${(b.width * 100).toFixed(1)}%"></span></span>
        <span class="val">${formatUnits(b.value, 3)}</span></li>`)
    .join("");
  const bookNote = d.book === d.ai
    ? `<p class="book-note">Basic strategy agrees: ${ACTION_LABELS[d.book].toLowerCase()}.</p>`
    : `<p class="book-note">Basic strategy says <strong>${ACTION_LABELS[d.book].toLowerCase()}</strong> here. This is one of the close calls the AI still gets wrong. ${d.bookReason}</p>`;
  return `<div class="situation">${d.situation}</div>
    <ul class="qbars">${bars}</ul>
    <p class="pick">AI plays <strong>${ACTION_LABELS[d.ai]}</strong><span class="tag">${d.confidence} confidence</span></p>
    <p class="why">${d.reason}</p>${bookNote}
    <p class="muted" style="font-size:.85rem;margin:6px 0 0">Q-values are the AI's expected result of each move, in bets.</p>`;
}

function renderPanel() {
  const s = ui.state;
  const body = $("ai-body");
  const live = s.dealt && !s.done;

  if (live && ui.mode !== "play") {
    body.className = "";
    body.innerHTML = aiViewHtml(s);
  } else if (ui.lastJudged && ui.mode !== "watch") {
    body.className = "";
    body.innerHTML = aiViewHtml(ui.lastJudged.decision);
  } else if (live) {
    body.className = "muted";
    body.textContent = "Make your move. The AI's take appears as soon as you act.";
  } else {
    body.className = "muted";
    body.textContent = ui.mode === "watch" ? "Deal to watch the AI play." : "Deal a hand to see what the AI thinks.";
  }

  const fb = $("feedback-card");
  if (ui.lastJudged && ui.mode !== "watch") {
    const j = ui.lastJudged.judged;
    fb.hidden = false;
    $("feedback").innerHTML = `<p class="verdict ${j.verdict}">${headline(j)}</p>
      <p class="muted">You: ${ACTION_LABELS[j.you]} · AI: ${ACTION_LABELS[j.ai]} · Basic strategy: ${ACTION_LABELS[j.book]}</p>`;
  } else {
    fb.hidden = true;
  }
}

function renderStats() {
  const ss = session();
  $("s-hands").textContent = ss.hands;
  $("s-net").textContent = formatUnits(ss.net);
  $("s-match").textContent = ss.decisions ? `${ss.matches}/${ss.decisions}` : "–";
  $("s-cost").textContent = ss.costGivenUp ? ss.costGivenUp.toFixed(2) : "0";
  if (ui.mode === "watch") {
    $("s-match").textContent = "–";
    $("s-cost").textContent = "–";
  }

  const ch = $("challenge");
  ch.classList.remove("done");
  if (ui.mode === "watch") {
    ch.textContent = `Seed ${seed}: add ?seed=${seed} to the URL to replay this shoe.`;
  } else if (ss.hands < CHALLENGE_HANDS) {
    ch.textContent = `Play ${CHALLENGE_HANDS} hands for a scorecard (${ss.hands}/${CHALLENGE_HANDS}).`;
  } else {
    ch.classList.add("done");
    ch.textContent = `Scorecard: over ${ss.hands} hands you matched the AI on ${ss.matches} of ${ss.decisions} decisions` +
      ` and gave up about ${ss.costGivenUp.toFixed(2)} bets of expected value by its estimate.` +
      ` For scale, the house edge is about 0.005 per hand.`;
  }

  const list = $("history");
  list.replaceChildren();
  for (const h of ss.history.slice(0, 20)) {
    const li = document.createElement("li");
    const mark = h.match ? "✓" : h.verdict === "close" ? "≈" : "✗";
    li.innerHTML = `<span>${mark} ${h.situation}: ${ACTION_LABELS[h.you].toLowerCase()}${h.match ? "" : ` (AI: ${ACTION_LABELS[h.ai].toLowerCase()})`}</span>
      <span class="cost">${h.cost ? "−" + h.cost.toFixed(2) : ""}</span>`;
    list.append(li);
  }
}

function render() {
  $("mode-help").textContent = MODE_HELP[ui.mode];
  renderTable();
  renderControls();
  renderPanel();
  renderStats();
}

// ---- Auto-play (watch mode) ---------------------------------------------

function startAuto() {
  stopAuto();
  ui.autoTimer = setInterval(aiStep, 800);
}

function stopAuto() {
  if (ui.autoTimer) clearInterval(ui.autoTimer);
  ui.autoTimer = null;
  $("auto").checked = false;
}

// ---- Wiring --------------------------------------------------------------

document.querySelectorAll(".modes button").forEach((b) =>
  b.addEventListener("click", () => setMode(b.dataset.mode)));
$("deal").addEventListener("click", deal);
$("step").addEventListener("click", aiStep);
$("auto").addEventListener("change", (e) => (e.target.checked ? startAuto() : stopAuto()));

document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return;
  const key = e.key.toLowerCase();
  if (key === "n" && (!ui.state.dealt || ui.state.done)) return deal();
  if (ui.mode === "watch") {
    if (key === " ") { e.preventDefault(); aiStep(); }
    return;
  }
  const action = Object.keys(ACTION_KEYS).find((a) => ACTION_KEYS[a] === key);
  if (action) playerAction(action);
});

render();
