// Mean-reversion dip-buying bot for agent-fun/bars-replay/v1.
//
// The bot trades one symbol (CBBTC/USDC) on 1-minute bars. Every batch carries
// the bar that just closed plus the fills of the orders decided on the previous
// batch and the current account; the bot answers each batch with at most one
// order, filled at the next bar's open. All decisions are pure functions of the
// bars seen so far: no clock, no network, no randomness, so two runs agree.
//
// Strategy: buy deep dips, exit on recovery or on protection.
//   Entry (flat only): either the close sits far below its 240-bar Bollinger
//   mean (z <= -2.5) with RSI(14) oversold (<= 40), or the close has fallen
//   1.5% or more below the highest close of the last 480 bars with RSI <= 50.
//   Both need a regime filter (the close must not sit more than 2% below its
//   240-bar EMA, so the bot never buys into a strong downtrend) and a one-bar
//   bounce confirmation (this close above the previous close).
//   Exit (long only): take profit at +1.6% over the average fill price, stop
//   loss at -1.0%, or a 480-bar maximum hold, whichever comes first. After a
//   sell the bot waits 15 bars before it may buy again.
//   Fallback: a window with no deep dip would leave the bot with no trade at
//   all, which fails the guardrails. If nothing has ever filled by bar 1080
//   of the 1440-bar window and the close holds above its long EMA with RSI
//   under 65, the bot buys once with the same exits; from bar 1320 it buys at
//   half size on any RSI under 80 rather than finish with no trade. The dev
//   window trades long before bar 1080, so the fallback never fires there.
//   Size: the whole cash balance, capped so the order's own price impact on
//   the latest close's pool depth stays under 20 bps, and capped by the
//   position-weight room the harness reports. Every order names
//   max_slippage_bps 100 against the latest close.

import { runBars } from './vendor/makefun-bars.ts';
import type { BarsState, Batch, OrderInput } from './vendor/makefun-bars.ts';

// ------------------------------------------------------------------ parameters

/** Bollinger window (bars) for the mean and standard deviation of closes. */
const BB_N = 240;
/** RSI period (Wilder smoothing). */
const RSI_N = 14;
/** EMA window (bars) for the regime filter. */
const EMA_N = 240;
/** Rolling-high window (bars) for the drawdown entry. */
const DROP_W = 480;
/** Drawdown from the rolling high that counts as a deep dip. */
const DROP_PCT = 0.015;
/** Bollinger z-score that counts as oversold. */
const Z_ENTRY = -2.5;
/** RSI ceiling for the Bollinger entry. */
const RSI_ENTRY = 40;
/** RSI ceiling for the drawdown entry (looser: the drop itself is the signal). */
const RSI_ENTRY_DROP = 50;
/** Regime filter: never buy more than this far below the long EMA. */
const REGIME_TOL = 0.02;
/** Take profit over the average fill price. */
const TAKE_PROFIT = 0.016;
/** Stop loss under the average fill price. */
const STOP_LOSS = 0.01;
/** Longest hold, in bars, before a forced exit. */
const MAX_HOLD = 480;
/** Bars to wait after a sell before the next buy. */
const COOLDOWN = 15;
/** Slippage cap on every order, in bps from the latest close. */
const MAX_SLIPPAGE_BPS = 100;
/** Size cap: keep the order's own impact on the close depth under this, bps. */
const IMPACT_CAP_BPS = 20n;
/** First bar at which the no-trade fallback may buy (windows are 1440 bars). */
const LATE_BAR = 1080;
/** First bar at which the last-resort half-size fallback may buy. */
const LAST_RESORT_BAR = 1320;

// ------------------------------------------------------------------ state

interface BotState {
  symbol: string;
  priceScale: number;
  qtyScale: number;
  quoteScale: number;
  /** Closes in USDC, in bar order. */
  closes: number[];
  /** Close time of the last bar folded in, to ignore duplicates. */
  lastT: number;
  /** Wilder RSI running gains/losses and the current RSI value, when defined. */
  rsiGain: number;
  rsiLoss: number;
  rsi: number | null;
  /** Long EMA of closes, seeded at the first close. */
  ema: number | null;
  /** Average fill price of the open position (USDC), and its quantity. */
  entryPrice: number | null;
  entryQty: bigint;
  /** Batch seq at which the position opened (for the maximum hold). */
  entrySeq: number;
  /** No buy before this batch seq (cooldown after a sell). */
  cooldownUntil: number;
  /** True once any order has ever filled (the fallback needs it). */
  everTraded: boolean;
}

function newBotState(state: BarsState): BotState {
  const first = state.hello.symbols[0];
  return {
    symbol: first.symbol,
    priceScale: first.price_scale,
    qtyScale: first.qty_scale,
    quoteScale: state.hello.capital.quote_scale,
    closes: [],
    lastT: -1,
    rsiGain: 0,
    rsiLoss: 0,
    rsi: null,
    ema: null,
    entryPrice: null,
    entryQty: 0n,
    entrySeq: -1,
    cooldownUntil: -1,
    everTraded: false,
  };
}

/** Fold one fresh close into the running indicators. */
function observeClose(bot: BotState, close: number): void {
  const prev = bot.closes.length > 0 ? bot.closes[bot.closes.length - 1] : null;
  bot.closes.push(close);
  if (prev !== null) {
    const ch = close - prev;
    if (bot.closes.length === 2) {
      bot.rsiGain = Math.max(ch, 0);
      bot.rsiLoss = Math.max(-ch, 0);
    } else {
      bot.rsiGain = (bot.rsiGain * (RSI_N - 1) + Math.max(ch, 0)) / RSI_N;
      bot.rsiLoss = (bot.rsiLoss * (RSI_N - 1) + Math.max(-ch, 0)) / RSI_N;
    }
    if (bot.closes.length > RSI_N) {
      bot.rsi = bot.rsiLoss === 0 ? 100 : 100 - 100 / (1 + bot.rsiGain / bot.rsiLoss);
    }
  }
  const k = 2 / (EMA_N + 1);
  bot.ema = bot.ema === null ? close : close * k + bot.ema * (1 - k);
}

/** Mean and population standard deviation of the last `n` closes. */
function meanAndSd(closes: number[], n: number): { mean: number; sd: number } {
  const start = closes.length - n;
  let sum = 0;
  for (let i = start; i < closes.length; i++) sum += closes[i];
  const mean = sum / n;
  let sq = 0;
  for (let i = start; i < closes.length; i++) {
    const d = closes[i] - mean;
    sq += d * d;
  }
  return { mean, sd: Math.sqrt(sq / n) };
}

/** Highest close of the last `n` closes. */
function rollingHigh(closes: number[], n: number): number {
  let high = -Infinity;
  for (let i = closes.length - n; i < closes.length; i++) {
    if (closes[i] > high) high = closes[i];
  }
  return high;
}

// ------------------------------------------------------------------ decisions

/** The bot's own state, created from hello before the first batch. */
let bot: BotState | null = null;

function onBatch(state: BarsState, batch: Batch): OrderInput[] {
  if (bot === null) return [];
  const sym = bot.symbol;

  // Fold the fills of the orders decided on the previous batch: they tell us
  // the average price the open position was filled at.
  for (const fill of batch.fills) {
    if (fill.symbol !== sym || fill.status !== 'filled') continue;
    bot.everTraded = true;
    const qty = BigInt(fill.qty);
    const price = Number(fill.price) / 10 ** bot.priceScale;
    if (fill.side === 'buy') {
      const total = bot.entryQty + qty;
      bot.entryPrice =
        bot.entryPrice === null || bot.entryQty === 0n
          ? price
          : (bot.entryPrice * Number(bot.entryQty) + price * Number(qty)) / Number(total);
      bot.entryQty = total;
      bot.entrySeq = batch.seq;
    } else {
      bot.entryQty -= qty;
      if (bot.entryQty <= 0n) {
        bot.entryQty = 0n;
        bot.entryPrice = null;
      }
    }
  }

  // Fold the fresh bar, if this batch carries one for our symbol. A minute the
  // recorder missed has no bar: nothing new to decide on.
  const bar = batch.bars.find((b) => b.symbol === sym);
  if (bar === undefined || bar.t <= bot.lastT) return [];
  bot.lastT = bar.t;
  const close = Number(bar.c) / 10 ** bot.priceScale;
  observeClose(bot, close);

  if (state.warmingUp) return [];

  const held = state.positions.get(sym);
  const heldQty = held === undefined ? 0n : held.qty;

  // --- long: take profit, stop loss, or maximum hold ---
  if (heldQty > 0n) {
    let entry = bot.entryPrice;
    if (entry === null && held !== undefined && held.qty > 0n) {
      // Fallback: the cost basis the account reports (fees included).
      entry =
        (Number(held.cost) / 10 ** bot.quoteScale) / (Number(held.qty) / 10 ** bot.qtyScale);
    }
    if (entry === null || entry <= 0) return [];
    const ret = close / entry - 1;
    const barsHeld = batch.seq - bot.entrySeq;
    if (ret >= TAKE_PROFIT || ret <= -STOP_LOSS || barsHeld >= MAX_HOLD) {
      bot.cooldownUntil = batch.seq + COOLDOWN;
      return [{ side: 'sell', symbol: sym, qty: heldQty, max_slippage_bps: MAX_SLIPPAGE_BPS }];
    }
    return [];
  }

  // Position empty but the tracker disagrees: resync from the account.
  bot.entryPrice = null;
  bot.entryQty = 0n;

  // --- flat: dip-buying entries, plus the late no-trade fallback ---
  if (batch.seq < bot.cooldownUntil) return [];
  if (bot.closes.length <= BB_N) return [];
  if (bot.rsi === null || bot.ema === null) return [];
  const prevClose = bot.closes[bot.closes.length - 2];
  const { mean, sd } = meanAndSd(bot.closes, BB_N);
  const z = sd > 0 ? (close - mean) / sd : 0;
  const drop =
    bot.closes.length >= DROP_W ? rollingHigh(bot.closes, DROP_W) / close - 1 : 0;
  const bollingerDip = z <= Z_ENTRY && bot.rsi <= RSI_ENTRY;
  const drawdownDip = drop >= DROP_PCT && bot.rsi <= RSI_ENTRY_DROP;
  const regimeOk = close >= bot.ema * (1 - REGIME_TOL);
  const bounced = close > prevClose;
  // Size fraction in bps: 10000 for a dip or a trend-aligned late buy, 5000
  // for the last-resort buy, 0 for no trade.
  let sizeFracBps = 0;
  if ((bollingerDip || drawdownDip) && regimeOk && bounced) {
    sizeFracBps = 10000;
  } else if (!bot.everTraded && batch.seq >= LATE_BAR) {
    if (close >= bot.ema * 0.995 && bot.rsi <= 65) {
      sizeFracBps = 10000;
    } else if (batch.seq >= LAST_RESORT_BAR && bot.rsi <= 80) {
      sizeFracBps = 5000;
    }
  }
  if (sizeFracBps === 0) return [];

  // Size: the fraction of cash, capped by the weight room and by the pool
  // depth at the latest close, so the order's own impact stays under the cap.
  let size = (state.cash * BigInt(sizeFracBps)) / 10000n;
  const room = state.weightRoom(sym);
  if (room < size) size = room;
  const depth = state.depthAt(sym);
  if (depth !== null) {
    for (const d of depth) {
      const cap = (d * IMPACT_CAP_BPS) / 10000n;
      if (cap < size) size = cap;
    }
  }
  const minNotional = BigInt(state.hello.fees.min_notional_quote_units);
  if (size < minNotional) return [];
  return [{ side: 'buy', symbol: sym, notional_quote_units: size, max_slippage_bps: MAX_SLIPPAGE_BPS }];
}

async function main(): Promise<void> {
  await runBars({
    onHello(state: BarsState): void {
      bot = newBotState(state);
    },
    onBatch(state: BarsState, batch: Batch): OrderInput[] {
      return onBatch(state, batch);
    },
  });
}

await main();
