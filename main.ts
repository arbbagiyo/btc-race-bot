// Dip-scalp BTC trading bot for make.fun's btc-race-24h-20261117-0603.
//
// Protocol agent-fun/bars-replay/v1: the harness sends hello once, then one
// batch per bar. Each batch carries the bar that just closed, the fills of the
// orders decided on the previous batch, and the account. This bot answers
// every batch with exactly one reply echoing its seq, carrying zero or one
// order. Orders decided on a batch fill at the next bar's open.
//
// Strategy: buy deep dips, exit on take-profit / stop-loss / maximum hold.
// A dip counts when ANY of these fires on the latest bar:
//   - Bollinger z-score: close 1.1 std below the 90-bar mean,
//   - confirmed bounce: the 30-bar z-score dipped below -1.3 within the last
//     8 bars and the last 5 bars already bounced 43 bps,
//   - drawdown: close 202 bps below the 480-bar rolling high.
// Every entry must also pass all gates: RSI(7) at or below 41, close within
// 87 bps below the 240-bar EMA (no strong downtrend), and at most 3 flat bars
// in the last 60 (no chop). Exits: +185 bps take-profit, -182 bps stop-loss,
// or 720 bars held. 30-bar cooldown after each exit. Size 99.5% of cash,
// capped so own impact on the latest close's depth stays under 30 bps.
//
// All decisions use only past bars. No clock, no network, no files, no
// randomness: the same bars always produce the same orders.

import { runBars } from './vendor/makefun-bars.ts';
import type { Batch, OrderInput } from './vendor/makefun-bars.ts';

const SYMBOL = 'CBBTC';

const CFG = {
  bollN: 90,
  bollZ: 1.1,
  bounceN: 30,
  bounceZ: 1.3,
  bounceW: 8,
  bounceMomK: 5,
  bounceMomBps: 43,
  ddWin: 480,
  ddDipBps: 202,
  rsiPeriod: 7,
  rsiMax: 41,
  emaWin: 240,
  emaFilterBps: 87,
  flatWin: 60,
  flatMax: 3,
  tpBps: 185,
  slBps: 182,
  holdBars: 720,
  cooldownBars: 30,
  warmBars: 30,
  sizeNum: 995n,
  sizeDen: 1000n,
  impactCapBps: 30n,
  slipBps: 100,
  minNotional: 1000000n,
};

// ---- streaming bar history (closes in USDC, floats) ----
const closes: number[] = [];
const flatFlags: boolean[] = [];
const cDepths: bigint[] = [];

// ---- streaming indicators (Wilder RSI, EMA) ----
let rsiGain = 0;
let rsiLoss = 0;
let ema = 0;

function rsiNow(idx: number): number {
  if (idx < CFG.rsiPeriod) return 50;
  if (rsiLoss === 0) return 100;
  return 100 - 100 / (1 + rsiGain / rsiLoss);
}

function meanStd(n: number, idx: number): { mean: number; std: number } {
  const lo = Math.max(0, idx - n + 1);
  let s = 0;
  for (let j = lo; j <= idx; j++) s += closes[j]!;
  const k = idx - lo + 1;
  const mean = s / k;
  let v = 0;
  for (let j = lo; j <= idx; j++) { const d = closes[j]! - mean; v += d * d; }
  return { mean, std: Math.sqrt(v / k) };
}

function rollHigh(n: number, idx: number): number {
  let m = -Infinity;
  for (let j = Math.max(0, idx - n + 1); j <= idx; j++) m = Math.max(m, closes[j]!);
  return m;
}

// ---- position state, rebuilt from the fills each batch carries ----
let posQty = 0n;
let posAvgNum = 0n; // sum of fill qty * fill price (price-scale units)
let entryBar = -1;
let peak = 0;
let cooldownUntil = -1;

function avgPrice(): number {
  if (posQty === 0n) return 0;
  return Number(posAvgNum / posQty) / 1e5;
}

await runBars({
  onBatch(state, batch: Batch): OrderInput[] {
    // 1. Fold the new bar into history (a batch may carry no bar on a gap minute).
    const bar = batch.bars.find((b) => b.symbol === SYMBOL);
    let idx = closes.length - 1;
    if (bar !== undefined) {
      const close = Number(bar.c) / 1e5;
      closes.push(close);
      flatFlags.push(bar.o === bar.h && bar.h === bar.l && bar.l === bar.c);
      cDepths.push(BigInt(bar.c_depth[0]!));
      idx = closes.length - 1;
      if (idx === 0) {
        ema = close;
      } else {
        const d = close - closes[idx - 1]!;
        const up = Math.max(0, d);
        const dn = Math.max(0, -d);
        const p = CFG.rsiPeriod;
        if (idx <= p) {
          rsiGain += up;
          rsiLoss += dn;
          if (idx === p) { rsiGain /= p; rsiLoss /= p; }
        } else {
          rsiGain = (rsiGain * (p - 1) + up) / p;
          rsiLoss = (rsiLoss * (p - 1) + dn) / p;
        }
        const k = 2 / (CFG.emaWin + 1);
        ema = close * k + ema * (1 - k);
      }
    }

    // 2. Fold the fills: rebuild the position from what actually filled.
    for (const f of batch.fills) {
      if (f.symbol !== SYMBOL || f.status !== 'filled') continue;
      if (f.side === 'buy') {
        const q = BigInt(f.qty);
        const p = BigInt(f.price);
        posQty += q;
        posAvgNum += q * p;
        entryBar = idx;
        peak = Number(p) / 1e5;
      } else {
        const q = BigInt(f.qty);
        if (q >= posQty) {
          posQty = 0n;
          posAvgNum = 0n;
          entryBar = -1;
          cooldownUntil = idx + CFG.cooldownBars;
        } else {
          posAvgNum = (posAvgNum * (posQty - q)) / posQty;
          posQty -= q;
        }
      }
    }
    // Safety: if the harness account says flat, drop any stale tracking.
    const held = state.positions.get(SYMBOL);
    if ((held === undefined || held.qty === 0n) && posQty !== 0n) {
      posQty = 0n;
      posAvgNum = 0n;
      entryBar = -1;
    }

    // 3. No decisions without a fresh bar, or while the harness warms up.
    if (bar === undefined || state.warmingUp) return [];
    const close = closes[idx]!;

    // 4. Long: exits only.
    if (posQty > 0n) {
      if (close > peak) peak = close;
      const gainBps = (close / avgPrice() - 1) * 1e4;
      const heldBars = idx - entryBar;
      if (gainBps <= -CFG.slBps || gainBps >= CFG.tpBps || heldBars >= CFG.holdBars) {
        return [{ side: 'sell', symbol: SYMBOL, qty: posQty, max_slippage_bps: CFG.slipBps }];
      }
      return [];
    }

    // 5. Flat: entries, after warm-up and cooldown.
    if (idx < CFG.warmBars || idx < cooldownUntil) return [];
    let signal = false;
    const ms = meanStd(CFG.bollN, idx);
    if (ms.std > 0 && (close - ms.mean) / ms.std <= -CFG.bollZ) signal = true;
    if (!signal) {
      let minZ = Infinity;
      for (let j = Math.max(0, idx - CFG.bounceW + 1); j <= idx; j++) {
        const m = meanStd(CFG.bounceN, j);
        if (m.std > 0) minZ = Math.min(minZ, (closes[j]! - m.mean) / m.std);
      }
      if (
        minZ <= -CFG.bounceZ &&
        idx >= CFG.bounceMomK &&
        (close / closes[idx - CFG.bounceMomK]! - 1) * 1e4 >= CFG.bounceMomBps
      ) signal = true;
    }
    if (!signal && close <= rollHigh(CFG.ddWin, idx) * (1 - CFG.ddDipBps / 1e4)) signal = true;
    if (!signal) return [];

    // Gates: weak RSI, no strong downtrend, no flat chop.
    if (rsiNow(idx) > CFG.rsiMax) return [];
    if (close < ema * (1 - CFG.emaFilterBps / 1e4)) return [];
    let flats = 0;
    for (let j = Math.max(0, idx - CFG.flatWin + 1); j <= idx; j++) if (flatFlags[j]) flats++;
    if (flats > CFG.flatMax) return [];

    // Size: 99.5% of cash, capped by own impact on the latest close's depth.
    // (The next bar's open depth is unknown when deciding; the latest close's
    // depth, a minute earlier, is the best estimate the protocol gives us.)
    let spend = (state.cash * CFG.sizeNum) / CFG.sizeDen;
    const cap = (cDepths[idx]! * CFG.impactCapBps) / 10000n;
    if (spend > cap) spend = cap;
    if (spend < CFG.minNotional) return [];
    return [{ side: 'buy', symbol: SYMBOL, notional_quote_units: spend, max_slippage_bps: CFG.slipBps }];
  },
});
