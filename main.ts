// Ensemble BTC trading bot for make.fun btc-race-24h-20261117-0603.
//
// Speaks agent-fun/bars-replay/v1 through the vendored shim: the harness sends
// hello once, then one batch per bar (each batch carries the bar that just
// closed, the fills of the previous batch's orders, and the account), and the
// bot answers every batch with exactly one reply echoing its seq. Orders
// decided on a batch fill at the next bar's open.
//
// Strategy: a weighted vote over twelve causal signals computed from the
// streaming bar history (dip z-scores, drawdowns, bounce, RSI, Donchian
// breakout, wick, EMA distance, trend continuation, up-count confirmation),
// gated by chop/volatility/trend/depth regime filters. Exits: take-profit
// (with an optional partial first target), stop-loss, trailing stop, maximum
// hold, RSI-overbought and momentum-fade exits, plus a cooldown after exits.
// All parameters live in CONFIG, tuned offline by evolutionary search over an
// exact fill-model simulator of the dev window. The bot itself is fully
// deterministic: no clock, no network, no randomness, no files.

import { runBars } from './vendor/makefun-bars.ts';
import type { BarsState, Batch, OrderInput } from './vendor/makefun-bars.ts';

// CONFIG: winning genome from evolutionary search over an exact fill-model
// simulator of the dev window (4 seeds x 300 generations wave 1, 4 seeds x 400
// generations refinement, ~1M evaluations; all runs converged here).
// Grade-local net PnL $43.07 on the dev window (simulator figure; the
// grade-local run is the arbiter). $50 was not reached: see the report.
const CONFIG = {
  w1: 2.1387288440018892, w2: 1.1854999338509515, w3: 0.553121151169762, w4: 1.7288235890562647,
  w5: 1.408765661995858, w6: 0.14364735460840167, w7: 0, w8: 0.32349153959657995,
  w9: 0, w10: 0.31995123010128734, w11: 0.965328929037787, w12: 0.08757428918033838,
  t1: -1.112285838922253, t2: -1.9852935831586365, t3: 0.01574856706238352, t4: 0.020099403387331405,
  t5a: -0.0004504911273252217, t5b: -1.3136048930871767, t6: 33.632630254141986,
  t7a: 0.002842296870192513, t7b: 0.004626744973426685, t8a: 0.6221597950195428, t8b: -0.5798866701079533,
  t9a: -0.012661257474974265, t9b: 0.002197838731575758, t10a: 0.0020191073481226345, t10b: 0.0022903175596147777,
  t11a: 0.014341998318047262, t11b: 4, t12a: 23.98703528114129, t12b: 6,
  theta: 2.8325975141488016,
  gateChop: 1, flatMax: 3,
  gateVolLo: 0, volLo: 0.0008200253161834553,
  gateVolHi: 0, volHi: 0.00306902308051358,
  gateTrend: 1, trendMin: -0.0076319896091474205,
  gateDepth: 1, depthMin: 0.9,
  gateVolRegime: 0, volRegMax: 2.042612457694486,
  tp: 0.017063368829328098, sl: 0.01597296143458225, tp1: 0.018784317395184193, tp1Frac: 0.3291460104170255,
  trailAct: 0.025, trailDist: 0.011332102213334291,
  maxHold: 149, rsiExit: 0, fadeOn: 0, fadeMom: -0.00958516012004111,
  cooldown: 51,
  sizeFrac: 1, impactCap: 59.479132753331214, maxSlip: 65,
  addOn: 1, addDD: 0.005568732735933736, addFrac: 0.7766138447448612,
  warmup: 60, fallback: 1, fallbackBar: 1089,
};

const SYMBOL = 'CBBTC';

// Rolling causal features over the bar history, mirroring the offline
// simulator's recurrences exactly (same running sums, same Wilder RSI, same
// window loops), so the live bot decides exactly as the simulator predicted.
class Features {
  closes: number[] = [];
  highs: number[] = [];
  lows: number[] = [];
  depths: number[] = [];
  flats: number[] = [];
  // running sums for SMAs
  private s20 = 0; private s45 = 0; private s90 = 0; private s240 = 0;
  private sLr60 = 0; private sLr240 = 0;
  private lr: number[] = [];
  // EMA states
  private e60 = 0; private e240 = 0; private eDepth = 0;
  // Wilder RSI states
  private rsi = new Map<number, { g: number; l: number; n: number }>();
  // ATR running sums
  private atr14 = 0; private atr30 = 0;
  private trs: number[] = [];

  get n(): number { return this.closes.length; }
  get close(): number { return this.closes[this.closes.length - 1]; }

  addBar(o: number, h: number, l: number, c: number, depth: number, flat: number): void {
    const i = this.closes.length;
    const prev = i > 0 ? this.closes[i - 1] : c;
    this.closes.push(c); this.highs.push(h); this.lows.push(l); this.depths.push(depth); this.flats.push(flat);
    this.s20 += c; if (i >= 20) this.s20 -= this.closes[i - 20];
    this.s45 += c; if (i >= 45) this.s45 -= this.closes[i - 45];
    this.s90 += c; if (i >= 90) this.s90 -= this.closes[i - 90];
    this.s240 += c; if (i >= 240) this.s240 -= this.closes[i - 240];
    const lr = i === 0 ? 0 : Math.log(c / prev);
    this.lr.push(lr);
    this.sLr60 += lr; if (i >= 60) this.sLr60 -= this.lr[i - 60];
    this.sLr240 += lr; if (i >= 240) this.sLr240 -= this.lr[i - 240];
    const k60 = 2 / 61, k240 = 2 / 241, kD = 2 / 121;
    this.e60 = i === 0 ? c : c * k60 + this.e60 * (1 - k60);
    this.e240 = i === 0 ? c : c * k240 + this.e240 * (1 - k240);
    this.eDepth = i === 0 ? depth : depth * kD + this.eDepth * (1 - kD);
    for (const w of [7, 14, 30]) {
      let st = this.rsi.get(w);
      if (!st) { st = { g: 0, l: 0, n: 0 }; this.rsi.set(w, st); }
      if (i === 0) { st.n = 0; continue; }
      const d = c - prev;
      if (st.n < w) {
        if (d > 0) st.g += d; else st.l -= d;
        st.n++;
        if (st.n === w) { st.g /= w; st.l /= w; }
      } else {
        st.g = (st.g * (w - 1) + Math.max(d, 0)) / w;
        st.l = (st.l * (w - 1) + Math.max(-d, 0)) / w;
      }
    }
    const tr = i === 0 ? h - l : Math.max(h - l, Math.abs(h - prev), Math.abs(l - prev));
    this.trs.push(tr);
    this.atr14 += tr; if (i >= 14) this.atr14 -= this.trs[i - 14];
    this.atr30 += tr; if (i >= 30) this.atr30 -= this.trs[i - 30];
  }

  private sma(sum: number, w: number): number {
    return this.n >= w ? sum / w : NaN;
  }
  private std(w: number, mean: number): number {
    if (this.n < w || Number.isNaN(mean)) return NaN;
    let s = 0;
    for (let j = this.n - w; j < this.n; j++) { const d = this.closes[j] - mean; s += d * d; }
    return Math.sqrt(s / w);
  }
  z(w: number): number {
    const sum = w === 20 ? this.s20 : w === 45 ? this.s45 : w === 90 ? this.s90 : this.s240;
    const m = this.sma(sum, w);
    const s = this.std(w, m);
    if (Number.isNaN(m) || s === 0) return 0;
    return (this.close - m) / s;
  }
  rsiValue(w: number): number {
    const st = this.rsi.get(w)!;
    if (st.n <= w) return NaN;
    if (st.l === 0) return 100;
    return 100 - 100 / (1 + st.g / st.l);
  }
  drawdown(w: number): number {
    let m = -Infinity;
    for (let j = Math.max(0, this.n - w); j < this.n; j++) if (this.closes[j] > m) m = this.closes[j];
    return this.close / m - 1;
  }
  ema60d(): number { return this.close / this.e60 - 1; }
  ema240d(): number { return this.close / this.e240 - 1; }
  mom(lag: number): number {
    return this.n > lag ? this.close / this.closes[this.n - 1 - lag] - 1 : 0;
  }
  atr14r(): number { return this.n >= 14 ? this.atr14 / 14 / this.close : 0; }
  donchHi(): number {
    let m = -Infinity;
    for (let j = Math.max(0, this.n - 45); j < this.n; j++) if (this.highs[j] > m) m = this.highs[j];
    return m;
  }
  wick(o: number, h: number, l: number): number {
    const r = h - l;
    return r > 0 ? (Math.min(o, this.close) - l) / r : 0;
  }
  flat60(): number {
    let s = 0;
    for (let j = Math.max(0, this.n - 60); j < this.n; j++) s += this.flats[j];
    return s;
  }
  depthRatio(): number { return this.depths[this.n - 1] / this.eDepth; }
  downCount(): number {
    let c = 0;
    for (let j = this.n - 1; j > 0; j--) {
      if (this.closes[j] < this.closes[j - 1]) c++; else break;
    }
    return c;
  }
  upCount5(): number {
    let c = 0;
    for (let j = Math.max(1, this.n - 5); j < this.n; j++) if (this.closes[j] > this.closes[j - 1]) c++;
    return c;
  }
  volRegime(): number {
    if (this.n < 240) return 1;
    const m60 = this.sLr60 / 60, m240 = this.sLr240 / 240;
    let s60 = 0, s240 = 0;
    for (let j = this.n - 60; j < this.n; j++) { const d = this.lr[j] - m60; s60 += d * d; }
    for (let j = this.n - 240; j < this.n; j++) { const d = this.lr[j] - m240; s240 += d * d; }
    const v60 = Math.sqrt(s60 / 60), v240 = Math.sqrt(s240 / 240);
    return v240 === 0 ? 1 : v60 / v240;
  }
}

async function main(): Promise<void> {
  const feats = new Features();
  let priceScale = 5, qtyScale = 8, quoteScale = 6;
  let entryIdx = -1;
  let entryPx = 0;
  let peakPx = 0;
  let tp1Done = false;
  let cooldownUntil = -1;
  let everTraded = false;
  let everRejected = false;
  let awaitingId: string | null = null;
  let orderSeq = 0;

  await runBars({
    onHello(state: BarsState): void {
      const s = state.symbols.get(SYMBOL);
      if (s) { priceScale = s.price_scale; qtyScale = s.qty_scale; }
      quoteScale = state.hello.capital.quote_scale;
    },
    onBatch(state: BarsState, batch: Batch): OrderInput[] {
      // Track fills first: they tell us whether our last order landed.
      let boughtThisBatch = false;
      for (const f of batch.fills) {
        if (f.client_id === awaitingId) awaitingId = null;
        if (f.status === 'filled') {
          everTraded = true;
          if (f.side === 'buy') boughtThisBatch = true;
        } else if (f.reason !== null) everRejected = true;
      }
      if (awaitingId !== null) return []; // an order is still in flight; wait for its fill
      const bar = batch.bars.find((b) => b.symbol === SYMBOL);
      if (bar === undefined) return []; // a gap minute: no bar, no decision
      const o = Number(BigInt(bar.o)) / 10 ** priceScale;
      const h = Number(BigInt(bar.h)) / 10 ** priceScale;
      const l = Number(BigInt(bar.l)) / 10 ** priceScale;
      const c = Number(BigInt(bar.c)) / 10 ** priceScale;
      const depth = Number(BigInt(bar.c_depth[0])) / 10 ** quoteScale;
      const flat = bar.o === bar.h && bar.h === bar.l && bar.l === bar.c ? 1 : 0;
      feats.addBar(o, h, l, c, depth, flat);
      const k = feats.n - 1; // bar index, as in the simulator
      if (state.warmingUp) return [];

      const pos = state.positions.get(SYMBOL);
      const qtyUnits = pos ? pos.qty : 0n;
      const qty = Number(qtyUnits) / 10 ** qtyScale;
      const cost = pos ? Number(pos.cost) / 10 ** quoteScale : 0;
      const cash = Number(state.cash) / 10 ** quoteScale;
      const equity = Number(state.equity) / 10 ** quoteScale;
      const g = CONFIG;

      // Position lifecycle, read from the harness account (fills already applied).
      if (qty > 0 && entryIdx < 0) {
        entryIdx = k; entryPx = cost / qty; peakPx = c; tp1Done = false;
      } else if (qty <= 0 && entryIdx >= 0) {
        entryIdx = -1; cooldownUntil = k + g.cooldown;
      } else if (qty > 0 && boughtThisBatch) {
        entryPx = cost / qty; // scale-in refreshes the cost basis, as in the simulator
      }
      if (qty > 0 && c > peakPx) peakPx = c;

      const send = (order: OrderInput): OrderInput[] => {
        orderSeq++;
        const id = `e${orderSeq}`;
        (order as { client_id?: string }).client_id = id;
        awaitingId = id;
        return [order];
      };

      // --- exits ---
      if (qty > 0 && entryIdx >= 0) {
        const gain = c / entryPx - 1;
        const holdBars = k - entryIdx;
        let sellFrac = 0;
        if (gain <= -g.sl) sellFrac = 1;
        else if (g.tp1 > 0 && !tp1Done && gain >= g.tp1) { sellFrac = g.tp1Frac; tp1Done = true; }
        else if (gain >= g.tp) sellFrac = 1;
        else if (g.trailAct > 0 && peakPx / entryPx - 1 >= g.trailAct && c < peakPx * (1 - g.trailDist)) sellFrac = 1;
        else if (holdBars >= g.maxHold) sellFrac = 1;
        else if (g.rsiExit > 0 && feats.rsiValue(14) >= g.rsiExit && gain > 0.002) sellFrac = 1;
        else if (g.fadeOn === 1 && gain > 0.003 && feats.mom(5) <= g.fadeMom) sellFrac = 1;
        if (sellFrac > 0) {
          const sellUnits = sellFrac >= 1 ? qtyUnits : (qtyUnits * BigInt(Math.floor(sellFrac * 10000))) / 10000n;
          if (sellUnits > 0n && (sellUnits === qtyUnits || Number(sellUnits) / 10 ** qtyScale * c >= 1)) {
            return send({ side: 'sell', symbol: SYMBOL, qty: sellUnits.toString(), max_slippage_bps: g.maxSlip });
          }
        }
      }

      // --- scale-in ---
      if (qty > 0 && entryIdx >= 0 && g.addOn === 1 && cash >= 1) {
        if (c / entryPx - 1 <= -g.addDD) {
          const heldVal = qty * c;
          const room = equity - heldVal;
          let amt = Math.min(cash * g.addFrac, room);
          if (amt >= 1) {
            let units = BigInt(Math.floor(amt * 10 ** quoteScale));
            if (units > state.cash) units = state.cash; // guard float rounding at full size
            if (units >= 1000000n) {
              return send({ side: 'buy', symbol: SYMBOL, notional_quote_units: units.toString(), max_slippage_bps: g.maxSlip });
            }
          }
        }
      }

      // --- entries: weighted vote over twelve signals, regime-gated ---
      if (qty <= 0 && k >= g.warmup && k >= cooldownUntil) {
        let gate = true;
        if (g.gateChop === 1 && feats.flat60() > g.flatMax) gate = false;
        if (g.gateVolLo === 1 && feats.atr14r() < g.volLo) gate = false;
        if (g.gateVolHi === 1 && feats.atr14r() > g.volHi) gate = false;
        if (g.gateTrend === 1 && feats.ema240d() < g.trendMin) gate = false;
        if (g.gateDepth === 1 && feats.depthRatio() < g.depthMin) gate = false;
        if (g.gateVolRegime === 1 && feats.volRegime() > g.volRegMax) gate = false;
        if (gate) {
          let score = 0;
          if (g.w1 > 0 && feats.z(90) <= g.t1) score += g.w1;
          if (g.w2 > 0 && feats.z(20) <= g.t2) score += g.w2;
          if (g.w3 > 0 && feats.drawdown(240) <= -g.t3) score += g.w3;
          if (g.w4 > 0 && feats.drawdown(480) <= -g.t4) score += g.w4;
          if (g.w5 > 0 && feats.mom(5) >= g.t5a && feats.z(45) <= g.t5b) score += g.w5;
          if (g.w6 > 0 && feats.rsiValue(14) <= g.t6) score += g.w6;
          if (g.w7 > 0 && c >= feats.donchHi() * (1 - g.t7a) && feats.mom(15) >= g.t7b) score += g.w7;
          if (g.w8 > 0 && feats.wick(o, h, l) >= g.t8a && feats.z(45) <= g.t8b) score += g.w8;
          if (g.w9 > 0 && feats.ema240d() <= g.t9a && feats.mom(5) >= g.t9b) score += g.w9;
          if (g.w10 > 0 && feats.mom(30) >= g.t10a && feats.ema60d() >= g.t10b) score += g.w10;
          if (g.w11 > 0 && feats.drawdown(120) <= -g.t11a && feats.upCount5() >= g.t11b) score += g.w11;
          if (g.w12 > 0 && feats.rsiValue(7) <= g.t12a && feats.downCount() >= g.t12b) score += g.w12;
          if (score >= g.theta) {
            let amt = cash * g.sizeFrac;
            amt = Math.min(amt, depth * (g.impactCap / 10000));
            amt = Math.min(amt, equity);
            if (amt >= 1) {
              let units = BigInt(Math.floor(amt * 10 ** quoteScale));
              if (units > state.cash) units = state.cash; // guard float rounding at full size
              if (units >= 1000000n) {
                return send({ side: 'buy', symbol: SYMBOL, notional_quote_units: units.toString(), max_slippage_bps: g.maxSlip });
              }
            }
          }
        }
      }

      // --- fallback: guarantee at least one trade in a window with no signal ---
      if (g.fallback === 1 && !everTraded && !everRejected && qty <= 0 && k === g.fallbackBar) {
        let units = BigInt(Math.floor(cash * 0.5 * 10 ** quoteScale));
        if (units > state.cash) units = state.cash;
        if (units >= 1000000n) {
          return send({ side: 'buy', symbol: SYMBOL, notional_quote_units: units.toString(), max_slippage_bps: g.maxSlip });
        }
      }
      return [];
    },
  });
}

main();
