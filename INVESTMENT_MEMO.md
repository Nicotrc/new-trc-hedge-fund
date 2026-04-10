# HedgeFund AI — Investment Memorandum
**Confidential | Seed Round 2025**

---

## Executive Summary

HedgeFund AI è un sistema operativo di trading quantitativo alimentato da intelligenza artificiale, progettato per identificare e gestire opportunità di investimento ad alto valore atteso su mercati globali attraverso una pipeline a 7 livelli completamente automatizzata.

Il sistema è oggi operativo in produzione, con architettura enterprise-grade su Docker, database time-series (TimescaleDB), 5 agenti quant specializzati con accesso asimmetrico ai dati, simulazione Monte Carlo a 10.000 percorsi, portfolio virtuale da $100k in paper trading e chatbot AI integrato per analisi on-demand sui singoli titoli.

**Stiamo raccogliendo €120.000 seed** per trasformare un prototipo tecnico avanzato in un prodotto scalabile con accesso a mercati globali, dati professionali e integrazione broker per trading automatico reale.

---

## Il Problema

I fondi hedge tradizionali spendono milioni in infrastruttura dati, team di analisti e quant per identificare opportunità che il mercato non ha ancora prezzato. Questo vantaggio informativo è sistematicamente precluso a:

- **Family office** e investitori privati con capitale < €10M
- **Gestori indipendenti** senza budget per Bloomberg Terminal o dati alternativi
- **Trader retail sofisticati** che operano su mercati globali senza strumenti professionali

I sistemi algoritmici esistenti sul mercato sono black box, costosi, non spiegabili e raramente integrano AI generativa con modelli quantitativi classici.

---

## La Soluzione

Un sistema AI che replica la logica decisionale di un comitato di investimento quantitativo istituzionale, rendendola accessibile, trasparente e autoapprendente.

### Architettura attuale — 7 Layer

| Layer | Funzione | Stato |
|---|---|---|
| **1. Data Layer** | 8 fonti: prezzi, fondamentali, news, macro (FRED), options flow, eventi FDA, short interest, insider trades | ✅ Operativo |
| **2. Signal Engine** | 5 strategie: Momentum Breakout, Mean Reversion, Event Biotech, Short Squeeze, Sector Rotation | ✅ Operativo |
| **3. Agenti Quant** | Momentum · Value · Event · Macro · Risk con accesso asimmetrico ai dati | ✅ Operativo |
| **4. Meta-Agent** | Bias detection, pesi adattivi per regime di mercato, amplificazione del dissenso | ✅ Operativo |
| **5. CIO v2** | Kelly Criterion + Monte Carlo 10.000 path → decisione probabilistica con VaR | ✅ Operativo |
| **6. Execution** | Paper trading $100k, stop-loss/take-profit/trailing stop automatici | ✅ Operativo |
| **7. Learning** | Calibrazione Bayesiana rolling 20 trade, aggiustamento pesi agenti | ✅ Operativo |

### Funzionalità chiave

- **Risk Agent Veto**: se il Risk Agent assegna score < 25 con alta conviction, la decisione viene bloccata in modo non-overridable — come un risk manager reale
- **Monte Carlo jump-diffusion**: per eventi binari (es. FDA PDUFA, Phase 3 readout) il sistema usa una distribuzione bimodale che riflette l'incertezza reale
- **Confluence bonus**: +15 punti quando lo stesso ticker appare in più strategie contemporaneamente
- **Chatbot AI integrato**: domande in linguaggio naturale su qualsiasi ticker con contesto dal DB in tempo reale
- **Dashboard live**: grafo pipeline animato, feed SSE real-time, opportunity cards con breakdown agenti

---

## Traction Attuale

- Sistema completamente operativo in produzione locale (Docker Compose)
- **200+ ticker attivi** su mercato USA: biotech, AI/semicon, difesa, energia, rinnovabili, crypto-adjacent, meme stocks, large cap
- Pipeline automatizzata: ingestion ogni 15 min → analisi → decisione → paper trade
- Codebase: **7.000+ righe**, architettura modulare, test-ready
- Audit trail completo su ogni decisione (DB TimescaleDB)

---

## Limiti Attuali e Piano di Miglioramento

### Limite 1 — Mercati solo USA

**Problema**: il sistema opera esclusivamente su mercati americani. Opportunità significative su mercati europei e asiatici sono completamente escluse.

**Soluzione post-seed**:
- Aggiunta suffissi ticker per LSE (`.L`), Euronext (`.PA`), XETRA (`.DE`), TSE (`.T`), HKEX
- Gestione multi-valuta con hedge automatico (EUR/USD, GBP/USD, JPY/USD)
- Agenti ricalibrati per specificità regionali (es. Value Agent: P/B < 1 è normale in Giappone, non in USA)
- Compliance MiFID II per mercati EU: log immutabile, giustificazione human-readable per ogni trade, report periodici

| Mercato | Opportunità principale |
|---|---|
| Europa (LSE, Euronext, XETRA) | Biotech europeo, energia, difesa |
| Asia (TSE, HKEX) | Semiconduttori, EV, tech cinese |
| Crypto (Binance/Coinbase) | 24/7, alta volatilità, pattern algoritmici forti |
| Forex | Hedging automatico su posizioni internazionali |
| Commodities (CME futures) | Oil, gold, correlati alle posizioni energy in portafoglio |

### Limite 2 — Dati standard, nessun edge informativo

**Problema**: il sistema usa dati pubblici standard (YFinance, FRED). I fondi hedge fanno i soldi veri con dati alternativi che il mercato non ha ancora scontato.

**Soluzione post-seed**:
- **Yahoo Finance API Premium / Polygon.io Pro**: dati intraday 1 minuto, Level 2 order book, grafici interattivi storici, notizie real-time con sentiment scoring
- **Sentiment social** (Reddit WallStreetBets, X/Twitter, StockTwits): il retail sentiment precede i movimenti su small cap e biotech — integrabile a basso costo via API pubbliche
- **Trascrizioni earnings call** (via Whisper + GPT): analisi automatica del tono del CEO, deviazione dalle aspettative, keyword detection — predittore forte di gap post-earnings
- **Unusual Whales Pro**: options flow istituzionale in tempo reale — tracciare i "big money" prima che il movimento sia visibile sul prezzo
- **Dati di spesa con carta di credito** (Earnest/Bloomberg Second Measure): predittore fatturato per consumer e retail tech 3-4 settimane prima dell'earnings

### Limite 3 — Agenti LLM puri: rischio di allucinazione

**Problema**: gli agenti attuali usano GPT per ragionamento quantitativo. I LLM possono allucinare numeri specifici, compromettendo l'accuratezza delle decisioni.

**Soluzione post-seed** — architettura ibrida:

```
LLM (ragionamento qualitativo, narrative analysis)
        +
Modelli ML classici (XGBoost per signal scoring,
LSTM per predizione prezzi a breve termine)
        +
Regole deterministiche hard-coded (stop-loss,
position limits, veto rules)
```

Il Momentum Agent ad esempio dovrebbe calcolare RSI, MACD, ATR con librerie numeriche — non chiedere a GPT di stimarli.

### Limite 4 — Nessun backtesting engine

**Problema critico per fundraising**: non esistono dati storici sulle performance del sistema. Non è possibile mostrare Sharpe ratio, max drawdown o win rate reali a potenziali investitori.

**Soluzione post-seed**:
- Engine di backtesting su dati storici 2019-2024 (5+ anni)
- Separazione rigorosa in-sample / out-of-sample per evitare overfitting
- Output: Sharpe ratio annualizzato, Sortino ratio, max drawdown, win rate per strategia, profit factor
- Walk-forward optimization per validare la stabilità dei parametri nel tempo

### Limite 5 — Gestione rischio solo per singolo trade

**Problema**: il sistema calcola Kelly e stop-loss per ogni trade individuale ma non gestisce il rischio a livello di portafoglio. Con 10 posizioni aperte, il rischio aggregato è ignoto.

**Soluzione post-seed**:
- **Correlation matrix real-time**: se 5 posizioni biotech long sono correlate al 0.85, il sistema riduce automaticamente le size
- **Portfolio VaR**: calcolo continuo del rischio aggregato — "quanto perdo se il mercato scende del 10%?"
- **Sector exposure limits**: max 20-25% del portafoglio in un singolo settore
- **Beta-adjusted sizing**: posizioni più piccole su titoli high-beta in regimi di alta volatilità (VIX > 25)
- **Drawdown circuit breaker**: blocco automatico di nuove posizioni se il portafoglio perde > 15% dal peak

### Limite 6 — Nessuna integrazione broker (paper trading only)

**Problema**: il sistema opera solo in paper trading. Tutto il valore generato dall'analisi rimane teorico.

**Soluzione post-seed**:
- **Fase 1 — Alpaca API** (mercati USA, API gratuita, ottima per automazione): ordini market/limit, position management, account balance sync
- **Fase 2 — Interactive Brokers TWS API**: accesso a mercati globali (EU, Asia), options, futures, forex — il broker più usato dai fondi hedge
- Switch manuale paper/live con approval flow: ogni decisione BUY sopra soglia richiede conferma esplicita prima dell'esecuzione reale
- Segregazione account: capitale di test separato dal capitale principale
- Audit trail immutabile per compliance: ogni ordine loggato con timestamp, rationale AI, prezzi eseguiti

---

## Roadmap 18 Mesi

### Q1 2025 — Dati e Credibilità
- Polygon.io Pro: dati intraday + grafici interattivi in dashboard
- Sentiment social (Reddit/X) integrato nell'Event Agent
- **Backtesting engine** — output: Sharpe, drawdown, win rate storici
- Portfolio-level VaR e correlation matrix

### Q2 2025 — Trading Reale
- Integrazione Alpaca (live trading, capital iniziale €10-15k)
- Trascrizioni earnings call automatiche (Whisper + GPT)
- Alert mobile per decisioni ad alta conviction
- Mercati europei: LSE + Euronext (primi 50 ticker)

### Q3 2025 — Espansione e ML
- Agenti ibridi: LLM + XGBoost/LSTM per signal scoring
- IBKR TWS API (accesso globale, options, forex)
- Asia: TSE + HKEX (semiconduttori, EV)
- Crypto 24/7 via Binance API

### Q4 2025 — Prodotto e Scaling
- Multi-portfolio: più strategie/account in parallelo
- App mobile: approval flow, alert push, performance dashboard
- Onboarding primi beta user esterni (family office, gestori indipendenti)
- Report compliance MiFID II automatizzati

---

## Utilizzo dei Fondi — €120.000

| Voce | Importo | Dettaglio |
|---|---|---|
| **Dati professionali** | €35.000/anno | Polygon.io Pro, Unusual Whales, dati alternativi |
| **Broker + capital live** | €20.000 | Setup IBKR, capital iniziale paper→live |
| **Sviluppo** | €45.000 | Backtesting, agenti ML, mobile app, mercati EU/Asia |
| **Infrastruttura cloud** | €12.000 | Server 24/7, backup, monitoring |
| **Legal/compliance** | €8.000 | MiFID II, struttura societaria |

---

## Vantaggio Competitivo

| Caratteristica | HedgeFund AI | Competitor tipico |
|---|---|---|
| Spiegabilità | Audit trail completo per ogni decisione | Black box |
| Autoapprendimento | Ricalibrazione Bayesiana rolling | Parametri fissi |
| Asimmetria dati | Risk Agent vede tutto, altri no | Accesso uniforme |
| Costo di accesso | Frazione dei costi istituzionali | Bloomberg: €25k/anno |
| Mercati | Multi-market (roadmap) | Spesso solo USA |
| Integrazione AI | LLM + ML classico + regole deterministiche | Solo uno dei tre |

---

## Team e Visione

Il sistema è stato progettato e costruito da zero con un approccio architetturale che privilegia la modularità, la spiegabilità e l'autoapprendimento — caratteristiche che i fondi hedge istituzionali costruiscono in anni con team di decine di ingegneri.

**Visione a 3 anni**: diventare l'infrastruttura AI di riferimento per family office e gestori indipendenti europei che operano su mercati globali, con un modello SaaS a abbonamento mensile basato sul capitale gestito.

---

## Ask

**€120.000 seed** per 18 mesi di runway completo.

In cambio: quota societaria da definire in fase di negoziazione, con board observer seat e diritti di pro-rata sui round successivi.

**Milestone per Series A** (target 18 mesi):
- Sharpe ratio > 1.5 verificato su backtesting 5 anni
- Live trading con performance tracciata per 6+ mesi
- 3+ mercati attivi (USA, Europa, Crypto)
- 5+ beta user esterni con AUM gestito > €500k aggregato

---

*Documento riservato. Non distribuire senza autorizzazione scritta.*
