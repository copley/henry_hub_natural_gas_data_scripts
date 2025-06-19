## 🔍  Decoding the Software That *Actually* Moves Energy Futures  
### (Read‑me for Natural‑Gas **NG** and WTI Crude‑Oil **CL** traders)

---

### 1.  How Price Discovery Really Happens  
There are **four software layers** you should know.  
They’re the same for NG (Henry Hub) and CL (WTI), just tuned to each product’s quirks:

| # | Layer | Who runs it | Typical code stack | Why it matters |
|---|-------|-------------|--------------------|----------------|
| **1** | **Exchange matcher** | CME Globex / ICE | C++ micro‑services, FIFO / Pro‑Rata / Threshold‑Pro‑Rata rules | Where every order is matched. Rules are public → you can simulate fills locally. |
| **2** | **Market‑maker bots** | Citadel Sec, Optiver, DRW, TrailStone, etc. | C++20/Rust gateways, kdb+ or ClickHouse for analytics, Python/q for research | Quote both sides, manage inventory with **Avellaneda‑Stoikov** maths. |
| **3** | **Directional / fundamental engines** | Hedge‑fund energy pods, majors (Shell/BP), utilities | Python + pandas/NumPy; PyTorch/XGBoost for flow models; QuantLib for curve bootstraps | Turn EIA/API stats, pipeline flows, weather into “fair value” curves. |
| **4** | **Momentum / breakout algos** | CTAs, prop desks, retail quants | Anything from PineScript → C++ | Opening‑range breakouts (Mark Fisher **ACD**), trend/momentum stacks. |

---

### 2.  Deep‑dive by Product  

#### 🌬️  **Natural‑Gas (NG) specifics**

| What moves it | Where to steal the code / data |
|---------------|--------------------------------|
| **Storage surprise (Thu 10:30 ET)** | EIA JSON feed → regress ∆Stocks vs. NG front‑month. |
| **Weather demand (GWDD / HDD / CDD)** | NOAA GFS GRIB2 → `eccodes + xarray` → feature matrix. |
| **Avellaneda‑Stoikov market‑making** | GitHub `nirajdsouza/market-making-strategy-simulation` (Python). |
| **Exchange sim** | CME “Matching Algorithm Overview” PDF + MDP 3 parser. |

**Quick‑start starter kit**

```bash
pip install websockets pandas duckdb eccodes ibapi streamlit
