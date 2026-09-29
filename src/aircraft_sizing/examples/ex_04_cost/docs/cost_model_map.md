# Cost Model Map

How aircraft design parameters turn into dollars, for the four models implemented in
[`ex_04_cost`](../methods/). Every equation here is cited to a published source and
implemented in [`methods/military.py`](../methods/military.py) or
[`methods/commercial.py`](../methods/commercial.py).

> **Teaching note:** a cost-estimating relationship (CER) is a statistical fit to
> historical programme data, not a physical law. Nothing below is dimensionally
> consistent, the exponents carry no physics, and each CER is only valid inside the
> dollar year and the aircraft class it was fitted to.

---

## The shape of the problem

```
   design parameters:  We, V_max, T, Tt4, Q, material mix, KSLOC, W_TO, geometry
                                      |
        +-----------------------------+-----------------------------+
        |                     |                    |                |
   DAPCA IV            Roskam Part VIII       COCOMO II     LO treatment
  (Raymer Ch. 18)       (whole programme)     (software)    (geometry -> $)
        |                     |                    |                |
        +---------------------+--------> acquisition cost <---------+
                              |                         |
             military: + O&M -> life-cycle cost    commercial: DOC model
                                                    -> $/trip -> $/seat-km
```

Acquisition cost is the hinge. The military models produce it; the commercial model
consumes it as an input.

---

## 1. DAPCA IV — RAND, via Raymer Ch. 18

The same CER family appears in two dollar years. **The exponents are identical**;
only the lead coefficients and the wrap rates they pair with change. Mixing a 1999
coefficient with a 2012 wrap rate is a common and invisible error.

| Quantity | Relation | 1999 lead | 2012 lead |
|---|---|---|---|
| Engineering hours | $H_{eng} = k\,W_e^{0.777} V^{0.894} Q^{0.163} D_{47}$ | 7.07 | 4.86 |
| Tooling hours | $H_{tool} = k\,W_e^{0.777} V^{0.696} Q^{0.263} D_{47}$ | 8.71 | 5.99 |
| Manufacturing hours | $H_{mfg} = k\,W_e^{0.82} V^{0.484} Q^{0.641} D_{47}$ | 10.72 | 7.37 |
| Quality control hours | $H_{qc} = 0.133\,H_{mfg}$ | — | — |
| Development support | $C_{ds} = k\,W_e^{0.63} V^{1.3}$ | 66 | 91.3 |
| Flight test | $C_{ft} = k\,W_e^{0.325} V^{0.822} FTA^{1.21}$ | 1807.1 | 2498 |
| Manufacturing materials | $C_{mm} = k\,W_e^{0.921} V^{0.621} Q^{0.799}$ | 16 | 22.1 |
| Wrap rates (eng/tool/mfg/QC) | $/hour | 86 / 88 / 73 / 81 | 115 / 118 / 98 / 108 |

$W_e$ is empty weight in lb, $V$ is maximum velocity in knots, $Q$ is the production
quantity, $FTA$ is the number of flight-test aircraft.

**Engine production** (Raymer): thrust in lbf, turbine inlet temperature in °R.

$$C_{ep} = 2.251\,(0.043\,T_{max} + 243.25\,M_{max} + 0.969\,T_{t4} - 2228)\times 1000 \times N_{eng}$$

**Material factor** — the structural mix scales *all four* labour pools at once:

$$D_{47} = \frac{1}{100}\sum_i p_i f_i$$

over aluminium, carbon fibre, fibreglass, steel and titanium.

Two factor sets are available, and they disagree about composites:

| Set | Al | Carbon fibre | Fibreglass | Steel | Titanium |
|---|---|---|---|---|---|
| Brandt workbook `Cost!C41:C45` (default, reproduces the sheet) | 1.0 | **0.9** | 0.9 | 1.0 | 1.5 |
| Raymer Ch. 18 as published (midpoints of his ranges) | 1.0 | **1.45** | 1.15 | 1.75 | 1.95 |

Raymer's own text puts every alternative to aluminium *above* 1.0 — graphite-epoxy
1.1–1.8, fibreglass 1.1–1.2, steel 1.5–2.0, titanium 1.7–2.2 — so **a pound of
composite costs more to build**, and Roskam's $F_{mat}$ (§2) agrees. The workbook's
0.9 is the outlier; it appears to net in the weight saving that composites buy, which
is a *sizing* effect and lives in a different chapter. See the note at the end of §2.

### Programme roll-up (Brandt workbook, sheet `Cost`)

```
C_subtotal   = C_eng + C_tool + C_mfg + C_qc + C_ds + C_ft + C_mm + C_ep   (C72)
C_avionics   = AF  * C_subtotal                                            (C74)
C_invest     = ICF * (C_subtotal + C_avionics)                             (C76)
C_total_base = C_subtotal + C_avionics + C_invest + stealth*Q + C_software  (C78)
C_program    = (1 + EF) * C_total_base                                     (C80)
C_unit       = C_program / Q                                               (K6)
```

The signature treatment is *recurring* -- every airframe gets treated -- so it is
carried in the recurring and flyaway figures too.  Software is *non-recurring* -- the
code is written once however many airframes are built -- so it lands only in the
non-recurring figure.  The workbook has neither line (it reports treatment costs in a
separate block and has no software at all), so `signature_level = "none"` with a zero
person-month rate reproduces the sheet — apart from the quality-control line, where
this model follows Raymer rather than the sheet (0.14% on unit cost; see Step 7 of the
walkthrough).

Seven figures come out, and they mean different things:

| Figure | Contains | F-16A, 2026 dollars |
|---|---|---|
| Recurring unit cost (`J6`) | production labour, materials, engines, avionics, signature | $70.8 M |
| Average flyaway cost (`L6`) | recurring plus tooling | $78.4 M |
| Programme unit cost (`K6`) | everything, non-recurring spread over the buy | $95.9 M |
| Acquisition unit cost | Roskam's $C_{ACQ}/N_m$: production and profit, no RDT&E | see §2 |
| Life-cycle unit cost (`F8`) | programme unit plus a lifetime of O&M | $130.8 M |
| Total non-recurring (`I6`) | design, tooling, test and software, spent once | $5.03 B |
| Total programme (`C80`) | the whole bill | $19.2 B |

> **Teaching note:** published "flyaway costs" mean **average flyaway cost**.
> Comparing your programme unit cost against a published flyaway cost overstates your
> aircraft by about 20% before you have made a single engineering decision.

### Operations, support and life-cycle cost

$$C_{O\&M} = \frac{FH}{DMT}\,DMF\cdot F_\$ \;+\; CR\cdot CH\cdot R_E(1+EF) \;+\; \frac{MMH}{FH}\,FH\,R_M(1+EF)$$

$$LCC = C_{O\&M}\cdot \text{life} + C_{unit}$$

---

## 1b. Low-observables treatment (Brandt `Cost!A114:G167`)

The one model here where a **geometric** parameter prices straight into dollars.
Nine treatments, each an area or a length times an installed unit rate. Two of them
carry a multiplier on the geometry, and the vertical tail is painted at Level A but
RAM-treated at B and C:

| Treatment | Quantity | Rate (2005 $) | Multiplier | Levels |
|---|---|---|---|---|
| Conductive paint — outer skin | skin wetted area [ft²] | 8 $/ft² | — | B, C |
| Conductive paint — inlet duct | inlet duct area [ft²] | 8 $/ft² | — | all |
| Conductive paint — vertical tail | vertical tail area [ft²] | 8 $/ft² | — | A |
| HF RAM — inlet lips | inlet lip length [ft] | 400 $/ft | — | all |
| HF RAM — airframe edges | treated edge length [ft] | 400 $/ft | — | B, C |
| HF RAM — hinge lines | hinge-line length [ft] | 400 $/ft | **× 2.0** | B, C |
| HF RAM — access panels | access-panel perimeter [ft] | 400 $/ft | **× 1.25** | B, C |
| HF RAM — vertical tail | vertical tail outer area [ft²] | 400 $/ft | — | B, C |
| HF RAM — radar bulkhead | radar bulkhead area [ft²] | 400 $/ft | — | A |
| High-temp material — exhaust | exhaust-washed area [ft²] | 800 $/ft | — | B, C |
| Volumetric edge material (RAS) | lips and edges [ft] | +6000 $/ft | — | C |
| Radome | — | +$500,000 flat | — | C with radome |

> **Two workbook quirks worth knowing before you check this by hand.** The 2.0 on hinge
> lines and the 1.25 on access panels are the sheet's own and are not folded into the
> rate, so a plain length × rate lands **42.4% below** the model on those two terms and
> **16.0% below** on the Level B total. And the sheet applies its **per-foot** RAM and
> high-temp rates to three **areas** (vertical tail, radar bulkhead, exhaust); that is
> dimensionally incoherent but it is what `Cost!A114:G167` does, and this model
> reproduces it.

Levels differ only in which treatments are switched on. F-16A, 2005 dollars: **A $9.5 k
· B $236 k · C $1.28 M · C with radome $1.78 M** per airframe. Before it joins the DAPCA
subtotal, the total is escalated from 2005 to the coefficient set's dollar year (1999
or 2012) with Roskam's CEF.

Worked example — Level B on the F-16A:

```
skin            1479.580 ft2 x    8          =   11,837
inlet duct       196.230 ft2 x    8          =    1,570
inlet lips        11.543 ft  x  400          =    4,617
airframe edges   162.599 ft  x  400          =   65,040
hinge lines       83.283 ft  x  400 x 2.00   =   66,626
access panels     45.000 ft  x  400 x 1.25   =   22,500
vertical tail     90.732 ft2 x  400          =   36,293
exhaust           34.629 ft2 x  800          =   27,703
                                                --------
                                                 236,186
```

To the nearest dollar, from geometry rounded to three decimals. Carried unrounded this
is `Cost!F162` = $236,185.5945, which is what
[`../tests/test_military_cost.py`](../tests/test_military_cost.py) asserts.

---

## 2. Roskam *Airplane Design Part VIII* — whole-programme cost

Where DAPCA estimates acquisition and stops, Roskam covers RDT&E, acquisition,
operations and disposal in one consistent set of CERs, and exposes judgement factors
DAPCA has no equivalent for.

**AMPR weight** — what the airframer actually builds. Two routes:

$$W_{ampr} = 10^{0.1936 + 0.8645\log_{10} W_{TO}} \qquad\text{or}\qquad W_{ampr} = W_e - \sum_{i=1}^{11} W_i$$

The second subtracts the eleven bought-out groups (wheels and brakes, engines,
starter, cooling fluid, fuel cells, electrical, instruments and avionics, armament,
air conditioning, APU, trapped fuel) and is the one to use when a weight statement
exists.

**Cost escalation** — the year of economics as a formula, not a buried constant:

$$CEF(\text{year}) = 6.31752 + 0.104415\,(\text{year} - 2017)$$

which reads 3.394 for 1989 and 7.153 for 2025.

**RDT&E (Ch. 3)**, with $F_{diff}$, $F_{CAD}$, $F_{mat}$, $F_{obs}$ the judgement factors:

```
MHR_aed  = 0.0396   * W_ampr^0.791 * V^1.526 * N_rdte^0.183 * F_diff * F_CAD
C_dst    = 0.008325 * W_ampr^0.873 * V^1.890 * N_rdte^0.346 * CEF * F_diff
C_ea     = (C_e * N_e + C_avionics) * (N_rdte - N_st)
MHR_man  = 28.984   * W_ampr^0.740 * V^0.543 * N_rdte^0.524 * F_diff
C_mat    = 37.632 * F_mat * W_ampr^0.689 * V^0.624 * N_rdte^0.792 * CEF
MHR_tool = 4.0127   * W_ampr^0.764 * V^0.899 * N_rdte^0.178 * N_r^0.066 * F_diff
C_fto    = 0.001244 * W_ampr^1.160 * V^1.371 * (N_rdte - N_st)^1.281 * CEF * F_diff * F_obs
```

**Every phase total is a closed-form fixed point.** Test facilities, profit and
financing are each a fraction *of the total they belong to*, so

$$C_{RDTE} = \frac{C_{aed} + C_{dst} + C_{fta} + C_{fto} + C_{soft}}{1 - f_{tsf} - f_{pro} - f_{fin}}$$

**Acquisition (Ch. 4)** re-evaluates each CER over the whole programme
($N_{program} = N_m + N_{rdte}$) and subtracts the RDT&E share already spent. That
difference — not a fresh CER — is the production cost, and it is what makes unit cost
fall with quantity.

$$C_{ACQ} = C_{MAN}\,(1 + F_{pro}), \qquad AEP = \frac{C_{ACQ} + C_{RDTE}}{N_m}$$

**Operations (Ch. 6)** charges against the fleet that is actually in service after
reserves and attrition, then grosses up by four overhead fractions the same way.

**Life-cycle (Ch. 2/7):** $LCC = (C_{RDTE} + C_{ACQ} + C_{OPS})/(1 - f_{disp})$.

> **Teaching note — composites have two effects, opposite in sign.**
>
> 1. *Cost per pound rises.* Raymer's DAPCA material factors are 1.1–1.8 for
>    graphite-epoxy and Roskam's $F_{mat}$ runs to 3.0 for carbon composite. **The
>    two textbooks agree.**
> 2. *You need fewer pounds.* Raymer credits roughly 5% off the empty-weight fraction
>    in the **sizing** chapter, and $W_e$ enters every DAPCA CER at an exponent of
>    0.63 to 0.92.
>
> This model takes empty weight as an **input**, so it sees only the first effect.
> For the whole trade, size the aircraft first and feed the lighter $W_e$ in here.
> And beware the Brandt workbook's 0.9 for carbon fibre: it nets the weight credit
> into the cost factor, so using it *and* reducing $W_e$ counts the benefit twice.

---

## 3. COCOMO — software, the path from lines of code to dollars

$$E = B + 0.01\sum SF, \qquad \text{Effort} = A\cdot KSLOC^{E}\prod EM, \qquad T_{dev} = 3.67\,\text{Effort}^{\,0.28 + 0.2(E - B)}$$

with $A = 2.94$, $B = 0.91$ for COCOMO II. The five scale factors move both the effort
and the schedule **exponents**, so they matter more the bigger the programme gets. The
schedule exponent is 0.28 only when $E = B$, i.e. when the scale factors sum to zero.
The seventeen effort multipliers only scale the effort. Basic COCOMO is the same shape
with fixed constants.

The exponent is above one, so **doubling the code more than doubles the cost**. On a
modern combat aircraft this is not a rounding error — software development can run to
a third of the entire RDT&E bill.

---

## 4. Commercial direct operating cost — AIAA 2025-3499

$$DOC = C_{depre} + C_{depBat} + C_{int} + C_{ins} + C_{fc} + C_{cc} + C_{fuel} + C_{elec} + C_{maint} + C_{ldg} + C_{nav} + C_{ETS}$$

| Group | Element | Equation |
|---|---|---|
| Ownership | Depreciation | $(C_{af} + 1.125[C_{eng}n_{eng} + C_{mot}n_{mot}])\dfrac{1 - RV}{D_a U}t_b\,CPI$ |
| | Battery depreciation | $C_{bat}(1 - RV_{bat}) / (\text{cycles}\cdot k_{fast})$, $k_{fast} = -0.139\,C_{rate} + 1.155$ |
| | Interest | level-payment amortisation per flight hour, $\times t_b$ |
| | Insurance | $C_{ac}\,r_{ins}(1 + k_{ele})\,t_b / U$ |
| Crew | Flight / cabin | $286\,t_b$ · $(39.1 + 1.75k_{intl})\,n_{CC}\,t_b$ |
| Energy | Fuel | $\dfrac{m_{fuel}}{\rho}C_{fb}\left[1 + \beta_{SAF}(\delta_{SAF} - 1)\right]$ |
| | Electricity | $E_{mission}\,r_{elec}$ |
| | Carbon | $k_{ETS}\,(3.16\,m_{fuel})(1 - \beta_{SAF})\,k_{eco}$ |
| Maintenance | Airframe | $(15.6 + 3.65\,OEW/1000)\,t_b\,CPI_{1994}$ |
| | Engine, each | $0.106(14.71\,T + 30.5\,t_b + 10.6)\,t_b\,CPI_{2010}$ |
| Fees | Landing | $k_{TANS}(MTOM/50)^{0.7}$ |
| | Navigation | $k_{en\text{-}route}\,(sl/100)\sqrt{MTOM/50}$ |

Reported as $\text{unit cost} = DOC / (n_{pax}\cdot sl)$ in dollars per available seat
kilometre.

Scenarios (Table 14):

| Scenario | $\beta_{SAF}$ | $\delta_{SAF}$ | $k_{ETS}$ [$/ton] | motor maintenance reduction | $C_{rate}$ |
|---|---|---|---|---|---|
| 2030 | 0.06 | 3.00 | 70 | 0.25 | 1.26 |
| 2040 | 0.34 | 2.24 | 130 | 0.50 | 2 |
| 2050 | 0.70 | 2.24 | 500 | 0.75 | 2 |

> **Assumption:** the paper does not publish its block-speed or fuel-burn model, so
> **block time, trip fuel and mission energy are inputs here**, and every element is
> then exactly reproducible.
>
> Get that assumption wrong and you will be far out. Taking the paper's Table-13
> cruise Mach of 0.55, multiplying by a sea-level speed of sound and calling the
> result a block speed gives 674 km/h; the ATR 72-500 actually cruises at about
> 509 km/h and blocks at about 420 once taxi, climb and descent are counted. That one
> substitution moves unit cost by 40%. `methods/aircraft.py` therefore states block
> speed and cruise TAS separately: block speed sets block time, cruise TAS sets the
> equivalent thrust in Eq. 12.

### Two documented typos, implemented as printed

- Eq. 4 is labelled [USD/trip] but the expression is per flight **hour** — the paper's
  own Fig. 8 plots it in USD/h. We multiply by block time.
- Eq. 7 reads $(C_{ccb} + 1.75\,k_{intl})$, an additive 1.75 $/h for international
  operations, though the prose reads like a 1.75× multiplier.

---

## What drives what

### Military

| Parameter | Strongest effect |
|---|---|
| Empty weight $W_e$ | manufacturing (^0.82) and materials (^0.921) — the biggest single driver |
| Maximum velocity $V$ | engineering (^0.894) and development support (^1.3) |
| Quantity $Q$ | totals rise; **unit** cost falls (^0.641, ^0.799) — the learning curve |
| Flight-test aircraft $FTA$ | flight test only (^1.21), a pure non-recurring adder |
| $T_{max}$, $T_{t4}$, $M_{max}$ | engine production cost, linearly |
| Material mix | $D_{47}$ scales all four labour pools at once |
| KSLOC | software effort, superlinearly |
| Treated geometry | low-observables adder, linearly |
| $AF$, $ICF$, $EF$ | pure multipliers on the total |
| Signature level | recurring adder -- moves every per-aircraft figure |
| Dollar year | reporting only: all figures escalated to a single year |

### Commercial

| Parameter | Effect |
|---|---|
| Block time $t_b$ | multiplies depreciation, insurance, both crew terms, both maintenance terms — dominant |
| Utilization $U$ | divides all three ownership terms |
| Stage length $sl$ | navigation fee linearly, and it is the **denominator** of the unit-cost metric |
| Trip fuel $m_{fuel}$ | fuel cost and carbon tax, both linearly |
| $OEW$ / $MTOM$ | airframe maintenance / landing (^0.7) and navigation (^0.5) fees |
| $\beta_{SAF}$, $k_{ETS}$ | the 2030 → 2050 scenario story |

---

## Sources

- Raymer, D. P., *Aircraft Design: A Conceptual Approach*, Ch. 18 (DAPCA IV).
- Brandt et al., *Introduction to Aeronautics: A Design Perspective* — companion
  workbook `Brandt-F16-A.xls`, sheet `Cost`.
- Roskam, J., *Airplane Design Part VIII: Airplane Cost Estimation*, Ch. 2–4, 6–7.
- Boehm, B., *Software Engineering Economics*; Boehm et al., *COCOMO II*.
- Espinosa-Juárez, E., Jouannet, C., Amadori, K., & Sánchez Mata, A., "Comparative
  Analysis on Aircraft Direct Operating Cost Models", AIAA AVIATION 2025,
  [10.2514/6.2025-3499](https://doi.org/10.2514/6.2025-3499).
