# Teaching Guide: Aircraft Cost

A 75-minute session built around the Streamlit app, weighted towards military
programmes. Each block below names what to show, what to say, and what to ask.

```bash
conda activate eng-des-opt-course
python -m streamlit run apps/streamlit/cost_app.py
```

Have [`cost_model_map.md`](cost_model_map.md) open in a second window for the
equations. The app opens in **Military fighter** mode on the F-16A baseline.

**Materials:** the app, the map, and [`ex_04_cost.md`](ex_04_cost.md) for the tables used in §6 and §9.

---

## 0. Before you start (2 min)

Ask the room: *what does an F-16 cost?* You will get numbers between $15 M and
$80 M. Write three or four on the board. Do not resolve it — §3 does.

---

## 1. Cost categories, and the one that matters (8 min)

**Say:**

- **RDT&E** — designing it, and the aircraft you build to test it. Spent once.
- **Production** — building the ones you keep. Spent per airframe.
- **Operations & support** — fuel, crew, maintenance, spares, for decades.
- **Disposal** — small, real, usually forgotten.
- **Life-cycle cost** is all of it. **Direct operating cost** is the airline's
  version: what one trip costs.

**Show:** *Ownership* tab. Sweep service life from 5 to 40 years and watch the O&M
share climb.

**Ask:** *If operations are the majority of life-cycle cost, why does every design
review argue about unit price?*

> The honest answer: unit price is the number in the contract, and O&M is somebody
> else's budget twenty years later. That is a real institutional problem, not a
> modelling one.

---

## 2. A CER is not physics (8 min)

**Show:** the map, DAPCA hours.

$$H_{mfg} = 10.72\,W_e^{0.82}\,V^{0.484}\,Q^{0.641}\,D_{47}$$

**Say:** there is no physics in that exponent. It is a regression through historical
programme data. It is dimensionally meaningless — you cannot check it by units. It is
only valid for the aircraft class and the **dollar year** it was fitted to.

**Show:** *Cost summary* tab. Point at the caption under the title: the CERs work in
one dollar year, the app reports in 2026, and the escalation factor between them is
printed. Then point at the wrap rates in the sidebar — 86/88/73/81 $/hr for the
F-16A, which are 1999 dollars.

**Say:** Raymer publishes the same DAPCA equations twice, once with 1999-dollar lead
coefficients and 1999 wrap rates, once with 2012 ones. The **exponents are
identical**; only the leads and their matching rates differ. Pairing a 1999
coefficient with a 2012 rate is silent and wrong, which is why the pairing is a
property of the aircraft here rather than a switch you can get wrong in the sidebar.

**Ask:** *Your team quotes $45 M. In what year's dollars?* If they cannot answer, the
number means nothing. Switch aircraft to Silver Scythe and note the caption changes
to a different model dollar year — the reported figures stay in 2026 either way.

---

## 3. Resolve the opening question (8 min)

**Show:** *Cost summary* tab, the first table. Seven figures, all in 2026 dollars.

| Figure | F-16A, Q = 200 | What it is |
|---|---|---|
| Recurring unit cost | $70.8 M | the marginal cost of one more airframe |
| Average flyaway cost | $78.4 M | **what a published "flyaway cost" means** |
| Programme unit cost | $95.9 M | the whole programme divided by the buy |
| Acquisition unit cost | see *DAPCA vs Roskam* | production + profit, no RDT&E |
| Life-cycle unit cost | $130.8 M | plus a lifetime of fuel, crew and maintenance |
| Total non-recurring | $5.03 B | spent once, however many you build |
| Total programme | $19.2 B | the whole bill |

**Say:** the numbers on the board are all defensible and all different, because they
are answers to different questions. If someone says "the aircraft costs $96 M" the
only useful reply is *which of these seven do you mean, and in what year's dollars?*

**Show:** *Learning curve* tab. All three per-aircraft figures on one plot. Set the
budget to $85 M and read off the break-even quantity.

**Ask:** *Unit cost falls with quantity. Does the programme get cheaper?* No — the
total rises. Watch the *Total programme* row in the summary table as you change
quantity.

**Ask:** *Why do the three curves converge at high quantity?* The non-recurring cost
is being spread thinner and thinner.

---

## 4. What actually drives your cost (12 min)

**Show:** *Sensitivity* tab. Two charts, every sidebar parameter on both.

**Say:** the tornado chart is in **dollars** — it answers "what moves the most
money". The elasticity chart is **dimensionless** — 0.8 means a 10% rise in that
input raises the cost by 8% — so it puts a production quantity and a labour rate on
the same axis. A tornado chart in dollars cannot do that, because a 20% change in
empty weight and a 20% change in a wrap rate are not comparable quantities.

**Do:** switch the *Cost figure* selector from **Programme unit cost** to
**Life-cycle unit cost** and watch the ranking rearrange. Acquisition is driven by
empty weight, quantity and Mach; ownership is driven by maintenance man-hours per
flight hour, service life and flight hours per year.

**Ask:** *Your team is trying to reduce life-cycle cost. Which three parameters do
you work on?* Nothing on that list is a wing.

> These two charts are the most useful pair in the app for a design team. They tell
> them where to spend their remaining weeks.

---

## 5. Cost as a constraint, not a report (10 min)

**Show:** *Design to cost* tab. Set the budget to your team's requirement.

**Say:** the red contour is the boundary of the affordable design space. Everything
above and to the left meets the budget — lighter airframes and larger buys. At a
500-aircraft buy you can afford about 24,000 lb empty; at 200 aircraft you cannot.

**Ask:** *Your RFP says $85 M and 300 aircraft. What empty weight does that buy you?*
Read it off the contour. *Now your weights team says the aircraft came out 15% heavier.
What quantity do you need to stay affordable?*

> The point: cost belongs in the sizing loop, not in the last slide.

---

## 6. Materials and signature (8 min)

**Show:** *Materials & signature* tab.

**Ask first:** *Do composites make an aircraft cheaper or more expensive?* Let the
room argue. Both answers are defensible and that is the point.

**Say:** there are **two effects, opposite in sign, in two different chapters**.

1. *Cost per pound goes up.* Raymer's DAPCA material factors are 1.1–1.8 for
   graphite-epoxy against 1.0 for aluminium. Roskam's $F_{mat}$ runs to 3.0 for
   carbon composite. **The two textbooks agree on this.**
2. *You need fewer pounds.* Raymer applies a separate ~5% empty-weight credit in the
   **sizing** chapter, and $W_e$ appears in every DAPCA CER at an exponent of 0.63
   to 0.92.

**Say:** this model takes empty weight as an **input**, so it sees only the first
effect. To get the whole trade you have to size the aircraft first and feed the
lighter $W_e$ in here — which is exactly what `ex_01` does.

**Show:** the material-factor plot. Note the Brandt workbook's own table uses 0.9 for
carbon fibre, *below* aluminium, so in this app more composite lowers $D_{47}$. That
is the outlier among the three sources, and it looks like the weight credit netted
into the cost factor.

**Ask:** *If you use the workbook's 0.9 and you also reduce your empty weight for
going composite, what have you done?* Counted the benefit twice.

**Show:** the low-observables table, levels none through C with radome.

**Say:** this is the only model here where a *geometric* parameter turns straight into
dollars — wetted area times $8/ft², edge length times $400/ft, volumetric edge
material at $6,000/ft for Level C. Shorten the treated edge length in the sidebar and
the bill falls. Level A to Level C is a factor of 135, and it shows up in programme
unit cost because every airframe gets treated.

---

## 7. Software (6 min)

**Show:** *Software* tab. Sweep KSLOC.

**Say:** COCOMO II, $\text{Effort} = 2.94\,KSLOC^{E}\prod EM$ with $E > 1$. The
exponent is above one, so doubling the code more than doubles the cost. The five
scale factors move the *exponent*, which is why they matter more the bigger the
programme gets.

**Ask:** *How many lines of code is a modern fighter?* (Millions.) On a programme of
that size software development runs to roughly a third of the entire RDT&E bill.

---

## 8. Two textbooks, one airplane (8 min)

**Show:** *DAPCA vs Roskam* tab, on whichever aircraft you have loaded.

**Say:** five unit prices for the same aircraft. Three are DAPCA figures that differ
only in what they include; two are Roskam's, which is a different method entirely.
They do not agree, and both are published and credible.

**Show:** the Roskam phase table underneath — RDT&E, acquisition, operations,
disposal. Roskam covers the whole programme and exposes judgement factors DAPCA has
no equivalent for: technology aggressiveness, CAD experience, material choice and
low-observability. Between them those four can swing the answer by a factor of two.

**Do:** switch aircraft in the sidebar and watch both methods move together. The
three are the F-16A reference case and the two senior-design programmes, and each
carries its own DAPCA *and* Roskam input set.

**Ask:** *A cost chart with no named method behind it — what is it worth?*

---

## 9. Reading a CER critically (8 min)

Two student senior-design programmes were given to us as reference implementations.
Both cite their sources correctly. Both contain transcription errors.

Put the table from [`ex_04_cost.md`](ex_04_cost.md) §6 on screen. Walk two rows:

- `MHR_aed` coefficient 0.0396 copied as 0.03396 — a dropped digit, 14% off the
  largest RDT&E labour term.
- $C_{ACQ} = C_{MAN}(1 + F_{pro})$ written as $(1 + F_{pro}C_{MAN})$ — a misplaced
  bracket that turns acquisition from 24% of life-cycle cost into 1.5%, and is not
  even dimensionally a cost.

**Say:** none of this is carelessness. It is what happens when you copy forty
equations out of a book. The only defence is to evaluate each relation at a point you
can check by hand.

Then the comparison table in §4 of the same document: DAPCA against five real
aircraft, ratios 0.84 to 1.58. Note the F-16A row — the workbook assumes 200 aircraft,
over 4,600 were built, and fixing only that moves the ratio from 3.10 to 1.27.

**Ask:** *Is a factor-of-two model useful?* Yes — for **comparing** designs. No — for
quoting a price. Raymer says DAPCA "is not the best method for any particular class
but gives reasonable results for most"; Torenbeek warns against comparing model output
with actual airline costs at all.

---

## 10. The commercial companion (if time, 5 min)

Switch to **Passenger transport**.

**Show:** *Stage length* tab. Unit cost collapses with distance because the fixed
per-trip charges are spread over a bigger seat-kilometre denominator.

**Show:** *Scenarios* tab. 2030 → 2050, SAF blending 6% → 70%, carbon $70 → $500/tonne.
Conventional aircraft get steadily more expensive; electrified ones are exempt from the
allowance.

**Say:** block time and trip fuel are **inputs** here, because the paper does not
publish the block-speed or fuel-burn model behind its own results.

**Show the trap.** Take the paper's Table-13 cruise Mach of 0.55, multiply by a
sea-level speed of sound, and use the result as a block speed: 674 km/h. The ATR 72-500
actually cruises at about 509 km/h and *blocks* at about 420 once taxi, climb and
descent are counted. The 674 km/h assumption puts the answer 40% below the paper; the
420 km/h block speed gives 0.113 $/ASK against the paper's 0.11.

**Ask:** *Where else in your own analysis is a cruise number standing in for a block
number?*

---

## Closing (2 min)

Three things to leave with:

1. **Say which number and which year.** Recurring, flyaway, programme unit,
   acquisition, life-cycle - seven different answers, and a dollar year on each.
2. **Cost is a constraint on the design space**, not a slide at the end.
3. **A CER you have not checked at a point is a CER you have not read** - and the
   assumption feeding it matters as much as the equation.

---

## Homework prompts

1. Take your team's current aircraft. Put its empty weight, Mach, thrust and planned
   buy into the app. Report *all three* per-aircraft numbers and say which one your
   RFP is asking for.
2. Produce the tornado chart for your aircraft against life-cycle cost. Name the three
   parameters you would work on next and say why.
3. Your programme is 15% over its unit-cost requirement. Using the design-to-cost
   contour, give two ways to close the gap and state what each costs you elsewhere.
4. Pick one CER from [`cost_model_map.md`](cost_model_map.md). Evaluate it by hand at
   your aircraft's numbers and confirm the code agrees. Show your arithmetic.
