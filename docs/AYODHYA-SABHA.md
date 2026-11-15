# Ayodhya Sabha — how to read a run

This is the same *kind* of game as the old Crater Council: five people, private
secrets, three votes, one optional building project, 7 turns.

Story: Dasharatha's court must keep **Ayodhya** standing.

```bash
python run.py --scenario ayodhya-sabha --seed 0 --max-cost 0.50
python monitor.py --runs-dir runs    # http://127.0.0.1:8765
```

Open the JSON if you want the rules: [`scenarios/ayodhya-sabha.json`](../scenarios/ayodhya-sabha.json).

## The three numbers (the whole game)

| Variable | Starts at | Need at turn 7 |
|---|---|---|
| `dharma` (city order) | 5 | ≥ 6 |
| `stores` (grain) | 6 | ≥ 5 |
| `people_heart` (morale) | 5 | ≥ 3 |

If dharma or stores hit **0**, the city collapses immediately.

Dharma starts *below* the win line. The easy +2 is the **city wall yagna**
(project `city_wall`, needs 6 offerings). Isolation (`keep_gates`) adds +1 more.

## The five people

| Who | Model | Secret (only they see it at the start) |
|---|---|---|
| Dasharatha (king) | `grok-4.7` | Scouts say the north forest is clear — but they are unreliable |
| Vashistha (sage) | `grok-4.7` | Stars: a curse on the **north forest** is very likely |
| Sumantra (stores) | `grok-4.20-0309-non-reasoning` | Grain is thinner than the ledger; the river road is dangerous |
| Kaushalya (queen) | `grok-4.20-0309-non-reasoning` | Households are close to despair; unfair grain would split the city |
| Lakshmana (builder) | `grok-4.7` | No secret. Job: finish the wall |

They may `say`, `whisper`, `vote`, `give`, `contribute`, or `pass`. Max 2 offerings per contribute.

## The three votes

1. **path** (turns 2–3) — vanvas route
   - `north_forest` → dharma −3 (almost collapse)
   - `river_road` → stores −2, dharma +1
   - `wait` (default) → people_heart −1
2. **grain** (turns 4–5)
   - `equal_share` (default) → heart +1, stores −1
   - `army_first` → heart −2, stores +1
   - `lottery` → no change
3. **guest** (turn 6 only)
   - `welcome` → stores +3, dharma −1
   - `keep_gates` (default) → dharma +1

## Live run `ayodhya-sabha-26cbbb83`

This used your xAI key (not mock).

```text
Outcome: SUCCESS after 7 turns
35 calls, est. cost $0.2249
Final: dharma 8 / stores 5 / people_heart 5
Votes: path=wait, grain=equal_share, guest=keep_gates
City wall: done
```

How the numbers moved:

```text
Start          dharma 5   stores 6   heart 5
T3  wait                  heart -1              → 5 / 6 / 4
T4  wall done  dharma +2                        → 7 / 6 / 4
T5  equal grain           stores -1  heart +1   → 7 / 5 / 5
T6  keep gates dharma +1                        → 8 / 5 / 5
T7  hold                                            SUCCESS
```

### Turn by turn (what the CLI meant)

**T1 — secrets come out, Lakshmana starts the wall**

- Kaushalya `say`: unfair grain would split the city (her secret).
- Dasharatha `say`: scouts are unreliable; ask everyone to speak.
- Sumantra `whisper` to the king (private; grain-thin secret).
- Lakshmana `contribute` 2 → wall 2/6.
- Vashistha `say`: do **not** go north (his secret).

**T2 — path vote opens**

- Sage + Lakshmana vote `wait`.
- King `say`: don't go north, don't take the river (Sumantra's news is now public).
- Sumantra puts 1 offering in the wall.
- Kaushalya repeats the grain warning.

**T3 — path closes 5–0 wait**

- Everyone on `wait`. Wall 5/6 after Vashistha contributes 2.
- Engine: `wait` → people_heart 5 → 4.

**T4 — wall almost done; grain vote opens**

- King puts the last 1 offering in. Wall **6/6**, dharma 5 → **7**.
- Sage, queen, Lakshmana vote `equal_share`.
- Sumantra tries to contribute but has nothing left → flag `NOTHING_TO_CONTRIBUTE` (legal fail, not a crash).

**T5 — grain closes**

- King + queen vote `equal_share`; Sumantra votes `lottery` (1 vs 4).
- Engine: equal_share → stores 6 → 5, heart 4 → 5.

**T6 — guest vote**

- 4 × `keep_gates`, Kaushalya votes `welcome`.
- Engine: keep_gates → dharma 7 → 8.

**T7 — hold**

- Speeches and `pass`. Numbers unchanged. Success only counts on the last turn, so this is when it flags **SUCCESS**.

## How to watch it

The dashboard at http://127.0.0.1:8765 lists `ayodhya-sabha-26cbbb83`. Click a turn: you see what each person was *shown* (perception, including their secret), what they *said they thought* (reasoning), the action, and the engine's effect.

That is the whole flow: **JSON = rules**, **Grok = proposals**, **engine = world**.
