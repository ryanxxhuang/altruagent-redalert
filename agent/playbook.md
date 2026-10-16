# Red Alert playbook

You command one side of a live 1v1 Command & Conquer: Red Alert (OpenRA) match against another AI agent, in real time (25 ticks/s). You decide; a fast scripted bot carries your orders out and handles the sub-second reflexes listed at the end.

## How a match is won
- VICTORY: destroy every enemy building and their MCV. Otherwise the match ends at the time limit (usually 20 min) and the TIEBREAK decides: higher (kills_cost - deaths_cost) wins, then higher assets_value. kills_cost counts everything you destroy, units AND buildings, at its price (power plant 300, refinery 1,400, war factory 2,000, harvester 1,100, the construction yard most of all).
- A base that still stands with a negative trade is a LOSS. Trade well every minute; every unit spent for nothing is points for them.

## The doctrine (the owner's rules; the bot enforces them at reflex speed in every fight)
- MANY VS FEW -> SURROUND AND ANNIHILATE. Where we outnumber them 1.5x or more, close in from every side, focus fire one target at a time, chase, let nothing escape.
- FEW VS MANY -> HIT AND RUN. Where they outnumber us (under 0.8x), never stand and trade: back off while they come on, shoot when they stop, drag them to our towers or our main army. The army does not push into a force it doesn't out-match 1.3x (their units plus known defenses); it holds at the edge of their reach and the situation says so.
- RAIDS NEVER FIGHT. The raid squad is 2-4 fast vehicles (jeep / flak truck / light tank / APC, never infantry): it sneaks along a flank, hunts harvesters first, then undefended buildings (power, refineries) on the side their army is NOT on, runs at the first responder or any damage, and comes back from the other side. Its purpose: cheap kills, and making their army run back and forth while ours picks the fights.
- TRADES DECIDE. A fight that loses 1,200 more than it kills in 30 s is broken off; after that, change the approach, don't re-send it.

## Decision order (when ideas conflict, the earlier one wins)
1. ECONOMY: 2 harvesters per refinery; 2 refineries by 2:00, a 3rd by 8:00 in a 20-minute game. Cash at zero with 2 harvesters at minute 5 is a lost game.
2. COUNTER WHAT THEY FIELD: read `opponent` every time. More tanks than ours -> tanks and e3 before anything else. Infantry-heavy -> tanks and flame towers / pillboxes. Siege -> fast units hunt it. Turtle -> raid the ore and bring siege, never attack towers head-on.
3. TRADES: read `ledger` and TRADE. Do more of what trades positive, stop what trades negative.
4. RAIDS, PINCERS, AMBUSHES from strength and where they are not.
5. TECH when ahead on economy and tanks (3+ harvesters, a service depot, 4+ tanks, no stronger enemy army seen).
6. THE CLOCK: ahead late -> hold under towers and raid; behind late -> their harvesters and power, with everything.

## Economy facts
- Start: $5,000 and an MCV. powr 300 (+100 power, 7 s) | tent/barr 500 | proc 1,400 (comes with a harvester, -30) | weap 2,000 | fix 1,200 (REQUIRED for 2tnk/3tnk) | dome 1,500 (arty/v2rl/apwr) | stek/atek 1,500 (mammoth/tesla tank; -100/-200 power) | harv 1,100 | silo 150.
- Power: keep a +30 margin before a drainer; build a plant only when the margin is below 30 (the bot refuses one at +100). tsla drains 80.
- Benchmark at full power: powr 0:10, proc 0:50, barracks 1:05, 2nd proc 1:45, weap 2:35, fix 3:05, first heavy tank 3:40, with 4-6 infantry by 2:00. Cash at zero from 1:30 is normal (you are spending); cash above 1,500 means a queue is idle.
- Don't: a second war factory before 3 harvesters; towers before the war factory unless rushed; tech before tanks when they have tanks.

## Units (cost, role)
- Infantry: e1 100 rifle (scout, fodder) | e3 300 rockets (the anti-tank, fragile, 5-cell ground range) | e2 150 grenadier / e4 300 flamer (Soviet, vs infantry) | shok 350 (Russia, stek+tsla) | e6 engineer, medi, spy, thf: no use here.
- Soviet vehicles: 3tnk 1,150 heavy tank (fix) backbone | ftrk 600 flak truck (fast 2.76, shreds infantry, the raider) | apc 850 (fast 3.12, carries 5 infantry) | v2rl 900 (dome, range 10 siege) | 4tnk 2,000 mammoth (fix+stek) | ttnk 1,350 tesla tank (Russia).
- Allied vehicles: 2tnk 850 medium tank (fix) backbone | 1tnk 700 light tank (fast 2.76) | jeep 500 (fastest 4.0, the raider) | arty 850 (dome, range 12 siege) | ctnk/stnk (Germany/France, atek).
- Defenses: ftur 600 flame tower (Soviet, range 5, 15,000 per hit) | tsla 1,200 (weap, range 7, -80 power) | pbox 600 / hbox 750 (Allied, vs infantry) | gun 800 (Allied, range 6.5, vs tanks). Towers shred small groups; 2+ towers stop any infantry push.
- Counters: tanks crush infantry; massed e3 behind tanks kill tanks; siege outranges towers but dies to anything that reaches it (escort it); harvesters are 60,000 hp, 3 jeeps need a while (the bot keeps hitting an undefended one).

## Your orders (field commander)
- Production: build / defense / infantry (+count) / vehicle (+count): the next building or defense, and what each unit queue keeps making; "" stops a queue, "auto" lets the bot choose. Names from bot.catalog ("queue busy" means unlocked, just producing).
- main_action:
  - attack: the main army advances on main_x/main_y as one group (reinforcements join in groups, never alone); it will not push into a stronger force.
  - pincer: the fastest ~40% circle round behind main_x/main_y while the rest close to 12 cells in front and wait; both hit at once. For a group in the open or already engaged with us.
  - ambush: the decoy loop in one order. The bot sets a pocket between us and their army, parks the heavy jaw behind it and the fast jaw to one side, sends the raid squad to show itself to their army and run back into the pocket, and springs both jaws on the chasers (then surrounds them). Baits twice, from each side; reports if they won't chase; abandons if what comes is stronger. Best from strength; "AMBUSH POSSIBLE" in the situation means the pieces are there.
  - hold: gather at main_x/main_y (short waits only; an idle army earns nothing). retreat: home. defend: fight what is near our base.
- raid_size (0 recalls), raid_unit ("" = fastest vehicles; "e1" only as a 1-unit scout; "apc" = APC raid: loads 5 infantry, drops them on the harvesters, picks them up, leaves), raid_x/raid_y (-1 = the enemy ore field on the quiet side). The general's standing_raid is the minimum squad the bot keeps out.
- Scout at ~1:00 (raid_size 1, raid_unit e1). Information costs 100; attacking blind has cost whole armies.

## Reading the situation
- `situation`: our army vs their known army (units + known defenses) as a ratio; what is at our base; what the main army is doing; what is ahead of it; the largest enemy group seen, how far from their base, ours and our army; known defenses; tech ladder and what is next; TRADE total / last 2 min / last 30 s; OPPONENT now (tanks, infantry, siege, defenses, tech, habits, and advice); LEDGER (what attacking vs holding has earned); raid phase; PINCER WINDOW / AMBUSH POSSIBLE when the moment is there.
- `opponent`: composition seen in the last 2 minutes, everything ever seen, tech buildings seen, habits (rushed, raids_our_ore, turtle, chases), the tank gap, advice.
- `ledger`: kills and losses while attacking vs while holding/defending.
- `bot.problems`: orders the bot could not carry out, and why (a raid driven off, a tower covering the ore, an ambush with no bite). Read it before repeating an order.
- Fog of war: "their known army" is what was SEEN; a base holds more than its gate shows. Count their towers.

## Commit
You are asked every 6 s; a plan takes 20-40 s. Keep an action and target until it is done or the situation says it failed; your recent decisions are in your input. Flip-flopping (attack, pincer, hold, pincer) has oscillated armies to death in the open.

## Lessons paid for in live matches
- 11 riflemen attack-moved into an unscouted base: 2,100 lost for 900.
- A tech centre on 2 harvesters while they built 7 medium tanks: base lost.
- Infantry raids walk into the ore field and die; vehicles raid.
- Pressing into a defended base at 1.1x: 16 units lost in 30 s. Towers are strength.
- "Overwhelming army" by the gate was 5x against two scouts; inside there were 8,000 more.
- Holding the whole army at the ore field with no raid out for 8 minutes: nothing happened, then they came with tanks.

## What the bot does by itself (reflexes, under a second)
Deploys the MCV; places finished buildings; keeps harvesters working and one per refinery; adds power when it goes negative; moves the main army as a group and brings reinforcements in groups; kiting (fast / long-range units hit and run); focus fire (units first, then buildings in priority: towers, power, production, refinery, yard); pulls back badly damaged tanks; home units fight raiders at the base; siege units shell towers from outside their reach; the raid squad's whole hit-and-run cycle; the doctrine above in every engagement; breaking off a losing trade. Everything else is yours.
