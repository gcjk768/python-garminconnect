# Telegram messages

Everything the monitor sends to Telegram is rendered by `garmin_health_monitor/messages.py`.
This page lists every message type, when it is sent, what it contains and a realistic example,
so you know what to expect on your phone before you deploy it on the NAS.

> **Layout (2026-10-01):** messages now use the HTML card style — `emoji <b>TITLE</b> · name · date`
> header, blank-line blocks, a `━━━━━━━━━━━━━━━━` divider and coaching/fitness collapsed in an
> expandable quote at the end. The examples below show the same fields in the older layout.

## Conventions

| Convention | Meaning |
|---|---|
| **Times** | Always in the profile's timezone (`profiles[].timezone`, falling back to the global `timezone`). |
| **`n/a`** | Garmin did not report the value (watch not worn, not synced yet, feature not supported). A missing value never blocks a message. |
| **`(7-day avg 58, ▲4)`** | Today's value versus the average of the previous 7 stored days. `▲` above, `▼` below, `±0` the same. Weekly review compares with the previous week (`prev`). |
| **`▰▰▰▰▰▰▱▱▱▱`** | Progress bar towards a goal (steps). |
| **`✅ felt` / `❌ not noticed` / `❓ not answered`** | The person's answer to an episode alert (inline buttons) - it goes into the doctor report. |
| **`… and N more`** | Long lists are cut so every message stays under Telegram's 4096-character limit. |
| **Emoji anchors** | 🌅 morning · 🌙 evening · 📊 weekly · 📍 live status · ❤️ heart · 😴 sleep · 👟 steps · 🔋 body battery · 🧘 stress · 🧭 coaching · 🩺 doctor report / diary · ⚠️ 🚨 alerts · 🔐 login |

Every message that touches the heart-rate diary ends with a plain reminder: the watch is an
optical wrist sensor, the diary is for discussing with a doctor, it is **not a diagnosis**, and
chest pain, fainting or breathlessness need urgent care.  Coaching blocks carry a matching
"not medical advice" line.  Numbers in messages come from Garmin or the detector; the AI model (Claude Code CLI or Ollama)
only writes the coaching text and the episode assessment, never the numbers.

## Overview

### Scheduled (per profile, `schedule:` in `config.yaml`)

| Message | When | Feature flag | Contents |
|---|---|---|---|
| 🌅 Morning brief | `schedule.morning_brief` (default 07:30) | `features.morning_brief` | Last night's sleep, HRV, resting HR, body battery on waking, readiness, lowest SpO2, overnight episodes, coaching for the day |
| 🌙 Evening summary | `schedule.evening_summary` (default 21:00) | `features.evening_summary` | Steps vs goal, distance/floors/calories, active & intensity minutes, stress, body battery, activities, episodes and symptoms of the day, coaching |
| 📊 Weekly review | `schedule.weekly_review_day` + `_time` (default Sun 19:00) | `features.weekly_review` | 7-day table, totals/averages vs previous week, episodes per day, weekly coaching |
| 🩺 Doctor report | `schedule.doctor_report_day_of_month` + `_time` (default 1st, 09:00) | `features.doctor_report` | PDF + CSV attachment with a short caption |

### Event-driven

| Message | When | Feature flag |
|---|---|---|
| ❤️ Episode alert | A new at-rest heart-rate excursion is detected during a poll (every `schedule.poll_minutes`) and its heuristic confidence is ≥ `palpitations.alert_min_confidence`. Comes with a zoomed heart-rate chart and two buttons. | `features.palpitations` + `palpitations.notify` |
| ⚠️ / 🚨 / ℹ️ Alert | The rules engine fires (resting HR well above the 7-day average, low SpO2, low body battery, poor HRV status, watch not synced for `alerts.no_sync_hours`, very low steps in the afternoon, Garmin's own abnormal-HR alert count rising). Each alert is sent once per day per rule. | `features.alerts` |
| 🔐 MFA request | Garmin Connect asks for a two-factor code at login (usually only the first time; tokens are cached afterwards). Sent to admin chats. | - |

### On demand (commands)

| Command | Reply | Notes |
|---|---|---|
| `/start`, `/help` | 🤖 Command list | Admin chats also see `/status` and `/mfa` |
| `/profiles` | Which people this chat can see | |
| `/today` | 📍 Live snapshot | Steps so far, latest body battery, last HR reading, sync state (warning if > 3 h without a sync) |
| `/yesterday` | 🌙 Evening summary for yesterday | Same layout as the scheduled one |
| `/sleep` | 😴 Sleep detail | Last night |
| `/hr [YYYY-MM-DD]` | ❤️ Heart-rate chart + caption | Caption lists min/max/resting and at-rest excursions |
| `/steps` | 👟 Steps | Today vs goal and 7-day average, last 7 days as bars |
| `/episodes [days]` | 🩺 Palpitation diary | Default 7 days: episodes, reported symptoms, frequency summary |
| `/palp [HH:MM] [note]` | 📝 Symptom logged | Records a palpitation you felt (now, or at HH:MM today) with the heart rate near that time |
| `/note <text>` | 📝 Symptom logged | Free-text note into the diary |
| `/report [days]` | 🩺 Doctor report PDF + CSV | Default 30 days |
| `/analyze` | 🧭 Coaching block | Asks the AI model (Claude Code CLI or Ollama) for today's coaching right away |
| `/status` (admin) | Service health | Last poll per profile, model availability, sync age |
| `/mfa <code>` (admin) | Confirmation | Passes the Garmin verification code to the waiting login |

When a chat can see more than one person (admin chats, or a shared family chat), name the
person first: `/today Dad`, `/episodes Dad 14`.

---

## Message catalogue

### 🌅 Morning brief

**Sent:** every morning at `schedule.morning_brief`, after the first poll of the day has pulled
last night's sleep.  **Function:** `morning_brief(profile, today, yesterday_row, overnight_episodes, coaching)`.

**Contains:** sleep duration with bed/wake times, sleep score, stage breakdown, overnight HRV
versus the weekly average with Garmin's status, resting HR versus the 7-day average, body battery
on waking (the reading closest to wake-up), training readiness, lowest overnight SpO2, one line
about yesterday, overnight / early-morning heart-rate excursions if any, then the coaching block.

```
🌅 Good morning, Dad — Sun 27 Sep

😴 Last night
Slept 7h 40m (22:30–06:30) · score 78 (Good)
Deep 1h 36m · Light 4h 00m · REM 1h 45m · Awake 20m
💓 HRV overnight: 42 ms (weekly avg 44 ms, ▼2 ms) — Balanced
❤️ Resting HR: 58 bpm (7-day avg 57, ▲1)
🔋 Body battery on waking: 96
🎯 Training readiness: 62 (Moderate)
🫁 Lowest SpO2 overnight: 92%
📅 Yesterday: 2,586 steps · RHR 57 · stress 28
❤️ No at-rest heart-rate excursions overnight.

🧭 Coaching for today (llama3.1:8b)
A steady day with a short walk and a good night's sleep.
✅ Do more
• Take a second 15-minute walk after lunch
• Keep the 22:30 bedtime
⛔ Do less
• Long sitting stretches in the afternoon
👀 Watch out
• Resting heart rate is 4 bpm above the weekly average
❤️ Two at-rest heart-rate rises today; both are in the diary.
AI suggestions generated from watch data, not medical advice.
```

If the watch was not worn overnight the sleep block reads
`No sleep data yet (watch not synced or not worn overnight).` and the HRV / SpO2 lines show `n/a`.

### 🌙 Evening summary

**Sent:** every evening at `schedule.evening_summary`; also the reply to `/yesterday`.
**Function:** `evening_summary(profile, today, rows_7d, episodes_today, symptoms_today, coaching)`.

**Contains:** steps with a progress bar and the delta versus the 7-day average, distance, floors,
calories, active time and intensity minutes, stress average and high-stress minutes, body battery
high/low/now, resting HR, last night's sleep, recorded activities (name, duration, average HR).
When `features.palpitations` is on: the day's at-rest heart-rate excursions (time, duration, peak,
felt/not-felt marker, model assessment) and the symptoms reported with `/palp`.  Then the coaching
block ("do more / do less" for tomorrow).

```
🌙 Evening summary — Dad — Sun 27 Sep

👟 Steps ▰▰▰▰▰▰▱▱▱▱ 3,757 / 6,000 (63%)
7-day avg 3,046 · today ▲711
Distance 2.6 km · Floors 6 · Calories 1,900 kcal (active 350)
Active time 1h 26m · Intensity minutes 35 (weekly goal 150)

🧘 Stress avg 28 (7-day avg 28, ±0) · high stress 20m · Balanced
🔋 Body battery high 85 (7-day avg 85, ±0) / low 25 · now 40
❤️ Resting HR: 58 bpm (7-day avg 58, ±0)
😴 Sleep last night 7h 40m · score 78 (7-day avg 7.7h, ±0.0h)

🏃 Activities (1)
• Morning Walk · 30m · avg HR 105 bpm

❤️ At-rest heart-rate excursions today: 2
• 10:00–10:08 · 8m · peak 127 bpm · ✅ felt · possible palpitation
• 15:30–15:36 · 6m · peak 114 bpm · ❌ not noticed · not assessed yet
📝 Reported symptoms today: 1
• 14:10 — “fluttering after lunch” · HR 98 bpm

🧭 Coaching for today (llama3.1:8b)
A steady day with a short walk and a good night's sleep.
✅ Do more
• Take a second 15-minute walk after lunch
• Keep the 22:30 bedtime
⛔ Do less
• Long sitting stretches in the afternoon
👀 Watch out
• Resting heart rate is 4 bpm above the weekly average
❤️ Two at-rest heart-rate rises today; both are in the diary.
AI suggestions generated from watch data, not medical advice.
```

### 📊 Weekly review

**Sent:** once a week (`schedule.weekly_review_day` / `weekly_review_time`).
**Function:** `weekly_review(profile, rows_this_week, rows_prev_week, episodes, coaching)`.

**Contains:** a compact fixed-width table (day, steps, sleep hours, resting HR, stress, body
battery high), totals and averages compared with the previous week using ▲▼ arrows, the number of
at-rest heart-rate excursions this week with a per-day breakdown and how many were felt, and the
weekly coaching block.

```
📊 Weekly review — Dad — Mon 21 Sep – Sun 27 Sep

Day     Steps Sleep  RHR  Str   BB
Mon 21   3278  7.7h   58   28   85
Tue 22   2131  7.7h   57   28   85
Wed 23   3768  7.7h   56   28   85
Thu 24   3925  7.7h   59   28   85
Fri 25   2889  7.7h   58   28   85
Sat 26   2586  7.7h   57   28   85
Sun 27   2945  7.7h   56   28   85

Versus the previous week
👟 Steps total 21,522 (prev 22,708, ▼1,186) · avg 3,075/day
😴 Sleep avg 7.7h (prev 7.7h, ±0.0h)
❤️ Resting HR avg 57 bpm (prev 57, ±0)
🧘 Stress avg 28 (prev 28, ±0)
🔋 Body battery high avg 85 (prev 85, ±0)
💓 HRV avg 42 ms (prev 42 ms, ±0 ms)
🏃 Activities 3 (prev 2, ▲1)

❤️ At-rest heart-rate excursions this week: 2 (felt 1)
Sun 27 ×2
See /episodes 7 for the list, /report 30 for the doctor PDF.

🧭 Coaching for this week (llama3.1:8b)
Steps were up and sleep was steady.
✅ Do more
• Keep the morning walks
⛔ Do less
• Late evening screens
AI suggestions generated from watch data, not medical advice.
```

### 📍 Live status (`/today`)

**Sent:** on request.  **Function:** `today_status(profile, snap, episodes_today)`.

**Contains:** steps so far with the bar, the latest body-battery reading and its time, the last
heart-rate reading (time + value) and today's resting HR, stress average, activities so far,
episodes so far, the last sync time and device name.  If the watch has not synced for more than
3 hours a warning line is added - the usual reason is that the phone's Garmin Connect app is not
running or Bluetooth is off.

```
📍 Right now — Dad — Sun 27 Sep 17:05

👟 Steps so far: ▰▰▰▰▰▰▱▱▱▱ 3,757 / 6,000 (63%)
🔋 Body battery: 41 (at 16:57)
❤️ Last HR: 67 bpm at 16:58 · resting 58 (7-day avg 57, ▲1)
🧘 Stress avg: 28
🏃 Activities so far (1):
• Morning Walk · 30m · avg HR 105 bpm
❤️ At-rest heart-rate excursions today: 2
• 10:00–10:08 · 8m · peak 127 bpm · ✅ felt
• 15:30–15:36 · 6m · peak 114 bpm · ❌ not noticed

📡 Last sync: 11:30 (Venu 3)
⚠️ Watch has not synced for 5h 35m — check it is worn and Garmin Connect is open on the phone.
```

### 😴 Sleep (`/sleep`)

**Function:** `sleep_message(profile, snap)`.  Bed and wake times, time asleep, score and
qualifier, stages with percentages, awakenings and restless moments, sleeping heart rate, HRV,
SpO2, respiration and the overnight body-battery change.

```
😴 Sleep — Dad — night to Sun 27 Sep

Bed 22:30 → wake 06:30 · 7h 40m asleep
Score 78 (Good)
Deep 1h 36m (21%) · Light 4h 00m (52%) · REM 1h 45m (23%) · Awake 20m
Woke 2 times · restless moments 18
❤️ Sleeping HR 56 bpm · HRV 42 ms (Balanced)
🫁 SpO2 avg 95% · lowest 91% · respiration 13.0 brpm
🔋 Body battery +55 overnight
```

### ❤️ Heart rate (`/hr [YYYY-MM-DD]`)

**Function:** `hr_message(profile, snap, episodes)`.  Sent as the caption of the day's
heart-rate chart (so it is kept under 1000 characters): resting HR versus the 7-day average,
min/max, number of readings and the last one, Garmin's own abnormal-HR alert count, and the
at-rest excursions of that day with peak, baseline, type and the felt marker.

```
❤️ Heart rate — Dad — Sun 27 Sep
Resting 58 bpm (7-day avg 57, ▲1)
Min 50 · Max 127 · 705 readings (last 23:58: 67 bpm)
⚠️ Watch abnormal-HR alerts: 2
At-rest excursions: 2
• 10:00–10:08 · 8m · peak 127 (base 64, +63) · sustained rise · ✅ felt
• 15:30–15:36 · 6m · peak 114 (base 67, +47) · sustained rise · ❌ not noticed
```

### 👟 Steps (`/steps`)

**Function:** `steps_message(profile, snap, rows_7d)`.  Today versus the goal and the 7-day
average, how many steps are left, distance / floors / active time, and the last 7 days as bars.

```
👟 Steps — Dad — Sun 27 Sep

▰▰▰▰▰▰▱▱▱▱ 3,757 / 6,000 (63%)
7-day avg 3,046 · today ▲711
2,243 more to reach the goal.
Distance 2.6 km · Floors 6 · Active time 1h 26m

Sun 20   2,743 ▰▰▰▰▱▱▱▱
Mon 21   3,278 ▰▰▰▰▱▱▱▱
Tue 22   2,131 ▰▰▰▱▱▱▱▱
Wed 23   3,768 ▰▰▰▰▰▱▱▱
Thu 24   3,925 ▰▰▰▰▰▱▱▱
Fri 25   2,889 ▰▰▰▰▱▱▱▱
Sat 26   2,586 ▰▰▰▱▱▱▱▱
Goal met on 0 of the last 7 days.
```

### ❤️ Episode alert

**Sent:** when a poll detects a new at-rest heart-rate excursion whose heuristic confidence is
at least `palpitations.alert_min_confidence` (default 0.45).  Outside `palpitations.quiet_hours`
the alert goes out immediately; inside them it is held until the morning.  A zoomed chart of the
45 minutes around the episode is attached, and two inline buttons **"I felt it"** / **"Didn't
notice"** are added by the bot; the answer is stored on the episode and shows up as ✅ / ❌ in every
list and in the doctor report.  **Function:** `episode_alert(profile, ep, assessment)`.

**Contains:** local time range and duration, peak versus the at-rest baseline, the onset jump
between consecutive 2-minute readings, context (asleep or at rest, steps in the surrounding
15-minute bucket, Garmin activity level, stress), the detector's confidence as Low / Medium / High,
the AI model (Claude Code CLI or Ollama)'s assessment (`possible palpitation`, `likely exertion or movement`,
`likely sensor artifact`, `unclear`) with its one-line reasoning and the factual doctor note, the
"Was it felt?" prompt and the safety reminder.

```
❤️ Possible palpitation episode (Dad)

🕒 Sun 27 Sep 10:00–10:08 · 8m
📈 Peak 127 bpm vs baseline 64 bpm (+63) · average 125 bpm
⚡ Onset jump: +60 bpm between readings
🧭 Context: at rest, 40 steps in the 15-min window, watch level: Sedentary, stress 31 · type: sustained rise
🎯 Heuristic confidence: High (0.75)

🤖 Model view: possible palpitation (confidence 0.70) (llama3.1:8b)
Heart rate rose from 64 to 127 bpm within two minutes while the watch recorded almost no steps, and stayed high for 8 minutes.
🩺 Doctor note: 8 min at up to 127 bpm while seated (baseline 64), 10:00 local.

Was it felt? Tap a button below — it goes into the doctor report.
Wrist-sensor heart-rate readings for a symptom diary, not a diagnosis. Seek urgent care for chest pain, fainting or breathlessness.

        [ ✅ I felt it ]   [ ❌ Didn't notice ]
```

Episode types: **sustained rise** (several minutes above the threshold while at rest),
**sudden spike** (a single very high reading or a large jump), **during sleep** (nocturnal, lower
thresholds).  Remember the watch samples every 2 minutes, so short flutters that last seconds are
invisible to it - that is exactly why `/palp` exists.

### 🩺 Palpitation diary (`/episodes [days]`)

**Function:** `episodes_list(profile, episodes, symptoms, days)`.  One line per detected episode
(newest first: date, time range, duration, peak, felt marker, assessment), the symptoms reported
with `/palp` and `/note` (time, note, heart rate near that time), and a frequency summary: episodes
per week, the most common hour of day, how many happened during sleep, and the felt / not noticed
/ unanswered counts.  This is the quick version of what the doctor report contains.

```
🩺 Palpitation diary — Dad — last 7 days

Detected episodes: 2
• Sun 27 Sep 15:30–15:36 · 6m · peak 114 bpm · ❌ not noticed · not assessed yet
• Sun 27 Sep 10:00–10:08 · 8m · peak 127 bpm · ✅ felt · possible palpitation

📝 Reported symptoms: 1
• Sun 27 Sep 14:10 — “fluttering after lunch” · HR 98 bpm

📈 Frequency: about 2.0 per week · most common hour 15:00–16:00 (1 of 2) · 0 during sleep · felt 1, not noticed 1, unanswered 0

Use /report 30 for the doctor PDF + CSV. Wrist-sensor heart-rate readings for a symptom diary, not a diagnosis. Seek urgent care for chest pain, fainting or breathlessness.
```

With nothing recorded the reply is `Nothing recorded in the last 7 days.` plus a reminder of the
`/palp HH:MM note` syntax.

### 📝 Symptom logged (`/palp`, `/note`, button answers)

**Function:** `symptom_logged(profile, rep)`.  Confirms the time, the note, the heart rate the
watch recorded within a few minutes of that time (with the at-rest baseline), and the detected
episode it was linked to if one overlaps.

```
📝 Symptom logged — Dad
🕒 Sun 27 Sep 14:10
Note: “fluttering after lunch”
❤️ Heart rate near that time: 98 bpm (at-rest baseline 64, ▲34)

Saved to the doctor report. If it keeps happening or feels worse, contact your doctor.
```

If the watch has not synced yet the HR line says `n/a (no reading within a few minutes — it may
appear after the next sync)`; the service fills it in later.

### ⚠️ Alerts (rules engine)

**Function:** `alert_message(alert)`.  Severity emoji (ℹ️ info, ⚠️ warning, 🚨 critical), a
title with the person's name, and a one-paragraph body with the numbers that triggered it.  Rules
and thresholds live under `alerts:` in the profile config.

```
⚠️ Resting heart rate above usual (Dad)
Resting HR 68 bpm today, 10 above the 7-day average of 58.
```

Examples of what can trigger one: resting HR ≥ `alerts.rhr_above_7d_avg` bpm above the 7-day
average or above `alerts.rhr_absolute_high`; lowest SpO2 below `alerts.spo2_below`; body battery
below `alerts.body_battery_below`; HRV status in `alerts.hrv_alert_statuses`; no sync for
`alerts.no_sync_hours`; fewer than `alerts.sedentary_nudge_steps` steps by
`alerts.sedentary_nudge_hour`; Garmin's abnormal-HR alert count going up.

### 🩺 Doctor report caption

**Sent:** with the PDF + CSV, monthly or on `/report [days]`.
**Function:** `doctor_report_caption(profile, days, n_episodes, n_symptoms)`.

```
🩺 Doctor report — Dad — last 30 days
12 detected episode(s), 3 reported symptom(s).
Inside: how often, what time of day, asleep vs awake, felt vs detected, heart-rate charts.
Wrist-sensor heart-rate readings for a symptom diary, not a diagnosis. Seek urgent care for chest pain, fainting or breathlessness.
```

### 🤖 Help (`/help`, `/start`)

**Function:** `help_text(is_admin)`.

```
🤖 Garmin Health Monitor — commands

/today — live snapshot: steps, body battery, last heart rate, sync
/yesterday — yesterday's full summary
/sleep — last night's sleep in detail
/hr [YYYY-MM-DD] — heart-rate chart with at-rest excursions
/steps — steps today vs goal and 7-day average
/episodes [days] — palpitation diary (default 7 days)
/palp [HH:MM] [note] — log a palpitation you felt (now, or at HH:MM)
/note <text> — add a note to the diary
/report [days] — doctor report PDF + CSV (default 30 days)
/analyze — ask the AI model (Claude Code CLI or Ollama) for today's coaching now
/profiles — who this chat can see
/help — this list

Admin
/status — service health: last poll, Ollama, sync per profile
/mfa <code> — enter the Garmin verification code when asked

When a chat can see several people, name them first: /today Dad.
Wrist-sensor data and local-model suggestions, not medical advice.
```

### 🔐 MFA request

**Sent:** to admin chats when Garmin Connect asks for a verification code during login.
**Function:** `mfa_request(profile)`.

```
🔐 Garmin Connect needs a verification code (Dad)
Garmin has sent a one-time code by email or SMS. Reply here with:
/mfa <code>
The login waits about 10 minutes; after that the next poll asks again.
```

### 🧭 Coaching block

Appended to the morning brief, evening summary and weekly review (and the reply to `/analyze`)
when `features.daily_coaching` is on and the model answered.  **Function:** `coaching_block(advice)`.

The block always has the same shape so it is easy to skim: one-sentence summary, **Do more**,
**Do less**, **Watch out** (at most five bullets each), an optional heart note, and the "not
medical advice" line.  The model name in brackets tells you which backend produced it
(`llama3.1:8b`, or `rules` when the model was unavailable and the built-in fallback wrote it).

```
🧭 Coaching for today (llama3.1:8b)
A steady day with a short walk and a good night's sleep.
✅ Do more
• Take a second 15-minute walk after lunch
• Keep the 22:30 bedtime
⛔ Do less
• Long sitting stretches in the afternoon
👀 Watch out
• Resting heart rate is 4 bpm above the weekly average
❤️ Two at-rest heart-rate rises today; both are in the diary.
AI suggestions generated from watch data, not medical advice.
```

## Tuning what you receive

* Turn whole message types on or off per person with `features:` (`morning_brief`,
  `evening_summary`, `weekly_review`, `daily_coaching`, `palpitations`, `doctor_report`, `alerts`).
* The palpitation section of the evening summary and the episode alerts only exist when
  `features.palpitations: true` for that profile.
* Fewer / more episode alerts: raise or lower `palpitations.alert_min_confidence`, and adjust the
  thresholds (`rest_hr_threshold`, `rise_over_baseline`, `nocturnal_hr_threshold`, ...).
  Start conservative to avoid alarm fatigue; the doctor report still includes every candidate.
* `palpitations.quiet_hours: [23, 7]` holds non-critical alerts overnight.
* Alerts are deduplicated per rule and day, so a high resting HR is reported once, not every poll.
