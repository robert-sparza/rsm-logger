# RSM Soccer Coach: product notes

Working notes on what the analyzer should become for a team coach. Not a plan for this week.
Researched October 2026; competitor features come from their own pages and announcements and
weren't tested.

## 1. What similar tools do

| Product | What it is | What it gives the coach | Model |
|---|---|---|---|
| **Veo** (Cam 3 + Analytics 2) | Auto-follow camera + cloud | Distance, sprints, speed and time played per player; heat maps; shot maps that jump to the clip; multi-match leaderboards; "Player Spotlight" reels; "Coach Assist": questions about one or more matches in plain language | Camera + subscription; analytics is a paid add-on |
| **XbotGo** (Falcon, the camera you have) | Auto-tracking 4K gimbal for a phone | AI highlight reels; "Pro Soccer" (Sept 2026): cloud match reports with goals, shots, saves, blocks and defensive actions, plus trend charts | Camera, no subscription for the basics |
| **Trace** | Youth camera system | A personal highlight video for every player; player-focused analytics | Subscription; recruiting-oriented |
| **Pixellot** (+ VidSwap) | Installed cameras for clubs and federations | Full-game recording, automatic highlights, game breakdowns, team and player stats | Facility install |
| **Spiideo** | Fixed multi-camera install | Broadcast-grade recording for academies and colleges | High price, installed |
| **Hudl Sportscode / Replay** | Pro analysis software | Live tagging, custom code windows, drawing on video (Studio), multi-angle | Pro / youth packs |
| **Wyscout** | Scouting database | Video and stats of other teams and players | Pro scouting |
| **Metrica Sports** | Tagging + presentation | Tag clips by key phrases; presentation-ready graphics | Subscription |

**What they have in common:** automatic recording, **highlights**, **physical stats** (distance,
sprints), **heat maps**, and stats that **jump to the video clip**. The newest feature is
plain-language questions about the match (Veo Coach Assist).

**What none of them focus on:**
- **Training sessions.** All of them are built around matches. Drills with bibs, rondos and small
  games (what you're recording now) are an open space.
- **Local and private.** They upload video of minors to the cloud. We run on the club's own PC.
- **The coach's own tags.** Hudl has live tagging, but it's pro-priced. You already have a phone logger.
- **Spanish-first, low cost, using a camera the club already owns.**

## 2. Where our analyzer stands (Phase 1–2)

Have: player detection and tracking with ID clean-up, a team split by kit or bib, pitch calibration,
heat maps, team width and depth, distance and speed, ball path, possession, passes and turnovers,
a minimap video, a live RTMP view, and our own detector trained on the club's footage.

Weak: the ball in play (the detector still misses it often), identity (no names; the IDs restart every
video), a fixed camera is required, the calibration is clicked by hand, and the event list is
basic (passes and turnovers).

## 3. Tools and techniques to add (by value for effort)

1. **More labelled sessions.** The biggest lever. Two or three more batches across day and night,
   matches and drills, and different kits. Tools are ready (`PHASE2.md`).
2. **A better ball:**
   - inference on **tiles** of the 4K frame (SAHI-style), so the ball isn't shrunk;
   - a **multi-frame ball model** (TrackNet idea: heat maps over consecutive frames, built for small,
     fast balls; published for tennis, so it's a candidate, not proven for football);
   - a **ball path with physics**: fill short gaps along a smooth trajectory.
3. **Automatic calibration**, using a **pitch keypoint model** (Roboflow `sports` field keypoints, or
   SoccerNet calibration; BroadTrack shows camera and tripod models help). It removes the clicking, and
   lets us use **XbotGo auto-tracking footage** (re-calibrate every frame).
4. **Team split with appearance embeddings** (SigLIP → UMAP → KMeans, as in roboflow/sports forks)
   instead of colour rules. It works for any kit without tuning, and puts goalkeepers in their team.
5. **Player identity:**
   - jersey-number reading (SoccerNet challenge methods; a small number detector reported ~88% mAP
     where OCR failed), combined with re-identification;
   - a simple **roster screen**: the coach names each track once, and the names are carried across
     the session.
   - That gives per-player stats over a season.
6. **Events:** shots, saves, recoveries, duels, and set-piece starts. SoccerNet "action spotting" is
   the research reference. Start with rules on the ball path and the players.
7. **Full-pipeline reference:** SoccerNet **Game State Reconstruction** (`sn-gamestate`) does what
   we do (tracking, identity, team, minimap) as an open baseline with a metric (GS-HOTA). It's useful
   for comparing our approach and borrowing parts.

## 4. The deliverable for a coach

**The coach's question after a session is: what happened, who did what, and show me.** So the
product is a **report with clips**, not a CSV.

### After every match or training session (automatic, ~20–40 min after upload on the club PC)
1. **One-page summary:**
   - possession;
   - passes and turnovers by zone;
   - team width, depth and compactness over time;
   - intensity: distance, sprints, and the % of time at high speed.
2. **Per player:**
   - minutes played;
   - distance and sprints;
   - heat map;
   - touches;
   - passes made and received;
   - a short **personal clip reel**.
3. **Every number is clickable and plays the moment.** Our tracks already have time stamps.
4. **Training mode:**
   - per-drill blocks;
   - bib team vs non-bib team;
   - work rate per player;
   - "time in the right zone" for positional drills.
5. **Coach tags from the phone:** **RSM Match Logger** (already in this repo) records goals and
   "⭐ JUGADA" with times. Synced to the video, every tap becomes a clip with the stats around it.
   No competitor at this price does that.

### Over a season
- Player progress: distance, sprints and involvement per match.
- Team identity: average shape, where we lose the ball, where we win it.
- Exports: a PDF for parents and players, and a clip pack for the team meeting.

### Later
- **Ask the match:** "show every turnover in our half in the 2nd half", "who covered most ground".
  This is Veo's newest feature; doable with an LLM over our tracks and events tables.
- **Live view on the touchline** (we already have the RTMP live mode): a tablet minimap and a
  simple live possession count.

### How it's delivered
- **Engine:** stays on the club PC (GPU). Video never leaves the club unless the coach shares it.
- **App:** a local web dashboard (the PC serves it on the club Wi‑Fi; phones and tablets open it).
  Optional: share selected clips or reports by link.
- **Workflow:** record → copy the video into a folder → processing runs automatically → notification
  "report ready".

## 5. Privacy (players are minors)

- Written **parent/guardian consent** before recording, renewed each season; footage used only for
  coaching and review (the common rule in youth federations' guidelines).
- **Local processing** by default; share clips only with the team; option to **blur faces** of
  bystanders and opponents in anything exported.
- For Mexico, check the current federal personal-data law for private parties (the LFPDPPP; a new
  law was passed in 2025) with someone qualified. Data about children needs parental consent and
  extra care. This needs verifying; it isn't legal advice.

## 6. Name

**RSM Soccer Coach** works and fits next to **RSM Match Logger**. A family of names:
- **RSM Coach**: the product (short, works in Spanish and English);
  - **RSM Coach · Logger**: the phone app for live tags (today's `index.html`);
  - **RSM Coach · Engine**: the analyzer on the club PC;
  - **RSM Coach · Report**: what the coach and parents receive.

Alternatives if "Coach" feels generic: *RSM Match Lab*, *RSM Pitch IQ*, *RSM Táctica*.
Trademark and domain availability haven't been checked.

## Sources

- Veo Analytics 2 overview: https://support.veo.com/hc/en-us/articles/11873714080657-Veo-Analytics-2-Overview-and-features
- Veo heat maps: https://support.veo.com/hc/en-us/articles/11466495470865
- XbotGo Falcon review (PetaPixel): https://petapixel.com/2026/01/07/the-hands-free-599-xbotgo-falcon-captures-sports-like-a-tv-broadcaster
- XbotGo Pro Soccer launch: https://techbuzzireland.com/2026/09/06/xbotgo-launches-pro-soccer-globally-at-ifa2026/
- Camera systems compared (Veo article): https://www.veo.com/en-us/article/soccer-camera-systems-compared
- Veo alternatives: https://visu.network/alternatives/veo/
- Hudl Sportscode/Wyscout: https://www.hudl.com/blog/competition-database-wyscout-hudl-sportscode
- Metrica Sports: https://www.soccerscene.com.au/?p=21747
- SoccerNet Game State Reconstruction: https://arxiv.org/abs/2404.11335
- BroadTrack calibration: https://arxiv.org/html/2412.01721v1
- Football-PitchVision (roboflow/sports fork: SigLIP teams, pitch keypoints): https://github.com/DesusLove/Football-PitchVision
- Pitch keypoint model: https://huggingface.co/Simon9/football-field-detection-roboflow
- TrackNet: https://arxiv.org/abs/1907.03698
- Sports re-ID with jersey cues: https://arxiv.org/pdf/2206.02373
- SoccerNet challenge win (re-ID, numbers): https://www.unibw.de/vis/news/cvpr2023-soccernet
- Recording minors, guidelines: https://kilkennygaa.ie/resources/game-recording-guidelines/
- Sports data and minors (Leiden): https://www.universiteitleiden.nl/en/in-the-media/2025/06/does-collecting-sports-images-and-data-breach-privacy-laws
