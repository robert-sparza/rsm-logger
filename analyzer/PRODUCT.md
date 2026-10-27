# RSM Soccer Coach: product notes

Working notes on what the analyzer should become for a team coach. Not a plan for this week.
Researched October 2026; competitor features come from their own pages and announcements and
weren't tested.

## 1. What similar tools do

| Product | What it is | What it gives the coach | Model |
|---|---|---|---|
| **Veo** (Cam 3 + Analytics 2) | Auto-follow camera + cloud | Distance, sprints, speed and time played per player; heat maps; shot maps that jump to the clip; multi-match leaderboards; "Player Spotlight" reels; "Coach Assist": questions about one or more matches in plain language | Camera + subscription; analytics is a paid add-on |
| **XbotGo** (Falcon, the camera you have) | Auto-tracking 4K camera with built-in lenses (stand-alone, not a phone gimbal) | AI highlight reels; "Pro Soccer" (Sept 2026): cloud match reports with goals, shots, saves, blocks and defensive actions, plus trend charts | Camera, no subscription for the basics |
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

## 7. Update, 25 Oct 2026: libraries, the coach summary, SAM 3

### Open-source pieces worth using
- **kloppy** (PySport) converts tracking data into one standard format. If `tracks.csv` is exported through it, every other library below works on our data without custom code.
- **floodlight** computes distance, velocity, acceleration, metabolic power and Voronoi space control from tracking data.
- **mplsoccer** draws the pitch plots (heat maps, pass maps, shape plots).
- **DataBallPy** loads and synchronises event and tracking data for one game.
- **Tryolabs' Norfair possession code** shows two ways to define possession: by passes and by time with the ball. Pick one and say which one the report uses.
- **FIFA's Enhanced Football Intelligence** has an open-source implementation (phases of play, line breaks, possession control). It is a thesis, so check the code before relying on it.
- **Pressing intensity from tracking** (Bekkers, 2025) uses player velocities and reaction times. It needs good speeds, so it comes after calibration is solid.
- I found **no mature open-source pass or event detection from tracking data**. We write those as rules on the ball path and the players, as planned in section 3.

### From numbers to a summary a coach can use
1. **Three layers.** (a) Three sentences: what happened. (b) The evidence: the numbers and a clip for each. (c) The detail: per player and per minute. The coach reads (a) and opens (b) only when they doubt it.
2. **Say which thresholds were used.** Youth research does not agree on one speed band. Fixed cutoffs of about 19.8 km/h (high-speed running) and 25.2 km/h (sprint) are common, but individual cutoffs as a share of each player's own top speed give different totals, and there is no agreed best model. My suggestion, still to confirm with a coach: show distance, **distance per minute**, and each player's **top speed in the session**, and mark the band used in every report. Fixed adult cutoffs may be too high for young players.
3. **Compare with the team's own history**, not with published benchmarks. The studies use different methods and cannot be compared directly.
4. **Training-game measures** (small-sided games) from the research: team **surface area**, **area per player**, **distance between team centres** (centroids), **stretch index** (average distance from the centre), and **length-to-width ratio**. They show compactness and width in one number each. Our summary already has team width and depth.
5. **Write the sentences from the numbers, not from a language model's memory.** A rule picks the finding ("distance per minute fell 18% in the second block"), a template writes the sentence, and each sentence links to its numbers and clip. A language model can reword it, never add a number. SciSports ships AI summaries with a note that they can contain errors, and a newsroom pipeline found wrong stat references to be the main failure. The same risk applies here.
6. **Be honest about accuracy.** Speed and distance from video are noisy. The report should print an error range, and flag a track as "uncertain" when the ID was stitched or the player was occluded.

### SAM 3
- **What it is:** a model that finds, masks and tracks every object matching a text phrase, in images and video. Roboflow offers it as **Auto Label**: you type a prompt per class ("person", "ball", "cone"), test the confidence on a few images, then run the batch.
- **Where it helps us:** pre-labelling new batches (night footage, multi-ball drills, new courts) faster than fixing our detector's guesses, and masks for a cleaner shirt-colour read in the team split.
- **Unknown:** how well it finds a small ball at night. I found nothing on small objects, and users report batches that returned 0 annotations. **Test it on 20 frames before running a batch.**
- **Not for live use:** its video speed is near real time for about 5 objects, and a game has 14 or more. Use it offline for labelling and as a check, not in the live view.
- Your own detector stays the one that runs the analysis. SAM 3 would only speed up and improve the labels it trains on.

### Next steps for this section
1. Export `tracks.csv` through kloppy so floodlight and mplsoccer can use it.
2. Add distance per minute, top speed and a visible speed band to `player_stats.csv`.
3. Add compactness numbers (area per player, stretch index) to `summary.json`.
4. Write the three-sentence summary generator from rules and templates.
5. Test SAM 3 Auto Label on the first moving XbotGo clip.

## 8. Coverage and cameras, 27 Oct 2026

**Rule of thumb (estimate, 4K video, 22 cm ball, about 8 px needed to find it):** the farthest point of the
field should be no more than about `53 m / tan(half the lens angle)` from the camera. 100° lens: 44 m. 90°: 53 m.
70°: 75 m. 56°: 100 m.

| Field | Players | Ball |
|---|---|---|
| up to ~30 m long (yours) | 1 camera, wide lens | 1 camera, wide lens, corner or 12-16 m back |
| ~55 x 36 m (7v7, 9v9) | 1 camera from a corner | 2 fixed cameras, one per half |
| ~70 x 45 m | 1 camera from a corner (players are ~40 px tall at the far corner) | 2 cameras, better 3 |
| 105 x 68 m | 2 cameras | 3 cameras, or 8K |

**Fixed cameras vs a follow camera.** A follow camera (XbotGo auto-tracking) only records the part it frames, so
players outside the frame are never seen and team measures (width, depth, heat maps, all positions) are
incomplete. It also needs a per-frame pitch registration (keypoint model; "No Bells, Just Whistles" and Roboflow's
keypoint tutorial are the starting points), and that fails when no lines are in view or the lines are faint (small
courts, night). It stays useful for highlights and ball events, not for full team analysis.

**Several fixed cameras:** calibrate each one separately, detect in each, convert to pitch metres, and merge
duplicates where the views overlap. Sync them by a clap or a flash at the start. This needs new code (not built).

**XbotGo Falcon:** the vendor says firmware v3.3.0 has a fixed full-field "Baseball Mode" with manual zoom and no
tracking. Test whether it gives a usable wide view for soccer. Vendor claim, not tested.

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
- SAM 3 in Hugging Face Transformers (video): https://huggingface.co/docs/transformers/model_doc/sam3_video
- SAM 3 paper: https://huggingface.co/papers/2511.16719
- Roboflow Auto Label with SAM 3: https://docs.roboflow.com/changelog/explore-by-month/november-2025/auto-label-data-with-sam-3
- Roboflow forum, SAM 3 auto label returning 0 annotations: https://discuss.roboflow.com/t/auto-label-with-sam-3-completed-with-0-annotations-but-consumed-credits/12302
- kloppy: https://kloppy.readthedocs.io
- floodlight: https://github.com/floodlight-sports/floodlight
- DataBallPy: https://science.ecosyste.ms/projects/808331
- Tryolabs possession with Norfair: https://tryolabs.com/blog/2022/10/17/measuring-soccer-ball-possession-ai-video-analytics
- FIFA EFI open-source implementation: https://capuana.ifi.uzh.ch/publications/PDFs/24098_An_Open_Source_Implementation_of_FIFA_s_Enhanced_Football_Intelligence.pdf
- Pressing intensity from tracking data (Bekkers): https://arxiv.org/pdf/2501.04712v2
- PPDA explained: https://thexgfootballclub.substack.com/p/quantifying-the-intensity-of-high-press-tactics-in-football-with-ppda-37d1ee09f768
- Youth speed thresholds, absolute vs individual: https://statsports.com/article/monitoring-high-speed-running-demands-in-youth-soccer-players-absolute-or-individualised-thresholds-a-case-study
- Speed zone models in elite youth soccer: https://gupea.ub.gu.se/items/6e865c27-2003-4b3c-8090-97860c746c19
- Narrative review of GPS load monitoring in male youth soccer: https://sponet.de/sponet/Record/4097972
- Small-sided game tactical metrics by age group: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7084353/
- SciSports AI match summaries: https://www.scisports.com/insight-in-an-instant-intelligence-the-moment-you-need-it
- RAG post-match tactical reports: https://ricardoheredia.substack.com/p/building-a-rag-system-that-generates
- Veo recording equipment guide: https://www.veo.com/en-us/article/football-recording-equipment-guide
- XbotGo Falcon baseball mode (fixed full-field view): https://xbotgo.com/blogs/knowledge/falcon-baseball-camera
- No Bells, Just Whistles (pitch registration): https://arxiv.org/abs/2404.08401
- Roboflow, camera calibration in sports with keypoints: https://blog.roboflow.com/camera-calibration-sports-computer-vision/
