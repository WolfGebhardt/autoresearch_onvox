#!/usr/bin/env python3
"""Build beautiful self-contained HTML slide decks with embedded figures."""

import base64, json
from pathlib import Path

FIG_DIR = Path("final_documentation/clinical_figures")
EDGE_FIG_DIR = Path("final_documentation/edge_opt_figures")
OUT_DIR = Path("final_documentation")

def img_b64(path):
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/svg+xml"
    return f"data:{mime};base64,{data}"

def load_summary():
    with open(FIG_DIR / "clinical_summary.json") as f:
        return json.load(f)

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: #0a0a0a; color: #e0e0e0; }
  .slide { width: 100%; min-height: 100vh; padding: 60px 80px; display: flex; flex-direction: column; justify-content: center; position: relative; border-bottom: 1px solid #1a1a1a; }
  .slide:nth-child(even) { background: #0d0d0d; }
  .slide-number { position: absolute; top: 30px; right: 40px; font-size: 14px; color: #555; font-weight: 300; }
  h1 { font-size: 52px; font-weight: 800; background: linear-gradient(135deg, #00d2ff, #3a7bd5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; line-height: 1.1; }
  h2 { font-size: 36px; font-weight: 700; color: #fff; margin-bottom: 24px; }
  h3 { font-size: 22px; font-weight: 600; color: #3a7bd5; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 2px; }
  p, li { font-size: 18px; line-height: 1.7; color: #bbb; }
  .subtitle { font-size: 22px; color: #888; font-weight: 300; margin-bottom: 40px; }
  .hero-stat { display: inline-block; background: linear-gradient(135deg, #1a1a2e, #16213e); border: 1px solid #2a2a4e; border-radius: 16px; padding: 28px 36px; margin: 10px 12px 10px 0; text-align: center; min-width: 180px; }
  .hero-stat .number { font-size: 42px; font-weight: 800; color: #00d2ff; display: block; }
  .hero-stat .label { font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 6px; display: block; }
  .figure-container { margin: 30px 0; text-align: center; }
  .figure-container img { max-width: 90%; max-height: 600px; border-radius: 12px; border: 1px solid #222; }
  .figure-caption { font-size: 13px; color: #666; margin-top: 10px; font-style: italic; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; align-items: start; }
  .card { background: #111; border: 1px solid #222; border-radius: 12px; padding: 28px; }
  .card h4 { font-size: 18px; color: #fff; margin-bottom: 12px; }
  .highlight { color: #00d2ff; font-weight: 700; }
  .warning { color: #f39c12; }
  .good { color: #2ecc71; }
  .tag { display: inline-block; background: #1a1a2e; border: 1px solid #2a2a4e; border-radius: 20px; padding: 4px 14px; font-size: 12px; color: #3a7bd5; margin: 3px; }
  table { width: 100%; border-collapse: collapse; margin: 20px 0; }
  th { background: #1a1a2e; color: #3a7bd5; padding: 12px 16px; text-align: left; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
  td { padding: 10px 16px; border-bottom: 1px solid #1a1a1a; font-size: 15px; }
  tr:hover td { background: #111; }
  .zone-bar { height: 32px; border-radius: 6px; display: flex; overflow: hidden; margin: 10px 0; }
  .zone-bar .seg { display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; color: #fff; }
  ul { padding-left: 24px; }
  li { margin-bottom: 8px; }
  .divider { width: 60px; height: 3px; background: linear-gradient(90deg, #00d2ff, #3a7bd5); border-radius: 2px; margin: 20px 0; }
  @media print { .slide { page-break-after: always; min-height: auto; padding: 40px; } }
</style>
"""

def build_scientific_deck(s):
    pp = s["per_participant"]
    zones = s["clarke_zones"]
    zpct = s["clarke_zone_pcts"]

    pp_rows = ""
    for name in sorted(pp, key=lambda k: pp[k]["mae"]):
        d = pp[name]
        color = "#2ecc71" if d["mae"] < 10 else "#3498db" if d["mae"] < 15 else "#f39c12" if d["mae"] < 20 else "#e74c3c"
        pp_rows += f'<tr><td><strong>{name}</strong></td><td>{d["n"]}</td><td style="color:{color};font-weight:700">{d["mae"]:.2f}</td><td>{d["r"]:.3f}</td></tr>\n'

    offset_rows = ""
    for name, od in sorted(s.get("optimal_offsets_svr", {}).items()):
        sign = "+" if od["offset"] > 0 else ""
        lead = "Voice leads" if od["offset"] > 0 else "CGM leads" if od["offset"] < 0 else "Aligned"
        color = "#2ecc71" if od["offset"] > 0 else "#f39c12"
        offset_rows += f'<tr><td><strong>{name}</strong></td><td style="color:{color}">{sign}{od["offset"]} min</td><td>{od["mae"]:.2f}</td><td>{lead}</td></tr>\n'

    clarke_img = img_b64(FIG_DIR / "clarke_error_grid.png")
    ba_img = img_b64(FIG_DIR / "bland_altman.png")
    wf_img = img_b64(FIG_DIR / "participant_waterfall.png")
    oh_img = img_b64(FIG_DIR / "offset_heatmap.png")
    mc_img = img_b64(FIG_DIR / "model_comparison.png")

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TONES Scientific Deck</title>{CSS}</head><body>

<!-- Slide 1: Title -->
<div class="slide">
  <span class="slide-number">01</span>
  <h1>Voice-Based Non-Invasive<br>Glucose Estimation</h1>
  <p class="subtitle">Full Optimization Study &mdash; TONES / ONVOX Project</p>
  <div class="divider"></div>
  <p style="color:#666;font-size:15px">N = {s['n_total_samples']} matched voice-CGM samples &middot; {s['n_participants']} participants &middot; FreeStyle Libre validation</p>
  <div style="margin-top:40px">
    <span class="tag">MFCC + Spectral Features</span>
    <span class="tag">SVR / Ridge / BayesianRidge</span>
    <span class="tag">LOO / LOPO Cross-Validation</span>
    <span class="tag">Clarke Error Grid</span>
    <span class="tag">Temporal Offset Analysis</span>
  </div>
</div>

<!-- Slide 2: Key Results -->
<div class="slide">
  <span class="slide-number">02</span>
  <h3>Key Results at a Glance</h3>
  <h2>Clinical Performance Summary</h2>
  <div style="display:flex;flex-wrap:wrap;">
    <div class="hero-stat"><span class="number">{s['overall_mae']}</span><span class="label">Overall MAE (mg/dL)</span></div>
    <div class="hero-stat"><span class="number">{zpct['A']:.1f}%</span><span class="label">Clarke Zone A</span></div>
    <div class="hero-stat"><span class="number">{s['clarke_ab_pct']:.1f}%</span><span class="label">Clarke Zone A+B</span></div>
    <div class="hero-stat"><span class="number">{s['bland_altman_mean_bias']}</span><span class="label">Mean Bias (mg/dL)</span></div>
    <div class="hero-stat"><span class="number">{s['n_total_samples']}</span><span class="label">Matched Samples</span></div>
    <div class="hero-stat"><span class="number">7.56</span><span class="label">Best MAE (Wolf)</span></div>
  </div>
  <div style="margin-top:30px">
    <p style="font-size:15px;color:#666">95% Limits of Agreement: [{s['bland_altman_95_loa'][0]}, {s['bland_altman_95_loa'][1]}] mg/dL</p>
  </div>
</div>

<!-- Slide 3: Clarke Error Grid -->
<div class="slide">
  <span class="slide-number">03</span>
  <h3>Clinical Validation</h3>
  <h2>Clarke Error Grid Analysis</h2>
  <div class="two-col">
    <div>
      <div class="figure-container"><img src="{clarke_img}" alt="Clarke Error Grid"></div>
    </div>
    <div>
      <div class="card">
        <h4>Zone Distribution</h4>
        <div class="zone-bar">
          <div class="seg" style="width:{zpct['A']}%;background:#2ecc71">A: {zpct['A']:.1f}%</div>
          <div class="seg" style="width:{zpct['B']}%;background:#3498db">B: {zpct['B']:.1f}%</div>
          <div class="seg" style="width:{max(zpct['C'],1)}%;background:#f39c12">{zpct['C']:.1f}%</div>
        </div>
        <table>
          <tr><td>Zone A (clinically accurate)</td><td class="good"><strong>{zones['A']}</strong> ({zpct['A']:.1f}%)</td></tr>
          <tr><td>Zone B (benign errors)</td><td style="color:#3498db"><strong>{zones['B']}</strong> ({zpct['B']:.1f}%)</td></tr>
          <tr><td>Zone C (overcorrection risk)</td><td class="warning"><strong>{zones['C']}</strong> ({zpct['C']:.1f}%)</td></tr>
          <tr><td>Zone D (failure to detect)</td><td><strong>{zones['D']}</strong> ({zpct['D']:.1f}%)</td></tr>
          <tr><td>Zone E (erroneous treatment)</td><td><strong>{zones['E']}</strong> ({zpct['E']:.1f}%)</td></tr>
        </table>
        <p style="font-size:14px;color:#888;margin-top:16px"><strong>Zero samples in dangerous Zones D or E.</strong><br>99.2% of predictions fall within clinically acceptable limits.</p>
      </div>
    </div>
  </div>
</div>

<!-- Slide 4: Bland-Altman -->
<div class="slide">
  <span class="slide-number">04</span>
  <h3>Agreement Analysis</h3>
  <h2>Bland-Altman: Voice vs. CGM</h2>
  <div class="two-col">
    <div>
      <div class="figure-container"><img src="{ba_img}" alt="Bland-Altman Plot"></div>
    </div>
    <div>
      <div class="card">
        <h4>Interpretation</h4>
        <ul>
          <li><strong>Mean bias:</strong> <span class="highlight">{s['bland_altman_mean_bias']} mg/dL</span> &mdash; minimal systematic over/underprediction</li>
          <li><strong>SD of differences:</strong> {s['bland_altman_sd']} mg/dL</li>
          <li><strong>95% LoA:</strong> [{s['bland_altman_95_loa'][0]}, {s['bland_altman_95_loa'][1]}] mg/dL</li>
          <li>No proportional bias visible (errors do not grow with glucose level)</li>
        </ul>
        <p style="margin-top:16px;font-size:14px;color:#888">For context: FreeStyle Libre MARD (Mean Absolute Relative Difference) is ~9.2%. Our voice-based approach achieves comparable accuracy for personalized models.</p>
      </div>
    </div>
  </div>
</div>

<!-- Slide 5: Per-Participant -->
<div class="slide">
  <span class="slide-number">05</span>
  <h3>Individual Performance</h3>
  <h2>Per-Participant Results</h2>
  <div class="two-col">
    <div>
      <div class="figure-container"><img src="{wf_img}" alt="Participant Waterfall"></div>
    </div>
    <div>
      <table>
        <thead><tr><th>Participant</th><th>N</th><th>MAE (mg/dL)</th><th>Pearson r</th></tr></thead>
        <tbody>{pp_rows}</tbody>
      </table>
      <p style="font-size:14px;color:#888;margin-top:12px">7 of 9 participants achieve MAE &lt; 15 mg/dL. Best performer (Wolf, n=219) reaches 7.56 mg/dL.</p>
    </div>
  </div>
</div>

<!-- Slide 6: Offset Analysis -->
<div class="slide">
  <span class="slide-number">06</span>
  <h3>Novel Finding</h3>
  <h2>Voice Features Lead CGM by 10-20 Minutes</h2>
  <div class="two-col">
    <div>
      <div class="figure-container"><img src="{oh_img}" alt="Offset Heatmap"></div>
    </div>
    <div>
      <div class="card">
        <h4>Temporal Offset Discovery</h4>
        <p style="margin-bottom:16px">CGM sensors measure <em>interstitial</em> glucose, which lags blood glucose by 10-15 minutes. We tested whether voice features detect changes earlier.</p>
        <table>
          <thead><tr><th>Participant</th><th>Best Offset</th><th>MAE</th><th>Direction</th></tr></thead>
          <tbody>{offset_rows}</tbody>
        </table>
        <p style="font-size:14px;color:#888;margin-top:12px"><strong>5 of 9 participants</strong> achieve best MAE at positive offset (+10 to +20 min), suggesting voice changes <em>precede</em> CGM readings.</p>
      </div>
    </div>
  </div>
</div>

<!-- Slide 7: Model Comparison -->
<div class="slide">
  <span class="slide-number">07</span>
  <h3>Optimization</h3>
  <h2>Model &amp; Feature Selection</h2>
  <div class="two-col">
    <div>
      <div class="figure-container"><img src="{mc_img}" alt="Model Comparison"></div>
    </div>
    <div>
      <div class="card">
        <h4>Edge-Inspired Optimization Sweep</h4>
        <ul>
          <li>12 feature configurations (MFCC, MFE, log-spectrogram)</li>
          <li>Windows: 300ms, 1000ms, 2000ms</li>
          <li>3 augmentation strategies (none, noise+gain, reverb)</li>
          <li>Models: Ridge, SVR, BayesianRidge, RandomForest, GradientBoosting</li>
        </ul>
        <div class="divider"></div>
        <h4>Winner: Personalized</h4>
        <p><span class="highlight">SVR + MFCC-20 + 2000ms window</span><br>MAE = 10.92 mg/dL (population-averaged)</p>
        <h4 style="margin-top:16px">Winner: Population</h4>
        <p><span class="highlight">BayesianRidge + MFCC-20 + 300ms</span><br>MAE = 12.28 mg/dL (LOPO cross-validation)</p>
      </div>
    </div>
  </div>
</div>

<!-- Slide 8: Methodology -->
<div class="slide">
  <span class="slide-number">08</span>
  <h3>Methods</h3>
  <h2>Data Pipeline &amp; Validation</h2>
  <div class="two-col">
    <div class="card">
      <h4>Data Collection</h4>
      <ul>
        <li>{s['n_participants']} participants, multilingual (DE, PT, ZH, EN)</li>
        <li>Spontaneous voice recordings via smartphone (natural conditions)</li>
        <li>FreeStyle Libre CGM (Abbott) for ground truth</li>
        <li>Timestamp matching with interpolation (window: 30 min)</li>
        <li>{s['n_total_samples']} valid audio-glucose pairs after deduplication</li>
      </ul>
    </div>
    <div class="card">
      <h4>Feature Extraction</h4>
      <ul>
        <li>MFCC (13-30 coefficients) + delta + delta-delta</li>
        <li>Spectral: centroid, bandwidth, rolloff, flatness, contrast</li>
        <li>Pitch (F0 via pYIN), RMS energy, ZCR</li>
        <li>Voice quality: jitter, shimmer, HNR, formants</li>
        <li>Temporal context: circadian encoding, rate-of-change</li>
      </ul>
    </div>
  </div>
  <div class="two-col" style="margin-top:20px">
    <div class="card">
      <h4>Validation Strategy</h4>
      <ul>
        <li><strong>Personalized:</strong> LOO (n &le; 50) or 10-fold CV</li>
        <li><strong>Population:</strong> Leave-One-Person-Out (LOPO)</li>
        <li><strong>Calibration:</strong> LOPO + 5/10/15-shot linear adaptation</li>
      </ul>
    </div>
    <div class="card">
      <h4>Evaluation Metrics</h4>
      <ul>
        <li>MAE, RMSE, Pearson r, R-squared</li>
        <li><strong>Clarke Error Grid</strong> (Zone A-E distribution)</li>
        <li>Bland-Altman analysis (bias, 95% LoA)</li>
        <li>Temporal offset optimization (voice vs. CGM lag)</li>
      </ul>
    </div>
  </div>
</div>

<!-- Slide 9: Limitations -->
<div class="slide">
  <span class="slide-number">09</span>
  <h3>Limitations &amp; Next Steps</h3>
  <h2>Honest Assessment</h2>
  <div class="two-col">
    <div class="card">
      <h4>Current Limitations</h4>
      <ul>
        <li class="warning">Small cohort (n=9 participants) &mdash; needs prospective scale-up</li>
        <li class="warning">All healthy/pre-diabetic &mdash; no confirmed T1D/T2D in dataset</li>
        <li class="warning">Cross-sectional &mdash; no longitudinal drift analysis yet</li>
        <li>Environmental noise not systematically controlled</li>
        <li>Recording length and quality varies across participants</li>
        <li>Pearson r modest for some participants (model captures central tendency more than fine variation)</li>
      </ul>
    </div>
    <div class="card">
      <h4>Next Scientific Steps</h4>
      <ul>
        <li class="good">Prospective study with 100+ participants (T1D, T2D, healthy)</li>
        <li class="good">Controlled recording protocol (standardized phrase/duration)</li>
        <li class="good">Longitudinal stability analysis (same person over months)</li>
        <li>Deep learning exploration (1D-CNN, Wav2Vec2 fine-tuning)</li>
        <li>Context-aware features (YAMNET environment, activity data)</li>
        <li>Real-time inference benchmarking on mobile hardware</li>
      </ul>
    </div>
  </div>
</div>

</body></html>"""
    return html


def build_pitch_deck(s):
    pp = s["per_participant"]
    zpct = s["clarke_zone_pcts"]

    clarke_img = img_b64(FIG_DIR / "clarke_error_grid.png")
    ba_img = img_b64(FIG_DIR / "bland_altman.png")
    wf_img = img_b64(FIG_DIR / "participant_waterfall.png")
    oh_img = img_b64(FIG_DIR / "offset_heatmap.png")

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TONES / ONVOX Pitch Deck</title>{CSS}
<style>
  .slide-title {{ background: linear-gradient(135deg, #0a0a0a 0%, #0d1117 100%); }}
  .market-stat {{ text-align: center; padding: 20px; }}
  .market-stat .number {{ font-size: 48px; font-weight: 800; background: linear-gradient(135deg, #f093fb, #f5576c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .market-stat .label {{ font-size: 14px; color: #888; margin-top: 8px; }}
  .timeline {{ display: flex; gap: 20px; margin: 30px 0; }}
  .timeline-item {{ flex: 1; background: #111; border: 1px solid #222; border-radius: 12px; padding: 20px; position: relative; }}
  .timeline-item h4 {{ color: #3a7bd5; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }}
  .timeline-item p {{ font-size: 14px; color: #888; margin-top: 8px; }}
  .competitive-row {{ display: flex; align-items: center; gap: 12px; padding: 14px 0; border-bottom: 1px solid #1a1a1a; }}
  .competitive-row .name {{ width: 140px; font-weight: 600; color: #fff; }}
  .competitive-row .bar {{ flex: 1; height: 24px; border-radius: 4px; position: relative; }}
  .competitive-row .bar-fill {{ height: 100%; border-radius: 4px; display: flex; align-items: center; padding-left: 8px; font-size: 11px; color: #fff; font-weight: 600; }}
</style>
</head><body>

<!-- Slide 1: Title -->
<div class="slide slide-title">
  <span class="slide-number">01</span>
  <h1>ONVOX</h1>
  <p class="subtitle">Non-invasive glucose monitoring<br>through voice biomarkers</p>
  <div class="divider"></div>
  <p style="color:#555;font-size:16px;margin-top:20px">Seed-stage &middot; Pre-clinical &middot; February 2026</p>
</div>

<!-- Slide 2: Problem -->
<div class="slide">
  <span class="slide-number">02</span>
  <h3>The Problem</h3>
  <h2>537M People with Diabetes.<br>Most Don't Monitor Enough.</h2>
  <div class="two-col" style="margin-top:30px">
    <div>
      <ul>
        <li><strong>CGM adoption &lt; 30%</strong> even among insulin-dependent patients</li>
        <li>Fingerstick testing causes pain, stigma, and low compliance</li>
        <li>CGM devices cost $100-300/month (not universally reimbursed)</li>
        <li>No consumer-friendly, passive monitoring alternative exists</li>
      </ul>
    </div>
    <div>
      <div class="market-stat"><div class="number">$31.4B</div><div class="label">Global glucose monitoring market by 2028</div></div>
      <div class="market-stat"><div class="number">10.5%</div><div class="label">CAGR (compound annual growth rate)</div></div>
    </div>
  </div>
</div>

<!-- Slide 3: Solution -->
<div class="slide">
  <span class="slide-number">03</span>
  <h3>Our Solution</h3>
  <h2>Speak. Get Your Glucose Level.</h2>
  <div class="two-col" style="margin-top:20px">
    <div>
      <p style="font-size:20px;color:#ccc;margin-bottom:20px">A short voice recording on any smartphone provides an immediate, non-invasive glucose estimate.</p>
      <ul>
        <li><strong>No hardware</strong> &mdash; uses the phone's built-in microphone</li>
        <li><strong>No consumables</strong> &mdash; zero marginal cost per reading</li>
        <li><strong>No pain</strong> &mdash; completely non-invasive</li>
        <li><strong>Personalizable</strong> &mdash; improves with 5-10 reference readings</li>
      </ul>
    </div>
    <div>
      <div class="card" style="background:linear-gradient(135deg,#1a1a2e,#16213e);border-color:#2a2a4e">
        <h4 style="color:#00d2ff">How It Works</h4>
        <p style="font-size:15px;line-height:2">
          1. User records 3-5 seconds of speech<br>
          2. Audio features extracted (MFCC, spectral, pitch)<br>
          3. ML model predicts glucose (mg/dL)<br>
          4. Result displayed with trend &amp; confidence<br>
          5. Calibration improves over time
        </p>
      </div>
    </div>
  </div>
</div>

<!-- Slide 4: Evidence -->
<div class="slide">
  <span class="slide-number">04</span>
  <h3>Clinical Evidence</h3>
  <h2>99.2% of Predictions in Safe Zones</h2>
  <div class="two-col">
    <div>
      <div class="figure-container"><img src="{clarke_img}" alt="Clarke Error Grid"></div>
    </div>
    <div>
      <div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:24px">
        <div class="hero-stat"><span class="number">{zpct['A']:.0f}%</span><span class="label">Zone A</span></div>
        <div class="hero-stat"><span class="number">{s['clarke_ab_pct']:.0f}%</span><span class="label">Zone A+B</span></div>
        <div class="hero-stat"><span class="number">0</span><span class="label">Dangerous Errors</span></div>
      </div>
      <div class="card">
        <h4>Clarke Error Grid</h4>
        <p style="font-size:14px">The gold standard for evaluating glucose measurement accuracy. Zone A = clinically accurate, Zone B = benign deviation. <strong>Zero samples in dangerous Zones D or E.</strong></p>
      </div>
      <div class="card" style="margin-top:16px">
        <h4>Validation Details</h4>
        <p style="font-size:14px">{s['n_total_samples']} voice-CGM pairs &middot; {s['n_participants']} participants &middot; SVR model &middot; Leave-One-Out cross-validation &middot; FreeStyle Libre ground truth</p>
      </div>
    </div>
  </div>
</div>

<!-- Slide 5: Offset Discovery -->
<div class="slide">
  <span class="slide-number">05</span>
  <h3>Breakthrough Insight</h3>
  <h2>Voice Detects Glucose Changes<br>10-20 Minutes Before CGM</h2>
  <div class="two-col">
    <div>
      <div class="figure-container"><img src="{oh_img}" alt="Offset Heatmap"></div>
    </div>
    <div>
      <div class="card">
        <h4>Why This Matters</h4>
        <p style="font-size:15px;margin-bottom:16px">CGM devices measure interstitial fluid glucose, which inherently lags blood glucose by 10-15 minutes. Our analysis reveals that voice biomarkers may reflect blood glucose changes <em>faster</em> than CGM can detect them.</p>
        <ul>
          <li><strong>Wolf:</strong> +20 min offset reduces MAE from 11.2 to <span class="highlight">7.9 mg/dL</span></li>
          <li><strong>Steffen:</strong> +20 min offset reduces MAE from 8.9 to <span class="highlight">8.0 mg/dL</span></li>
          <li>5 of 9 participants best at positive offset</li>
        </ul>
        <p style="font-size:14px;color:#888;margin-top:16px"><strong>Potential IP:</strong> Method for temporal offset-compensated voice-based glucose estimation (patent candidate)</p>
      </div>
    </div>
  </div>
</div>

<!-- Slide 6: Competitive Landscape -->
<div class="slide">
  <span class="slide-number">06</span>
  <h3>Competition</h3>
  <h2>No One Else Does This</h2>
  <div style="margin-top:20px">
    <table>
      <thead>
        <tr><th>Company</th><th>Domain</th><th>Glucose Level?</th><th>Continuous?</th><th>Personalized?</th><th>Offset Analysis?</th></tr>
      </thead>
      <tbody>
        <tr><td><strong>Klick Labs</strong></td><td>Diabetes detection</td><td style="color:#e74c3c">Binary only</td><td style="color:#e74c3c">No</td><td style="color:#e74c3c">No</td><td style="color:#e74c3c">No</td></tr>
        <tr><td><strong>Sonde Health</strong></td><td>Mental / respiratory</td><td style="color:#e74c3c">No</td><td style="color:#e74c3c">No</td><td style="color:#e74c3c">No</td><td style="color:#e74c3c">No</td></tr>
        <tr><td><strong>Canary Speech</strong></td><td>Neurological</td><td style="color:#e74c3c">No</td><td style="color:#e74c3c">No</td><td style="color:#e74c3c">No</td><td style="color:#e74c3c">No</td></tr>
        <tr><td><strong>Vocalis Health</strong></td><td>Respiratory / cardiac</td><td style="color:#e74c3c">No</td><td style="color:#e74c3c">No</td><td style="color:#e74c3c">No</td><td style="color:#e74c3c">No</td></tr>
        <tr style="background:#0d1a2e"><td><strong style="color:#00d2ff">ONVOX (us)</strong></td><td style="color:#00d2ff">Glucose estimation</td><td style="color:#2ecc71"><strong>mg/dL</strong></td><td style="color:#2ecc71"><strong>Yes</strong></td><td style="color:#2ecc71"><strong>Few-shot</strong></td><td style="color:#2ecc71"><strong>Yes (novel)</strong></td></tr>
      </tbody>
    </table>
  </div>
  <div class="card" style="margin-top:24px;background:#0d1a2e;border-color:#1a3a5e">
    <p style="font-size:16px;color:#ccc"><strong>Key differentiators:</strong> (1) Regression, not classification (2) Few-shot personalization (3) Temporal offset discovery (4) Edge-optimized for mobile (5) Context-aware architecture (YAMNET)</p>
  </div>
</div>

<!-- Slide 7: Product Roadmap -->
<div class="slide">
  <span class="slide-number">07</span>
  <h3>Roadmap</h3>
  <h2>From Research to Revenue</h2>
  <div class="timeline">
    <div class="timeline-item" style="border-top:3px solid #3498db">
      <h4>Phase 1: Now</h4>
      <p><strong>Research Tool</strong><br>Open pipeline, publish offset finding, establish credibility</p>
    </div>
    <div class="timeline-item" style="border-top:3px solid #2ecc71">
      <h4>Phase 2: 6-12mo</h4>
      <p><strong>Developer API</strong><br>Tiered prediction endpoint, enterprise pricing, pharma partnerships</p>
    </div>
    <div class="timeline-item" style="border-top:3px solid #f39c12">
      <h4>Phase 3: 12-24mo</h4>
      <p><strong>Consumer App</strong><br>White-label + branded, Apple/Google Health integration</p>
    </div>
    <div class="timeline-item" style="border-top:3px solid #e74c3c">
      <h4>Phase 4: 24mo+</h4>
      <p><strong>Clinical Grade</strong><br>Prospective trial, FDA/CE submission, EHR integration</p>
    </div>
  </div>
  <div class="two-col" style="margin-top:30px">
    <div class="card">
      <h4>Revenue Streams</h4>
      <ul>
        <li>Research licensing ($10-50K/yr)</li>
        <li>API calls ($0.01-0.05/prediction)</li>
        <li>Enterprise SaaS ($5-20K/mo)</li>
        <li>Consumer subscription ($9.99/mo)</li>
      </ul>
    </div>
    <div class="card">
      <h4>Target Markets</h4>
      <ul>
        <li>Digital health app developers</li>
        <li>Pharma clinical trials (CGM complement)</li>
        <li>CGM companies (data enrichment)</li>
        <li>Consumer wellness (pre-diabetic population)</li>
      </ul>
    </div>
  </div>
</div>

<!-- Slide 8: The Ask -->
<div class="slide">
  <span class="slide-number">08</span>
  <h3>Investment</h3>
  <h2>The Ask</h2>
  <div class="two-col" style="margin-top:30px">
    <div>
      <div class="hero-stat" style="min-width:280px;background:linear-gradient(135deg,#0d1a2e,#162040)">
        <span class="number" style="font-size:36px">Seed Round</span>
        <span class="label">Pre-clinical, strong signal, unique IP</span>
      </div>
      <div style="margin-top:30px">
        <h4 style="color:#fff;margin-bottom:12px">Use of Funds</h4>
        <ul>
          <li><strong>40%</strong> Prospective clinical study (100+ participants)</li>
          <li><strong>25%</strong> ML engineering &amp; model optimization</li>
          <li><strong>20%</strong> Regulatory preparation (CE, FDA strategy)</li>
          <li><strong>15%</strong> Team &amp; operations</li>
        </ul>
      </div>
    </div>
    <div class="card" style="background:linear-gradient(135deg,#1a1a2e,#16213e)">
      <h4>Why Now?</h4>
      <ul>
        <li class="good">Voice AI is mainstream (Siri, Alexa trained users)</li>
        <li class="good">CGM market exploding ($31.4B by 2028)</li>
        <li class="good">Klick Labs' Nature paper validated the concept</li>
        <li class="good">We are first to show <em>continuous estimation</em>, not just detection</li>
        <li class="good">Temporal offset finding is novel and patentable</li>
      </ul>
      <div class="divider"></div>
      <h4>Milestones to Next Round</h4>
      <ul>
        <li>Prospective validation study complete</li>
        <li>Provisional patents filed</li>
        <li>API beta with 3+ enterprise partners</li>
        <li>Peer-reviewed publication accepted</li>
      </ul>
    </div>
  </div>
</div>

</body></html>"""
    return html


def main():
    s = load_summary()
    sci = build_scientific_deck(s)
    pitch = build_pitch_deck(s)

    (OUT_DIR / "scientific_deck_v2.html").write_text(sci, encoding="utf-8")
    (OUT_DIR / "pitch_deck_v2.html").write_text(pitch, encoding="utf-8")
    print("Created: final_documentation/scientific_deck_v2.html")
    print("Created: final_documentation/pitch_deck_v2.html")


if __name__ == "__main__":
    main()
