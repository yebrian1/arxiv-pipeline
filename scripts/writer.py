#!/usr/bin/env python3
"""
writer.py
Phase 1C of the Sunday Batch.
Reads _fetch_results.json produced by researcher.py, then makes one
Gemini call (no grounding needed) to generate all 7 daily briefings
as a structured JSON array saved to queue.json.

Article structure per slot:
  - Paper 1 (Lead):       full treatment
  - Paper 2 (Supporting): 2-3 paragraphs
  - Paper 3 (Horizon):    1 paragraph, only if score >= horizon_paper_score_gate
  - Revisit article:      short format, delta only, links original briefing

Skipped slots (status=skipped in fetch results) are omitted entirely.
"""
import json, re
from datetime import datetime, timezone
from pathlib import Path
from google import genai
from google.genai import types

BASE        = Path(__file__).parent.parent
CONFIG      = json.loads((BASE / "config.json").read_text())
VAULT       = json.loads((BASE / "memory_vault.json").read_text())
TOPICS_RAW  = (BASE / "weekly_topics.md").read_text().strip()
FETCH       = json.loads((BASE / "scripts" / "_fetch_results.json").read_text())
SECRETS     = (BASE / ".secrets").read_text().strip()
CLIENT      = genai.Client(api_key=SECRETS)
MODEL       = CONFIG.get("gemini_model", "models/gemini-3.1-pro-preview")
HORIZON_GATE = CONFIG.get("horizon_paper_score_gate", 0.50)
GITHUB_BASE  = CONFIG.get("github_pages_base", "https://yourusername.github.io/arxiv-pipeline")


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_tier2_ledger():
    return VAULT.get("tier_2_high_impact", [])

def today_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def briefing_url(date_str, slot_index):
    """Deterministic GitHub Pages URL for a briefing."""
    return f"{GITHUB_BASE}/{date_str}-slot{slot_index}-briefing"


# ── Prompt building ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert science communicator writing a daily research briefing newsletter.
Your readers are technically sophisticated researchers and engineers in robotics, AI, and related fields.
Your writing is clear, precise, and insightful — not hype, not oversimplified.
You connect ideas across papers and highlight what truly matters for the field.

WRITING RULES:
- Use markdown formatting throughout
- Each briefing starts with a ## heading (the briefing title)
- Bold key technical terms on first use
- Include all paper URLs as markdown hyperlinks
- Never fabricate details not present in the provided summaries
- Be specific about methods, results, and numbers where available
- Maintain a consistent voice across all briefings"""


def build_writer_prompt(active_slots, tier2_ledger, batch_date):
    """Build the full writer prompt with all slot data."""

    # Format Tier 2 ledger for dual-citation context
    if tier2_ledger:
        ledger_text = "HIGH-IMPACT PAPER LEDGER (for dual-citation — link these when relevant):\n"
        for entry in tier2_ledger[:20]:  # cap at 20 to manage token count
            ledger_text += (
                f"  - [{entry['title']}]({entry.get('url', '#')}) | "
                f"impact={entry.get('impact_score', 0):.2f} | "
                f"prev_article={entry.get('local_archive_url', 'N/A')}\n"
                f"    Summary: {entry.get('summary', '')}\n"
            )
    else:
        ledger_text = "HIGH-IMPACT PAPER LEDGER: Empty (first batch).\n"

    # Format each active slot
    slots_text = ""
    for slot in active_slots:
        i           = slot["topic_index"]
        topic       = slot["topic"]
        is_revisit  = bool(slot.get("revisit_papers"))
        candidates  = slot["candidates"]
        revisits    = slot.get("revisit_papers", [])

        slots_text += f"\n{'='*60}\n"
        slots_text += f"SLOT {i} | day_index={i}\n"
        slots_text += f"TOPIC: {topic}\n"

        if is_revisit and revisits:
            slots_text += f"STATUS: REVISIT ARTICLE\n"
            r = revisits[0]
            t2 = r.get("_tier2", {})
            slots_text += (
                f"  Original paper: [{r['title']}]({r['url']})\n"
                f"  First covered:  {t2.get('local_archive_url', 'N/A')}\n"
                f"  Citation delta: +{t2.get('citation_count',0) - t2.get('citation_count_at_last_cover',0)} since last cover\n"
                f"  Summary: {r['summary']}\n"
            )
            if candidates:
                slots_text += "  NEW CITING PAPERS:\n"
                for c in candidates[:3]:
                    slots_text += f"    - [{c['title']}]({c['url']}) | {c['published']} | {c['summary']}\n"
        else:
            slots_text += f"CANDIDATES (ranked, pick Lead + Supporting + optional Horizon):\n"
            for j, c in enumerate(candidates):
                role_hint = "Lead" if j == 0 else ("Supporting" if j == 1 else f"Horizon (only if score justifies)")
                slots_text += (
                    f"  [{j+1}] {role_hint} | score={c.get('_rank_score', '?')} | {c['published']}\n"
                    f"      Title:   {c['title']}\n"
                    f"      Authors: {', '.join(c.get('authors', [])[:3])}\n"
                    f"      Source:  {c.get('source', 'unknown')}\n"
                    f"      URL:     {c['url']}\n"
                    f"      Summary: {c['summary']}\n"
                )

        # Pre-compute briefing URL for this slot
        slots_text += f"BRIEFING_URL (for local_archive_url): {briefing_url(batch_date, i)}\n"

    horizon_gate_note = f"Only include a Horizon paper if it is genuinely noteworthy. Score gate: {HORIZON_GATE}."

    prompt = f"""{ledger_text}

BATCH DATE: {batch_date}
HORIZON GATE NOTE: {horizon_gate_note}

DUAL-CITATION INSTRUCTION:
If any candidate paper builds upon, cites, or is directly related to a paper in the HIGH-IMPACT PAPER LEDGER,
you MUST mention the connection explicitly in the article body and embed markdown links to BOTH:
  1. The original paper URL
  2. The prev_article URL from the ledger
Also set cites_high_impact_id to that paper's paper_id in your JSON output.

ARTICLE FORMAT PER SLOT:
- REVISIT slots: short format (~400 words). Open with link to original briefing. Cover only the delta.
  End with: "**Impact update:** [brief note on why this paper's significance has grown]"
- NORMAL slots: 
  Lead paper: ~400 words. Full treatment of methods, results, significance.
  Supporting paper: ~200 words. How it complements or contrasts the lead.
  Horizon paper: ~100 words. One paragraph. "Worth watching because..."
  End each briefing with a "**Key takeaway:**" one-liner.

{slots_text}

OUTPUT INSTRUCTIONS:
Return ONLY a valid JSON array of {len(active_slots)} objects. No markdown fences, no preamble.
Each object must have exactly these fields:
{{
  "day_index": <integer 1-7>,
  "topic_slot": "<topic name, short>",
  "title": "<compelling briefing headline>",
  "is_revisit": <true|false>,
  "markdown_content": "<full briefing in markdown, all on one line with \\n for newlines>",
  "papers": [
    {{
      "role": "lead|supporting|horizon|revisit",
      "title": "<paper title>",
      "url": "<paper url>",
      "is_high_impact": <true|false>,
      "cites_high_impact_id": "<paper_id or null>"
    }}
  ]
}}"""

    return prompt


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    batch_date   = today_str()
    tier2_ledger = get_tier2_ledger()

    # Filter to active (non-skipped) slots
    active_slots = []
    skipped_slots = []
    for key in sorted(FETCH.keys()):
        slot = FETCH[key]
        if slot.get("status") == "skipped":
            skipped_slots.append(slot["topic_index"])
            print(f"  Slot {slot['topic_index']} SKIPPED — {slot.get('skip_reason','?')[:80]}")
        elif slot.get("candidates") or slot.get("revisit_papers"):
            active_slots.append(slot)
        else:
            skipped_slots.append(slot.get("topic_index", "?"))
            print(f"  Slot {slot.get('topic_index','?')} has no candidates — skipping")

    print(f"Active slots: {len(active_slots)} | Skipped: {len(skipped_slots)}")

    if not active_slots:
        print("ERROR: No active slots to write. Check researcher output.")
        return None

    # Build prompt
    prompt = build_writer_prompt(active_slots, tier2_ledger, batch_date)

    print(f"\nCalling writer ({MODEL})...")
    print(f"Prompt size: ~{len(prompt)//4} tokens (estimated)")

    try:
        response = CLIENT.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,       # some creativity for writing
                max_output_tokens=16384,
            )
        )
        raw = response.text.strip()

        # Strip fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```\s*$', '', raw, flags=re.MULTILINE)
        raw = raw.strip()

        # Find array boundaries
        start = raw.find('[')
        end   = raw.rfind(']')
        if start == -1 or end == -1:
            raise ValueError("No JSON array found in writer response")
        raw = raw[start:end+1]

        briefings = json.loads(raw)
        print(f"Received {len(briefings)} briefings from writer")

        # Validate structure
        required = {"day_index", "topic_slot", "title", "is_revisit", "markdown_content", "papers"}
        valid_briefings = []
        for b in briefings:
            missing = required - set(b.keys())
            if missing:
                print(f"  WARNING: Briefing day={b.get('day_index','?')} missing fields: {missing}")
            else:
                valid_briefings.append(b)

        print(f"Valid briefings: {len(valid_briefings)}")
        return valid_briefings, skipped_slots

    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parse failed: {e}")
        print(f"Raw response (first 500 chars): {raw[:500]}")
        return None, skipped_slots
    except Exception as e:
        print(f"ERROR: Writer call failed: {e}")
        return None, skipped_slots


if __name__ == "__main__":
    print("=== Writer Phase ===")
    print(f"Model: {MODEL} | Batch date: {today_str()}")
    print(f"Loading fetch results from: {BASE / 'scripts' / '_fetch_results.json'}")

    result = run()
    if result is None:
        print("Writer failed — check errors above")
        exit(1)

    briefings, skipped = result
    if briefings is None:
        print("Writer failed — check errors above")
        exit(1)

    # Save to queue.json
    queue_path = BASE / "queue.json"
    queue_path.write_text(json.dumps(briefings, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"Saved {len(briefings)} briefings to {queue_path}")
    if skipped:
        print(f"Skipped slots: {skipped} — no articles generated for these days")

    # Print preview of each briefing title
    print("\nBriefing titles:")
    for b in briefings:
        revisit_tag = " [REVISIT]" if b.get("is_revisit") else ""
        print(f"  Day {b['day_index']}: {b['title']}{revisit_tag}")

    print("\n=== Done ===")
