#!/usr/bin/env python3
"""
researcher.py
Phase 1B of the Sunday Batch.
Makes one grounded Gemini call per topic slot to find recent papers,
deduplicates against the memory vault, checks revisit candidates,
and ranks results. Saves _fetch_results.json for the writer stage.

Search count: researcher_search_count (default 10) per slot
Candidate count passed to writer: llm_candidates_per_slot (default 5)

If a slot finds 0 valid papers, it is marked status=skipped with a
human-readable reason. The writer skips that slot. skipped_slots.log
records all skips for morning review.
"""
import json, time, re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from google import genai
from google.genai import types

BASE     = Path(__file__).parent.parent
CONFIG   = json.loads((BASE / "config.json").read_text())
VAULT    = json.loads((BASE / "memory_vault.json").read_text())
TOPICS   = (BASE / "weekly_topics.md").read_text().strip().splitlines()
SECRETS  = (BASE / ".secrets").read_text().strip()
CLIENT   = genai.Client(api_key=SECRETS)
MODEL    = CONFIG.get("gemini_model", "models/gemini-3.1-pro-preview")
LOOKBACK = CONFIG.get("arxiv_lookback_days", 30)
N_SEARCH = CONFIG.get("researcher_search_count", 10)
N_CANDS  = CONFIG.get("llm_candidates_per_slot", 5)
WEIGHTS  = CONFIG["ranking_weights"]
SKIP_LOG = BASE / "logs" / "skipped_slots.log"


# ── Vault helpers ──────────────────────────────────────────────────────────────

def get_tier1_ids():
    return set(VAULT.get("tier_1_seen_ids", []))

def get_tier2_lookup():
    return {e["paper_id"]: e for e in VAULT.get("tier_2_high_impact", [])}


# ── Skip logging ───────────────────────────────────────────────────────────────

def log_skip(slot_index, topic_line, reason):
    """Append a human-readable skip record to skipped_slots.log."""
    SKIP_LOG.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    entry = (
        f"[{today}] Slot {slot_index} SKIPPED\n"
        f"  Topic:  {topic_line.strip()}\n"
        f"  Reason: {reason}\n"
        f"  Action: Review and broaden topic in weekly_topics.md before next batch.\n"
        f"{'─'*60}\n"
    )
    with open(SKIP_LOG, "a") as f:
        f.write(entry)
    print(f"  Logged to {SKIP_LOG}")


def classify_error(exception, raw_response=None):
    """Return a human-readable reason string for a slot skip."""
    err = str(exception).lower()
    if "500" in err or "internal" in err:
        return (
            "Gemini internal error during grounded search. This often means the topic "
            "is too niche or specialised for the search grounding to find sufficient "
            "recent papers. Consider broadening the topic description."
        )
    if "429" in err or "quota" in err or "rate" in err:
        return (
            "API rate limit hit. The slot was skipped due to quota exhaustion. "
            "This is transient — no topic change needed, it will retry next Sunday."
        )
    if "timeout" in err or "timed out" in err:
        return (
            "Request timed out. Gemini search took too long, possibly due to server load. "
            "No topic change needed — retry next Sunday."
        )
    if "json" in err or "parse" in err:
        return (
            "Gemini returned malformed JSON after all retries. The topic may be too broad, "
            "causing an overlong response, or too niche to find enough papers to fill the "
            "required structure. Consider narrowing or rewording the topic."
        )
    if raw_response and len(raw_response.strip()) < 50:
        return (
            "Gemini returned an empty or near-empty response. Topic is likely too niche "
            "or uses terminology not well-indexed in recent literature. "
            "Try alternative keywords or broaden the scope."
        )
    return (
        f"Unexpected error: {str(exception)[:120]}. "
        "Review the topic and consider broadening if this recurs."
    )


# ── Grounded search ────────────────────────────────────────────────────────────

RESEARCHER_PROMPT = """You are a research assistant with web search access.
Search the web and find {n} recent academic papers on the following topic published in the last {days} days.

TOPIC: {topic}

Search across arXiv, Nature, IEEE, Science, ACM, bioRxiv, and major journals.
Prioritize the most recently published papers first.
Only include papers you can verify actually exist with a real URL.

Return ONLY a valid JSON array. No markdown, no explanation, no code fences.
Each object must have exactly these fields:
[
  {{
    "title": "exact full paper title",
    "authors": ["First Last", "First Last"],
    "published": "YYYY-MM-DD",
    "url": "direct URL or DOI (must start with https://)",
    "source": "journal or conference name",
    "summary": "2 sentence summary of key contribution"
  }}
]

If fewer than {n} papers exist for this topic in the timeframe, return however many you find.
Never invent or hallucinate papers."""


def search_topic(slot_index, topic_line, retries=2):
    topic_text = topic_line.split(".", 1)[-1].strip()
    prompt = RESEARCHER_PROMPT.format(n=N_SEARCH, days=LOOKBACK, topic=topic_text)
    print(f"  Searching ({N_SEARCH} papers): {topic_text[:65]}")

    last_error    = None
    last_raw      = ""

    for attempt in range(1, retries + 2):
        try:
            response = CLIENT.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                    max_output_tokens=8192,
                )
            )
            last_raw = response.text.strip()

            # Strip markdown fences
            raw = re.sub(r'^```(?:json)?\s*', '', last_raw, flags=re.MULTILINE)
            raw = re.sub(r'\s*```\s*$', '', raw, flags=re.MULTILINE)
            raw = raw.strip()

            # Find JSON array boundaries robustly
            start = raw.find('[')
            end   = raw.rfind(']')
            if start == -1 or end == -1:
                raise ValueError("No JSON array found in response")
            raw = raw[start:end+1]

            papers = json.loads(raw)

            # Validate fields and URLs
            valid = []
            for p in papers:
                if all(k in p for k in ("title", "authors", "published", "url", "source", "summary")):
                    if p["url"].startswith("http"):
                        valid.append(p)
                    else:
                        print(f"  SKIP (bad URL): {p['title'][:50]}")
                else:
                    print(f"  SKIP (missing fields): {p.get('title', '?')[:50]}")

            print(f"  Found: {len(valid)} valid papers")
            return valid, None  # papers, no error

        except json.JSONDecodeError as e:
            last_error = e
            print(f"  [attempt {attempt}] JSON parse error: {e}")
            if attempt <= retries:
                print(f"  Retrying in 10s...")
                time.sleep(10)

        except Exception as e:
            last_error = e
            print(f"  [attempt {attempt}] ERROR: {e}")
            if attempt <= retries:
                wait = 15 * attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)

    # All retries exhausted
    reason = classify_error(last_error, last_raw)
    return [], reason


# ── Dedup & revisit ───────────────────────────────────────────────────────────

def normalize_id(paper):
    url = paper.get("url", "").strip().rstrip("/").lower()
    doi_match = re.search(r'10\.\d{4,}/\S+', url)
    if doi_match:
        return doi_match.group(0).rstrip(".,)")
    url = re.sub(r'\?.*$', '', url)
    return url

def check_revisit(t2, all_papers):
    cfg   = CONFIG["revisit_thresholds"]
    delta = t2.get("citation_count", 0) - t2.get("citation_count_at_last_cover", 0)
    if delta >= cfg["new_citations_delta"]:
        return True
    pid    = t2.get("paper_id", "")
    citing = sum(1 for p in all_papers if pid in p.get("url", "") or pid in p.get("summary", ""))
    if citing >= cfg["citing_papers_in_batch"]:
        return True
    last = t2.get("last_covered_date", t2.get("first_seen_batch", ""))
    if last:
        days_since = (
            datetime.now(timezone.utc) -
            datetime.strptime(last, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        ).days
        if days_since < CONFIG.get("revisit_frequency_cap_days", 28):
            return False
    return False


# ── Ranking ───────────────────────────────────────────────────────────────────

def recency_score(published_str):
    try:
        days_old = (
            datetime.now(timezone.utc) -
            datetime.strptime(published_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        ).days
        return max(0.0, (LOOKBACK - days_old) / LOOKBACK)
    except Exception:
        return 0.5

def rank_paper(paper, t2):
    recency  = recency_score(paper.get("published", "")) * WEIGHTS["recency"]
    velocity = 0.0
    impact   = 0.0
    if t2:
        delta    = t2.get("citation_count", 0) - t2.get("citation_count_at_last_cover", 0)
        velocity = min(1.0, max(0.0, delta / 50)) * WEIGHTS["citation_velocity"]
        impact   = t2.get("impact_score", 0.0) * WEIGHTS["impact_score"]
    relevance = 0.6 * WEIGHTS["relevance"]
    return round(recency + velocity + impact + relevance, 4)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    tier1    = get_tier1_ids()
    tier2    = get_tier2_lookup()
    results  = {}
    skipped  = []

    for i, topic_line in enumerate(TOPICS, start=1):
        print(f"\n{'='*60}")
        print(f"[Slot {i}] {topic_line[:75]}")

        raw_papers, error_reason = search_topic(i, topic_line)

        # Delay between slots
        if i < len(TOPICS):
            print("  Waiting 5s before next slot...")
            time.sleep(5)

        # ── Skip logic ──
        if not raw_papers:
            reason = error_reason or (
                "No papers found matching this topic in the search window. "
                "The topic may be too niche or use terminology not well-indexed "
                "in recent literature. Consider broadening the keywords."
            )
            print(f"  SKIPPED: {reason[:100]}")
            log_skip(i, topic_line, reason)
            skipped.append(i)
            results[f"slot_{i}"] = {
                "topic_index":    i,
                "topic":          topic_line,
                "status":         "skipped",
                "skip_reason":    reason,
                "candidates":     [],
                "revisit_papers": [],
            }
            continue

        # ── Dedup & revisit check ──
        new_papers, revisit_papers = [], []
        for paper in raw_papers:
            pid = normalize_id(paper)
            paper["paper_id"] = pid
            t2  = tier2.get(pid)
            if pid in tier1 and not t2:
                continue
            elif t2:
                if check_revisit(t2, raw_papers):
                    revisit_papers.append({**paper, "_tier2": t2})
                    print(f"  REVISIT: {paper['title'][:55]}")
            else:
                new_papers.append(paper)

        # ── Skip if dedup leaves nothing ──
        if not new_papers and not revisit_papers:
            reason = (
                "All papers found this week were already covered in previous briefings "
                "(present in Tier 1 seen IDs). No new content available for this topic. "
                "The topic may need broadening, or coverage will resume when new papers appear."
            )
            print(f"  SKIPPED (all seen): {reason[:80]}")
            log_skip(i, topic_line, reason)
            skipped.append(i)
            results[f"slot_{i}"] = {
                "topic_index":    i,
                "topic":          topic_line,
                "status":         "skipped",
                "skip_reason":    reason,
                "candidates":     [],
                "revisit_papers": [],
            }
            continue

        print(f"  After dedup: {len(new_papers)} new, {len(revisit_papers)} revisit")

        # ── Rank and select top candidates ──
        ranked     = sorted(new_papers, key=lambda p: rank_paper(p, tier2.get(p["paper_id"])), reverse=True)
        candidates = ranked[:N_CANDS]

        for j, p in enumerate(candidates):
            score = rank_paper(p, tier2.get(p["paper_id"]))
            print(f"  [{j+1}] score={score} | {p['published']} | {p['title'][:50]}")

        results[f"slot_{i}"] = {
            "topic_index":    i,
            "topic":          topic_line,
            "status":         "ok",
            "candidates":     candidates,
            "revisit_papers": revisit_papers,
        }

    return results, skipped


if __name__ == "__main__":
    print("=== Researcher Phase ===")
    print(f"Model: {MODEL} | Lookback: {LOOKBACK}d | Search: {N_SEARCH} | Pass to writer: {N_CANDS}")
    start = datetime.now()

    results, skipped = run()
    elapsed = (datetime.now() - start).seconds

    out = BASE / "scripts" / "_fetch_results.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    ok_count = sum(1 for v in results.values() if v.get("status") == "ok")
    print(f"Slots OK:      {ok_count}/7")
    if skipped:
        print(f"Slots skipped: {len(skipped)} — slots {skipped}")
        print(f"Skip log:      {SKIP_LOG}")
        print(f"Action:        Review skipped topics in weekly_topics.md")
    print(f"Saved to:      {out}")
    print(f"Completed in:  {elapsed}s")
    print("=== Done ===")
