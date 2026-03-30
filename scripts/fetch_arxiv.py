#!/usr/bin/env python3
"""
fetch_arxiv.py - Fetches and ranks arXiv papers per topic slot.
"""
import json, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE      = Path(__file__).parent.parent
CONFIG    = json.loads((BASE / "config.json").read_text())
VAULT     = json.loads((BASE / "memory_vault.json").read_text())
TOPICS    = (BASE / "weekly_topics.md").read_text().strip().splitlines()
ARXIV_API = "http://export.arxiv.org/api/query"
NS        = "{http://www.w3.org/2005/Atom}"

# Manually tuned short queries per slot
SLOT_QUERIES = {
    1: 'abs:"tactile" AND abs:"manipulation"',
    2: 'abs:"ultrasound" AND abs:"robot"',
    3: 'abs:"soft robot" OR abs:"soft end-effector"',
    4: 'abs:"SLAM" OR abs:"semantic SLAM" OR abs:"depth estimation"',
    5: 'abs:"agricultural robot" OR abs:"laser weeding" OR abs:"fruit picking"',
    6: 'abs:"imitation learning" OR abs:"ACT" OR abs:"embodied learning"',
    7: 'abs:"PINN" OR abs:"physics-informed" OR abs:"neural control"',
}


def get_tier1_ids():
    return set(VAULT.get("tier_1_seen_ids", []))


def get_tier2_lookup():
    return {e["arxiv_id"]: e for e in VAULT.get("tier_2_high_impact", [])}


def fetch_topic(slot_index, lookback_days=30, max_results=25):
    query     = SLOT_QUERIES[slot_index]
    date_from = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y%m%d")
    date_to   = datetime.now(timezone.utc).strftime("%Y%m%d")
    params = urllib.parse.urlencode({
        "search_query": f"({query}) AND submittedDate:[{date_from}0000 TO {date_to}2359]",
        "start":        0,
        "max_results":  max_results,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    })
    url = f"{ARXIV_API}?{params}"
    print(f"  Query: {query[:80]}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "arxiv-pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            xml_data = resp.read()
    except Exception as e:
        print(f"  [fetch] ERROR: {e}")
        return []
    finally:
        print("  Waiting 5s (arXiv rate limit)...")
        time.sleep(5)
    root   = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall(f"{NS}entry"):
        try:
            arxiv_id  = entry.find(f"{NS}id").text.strip().split("/abs/")[-1].split("v")[0]
            title     = entry.find(f"{NS}title").text.strip().replace("\n", " ")
            abstract  = entry.find(f"{NS}summary").text.strip().replace("\n", " ")
            published = entry.find(f"{NS}published").text.strip()[:10]
            authors   = [a.find(f"{NS}name").text for a in entry.findall(f"{NS}author")][:5]
            papers.append({
                "arxiv_id":  arxiv_id,
                "title":     title,
                "abstract":  abstract,
                "published": published,
                "authors":   authors,
                "url":       f"https://arxiv.org/abs/{arxiv_id}",
            })
        except Exception:
            continue
    return papers


def recency_score(published, lookback_days=30):
    try:
        days_old = (
            datetime.now(timezone.utc) -
            datetime.strptime(published, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        ).days
        return max(0.0, (lookback_days - days_old) / lookback_days)
    except Exception:
        return 0.0


def rank_paper(paper, t2, weights):
    recency   = recency_score(paper["published"]) * weights["recency"]
    velocity  = 0.0
    impact    = 0.0
    if t2:
        delta    = t2.get("citation_count", 0) - t2.get("citation_count_at_last_cover", 0)
        velocity = min(1.0, max(0.0, delta / 50)) * weights["citation_velocity"]
        impact   = t2.get("impact_score", 0.0) * weights["impact_score"]
    relevance = 0.5 * weights["relevance"]
    return round(recency + velocity + impact + relevance, 4)


def check_revisit(t2, all_batch):
    if not t2:
        return False
    cfg   = CONFIG["revisit_thresholds"]
    delta = t2.get("citation_count", 0) - t2.get("citation_count_at_last_cover", 0)
    if delta >= cfg["new_citations_delta"]:
        return True
    citing = sum(1 for p in all_batch if t2["arxiv_id"] in p.get("abstract", ""))
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


def run():
    tier1    = get_tier1_ids()
    tier2    = get_tier2_lookup()
    weights  = CONFIG["ranking_weights"]
    lookback = CONFIG.get("arxiv_lookback_days", 30)
    n_cands  = CONFIG.get("llm_candidates_per_slot", 5)
    results  = {}

    for i, topic_line in enumerate(TOPICS, start=1):
        print(f"\n[Slot {i}] {topic_line[:80]}")
        raw = fetch_topic(i, lookback_days=lookback)
        print(f"  Fetched: {len(raw)} papers")

        new_papers, revisit_papers = [], []
        for paper in raw:
            pid = paper["arxiv_id"]
            t2  = tier2.get(pid)
            if pid in tier1 and not t2:
                continue
            elif t2:
                if check_revisit(t2, raw):
                    revisit_papers.append({**paper, "_tier2": t2})
                    print(f"  REVISIT flagged: {pid} — {paper['title'][:50]}")
            else:
                new_papers.append(paper)

        print(f"  After dedup: {len(new_papers)} new, {len(revisit_papers)} revisit")

        ranked     = sorted(new_papers, key=lambda p: rank_paper(p, tier2.get(p["arxiv_id"]), weights), reverse=True)
        candidates = ranked[:n_cands]
        for j, p in enumerate(candidates):
            score = rank_paper(p, tier2.get(p["arxiv_id"]), weights)
            print(f"  [{j+1}] score={score} | {p['arxiv_id']} | {p['title'][:55]}")

        results[f"slot_{i}"] = {
            "topic_index":    i,
            "topic":          topic_line,
            "candidates":     candidates,
            "revisit_papers": revisit_papers,
        }

    return results


if __name__ == "__main__":
    print("=== arXiv Fetch & Filter ===")
    print(f"Lookback: {CONFIG.get('arxiv_lookback_days', 30)} days | Topics: {len(TOPICS)}")
    results = run()
    out = BASE / "scripts" / "_fetch_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")
    print("=== Done ===")
