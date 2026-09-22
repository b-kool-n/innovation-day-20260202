import json
import os
import re
from typing import Optional

import requests
import anthropic

BRAZE_HOME = "https://www.braze.com/docs/releases/home"
STATE_PATH = "state.json"

# Braze embeds the full release notes as a JavaScript object literal:
#   var release_notes_feed = {"items":[ ... ]};
# The page then renders it client-side. We parse the raw object directly, so
# we never depend on JS-rendered DOM (which is what broke the old scraper).
FEED_MARKER = '{"items":['

# Map the feed's status field to the maturity tags the Slack summary uses.
STATUS_TAG = {
    "ga": "GA",
    "beta": "Beta",
    "early-access": "EA",
    "breaking": "BREAKING",
    "": "",
}


def load_state() -> dict:
    """
    Current state shape:
        {"seen": ["<sort_date>:<title-slug>", ...], "baselined": true}
    Older versions stored {"last_seen_id": "..."}; that is treated as
    un-baselined so the next run reseeds cleanly and posts nothing.
    """
    default = {"seen": [], "baselined": False}
    if not os.path.exists(STATE_PATH):
        return default

    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return default
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return default

    if "seen" in data:
        return {"seen": list(data.get("seen", [])), "baselined": bool(data.get("baselined", False))}

    # Legacy format: no per-item history to trust, so rebaseline.
    return default


def save_state(state: dict) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        f.write("\n")


def fetch(url: str) -> str:
    r = requests.get(url, timeout=30, headers={"User-Agent": "braze-release-watcher/1.0"})
    r.raise_for_status()
    return r.text


def _extract_object(text: str, start: int) -> Optional[str]:
    """Return the balanced {...} object beginning at `start`, honouring strings."""
    depth = 0
    in_str = False
    esc = False
    quote = ""
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == quote:
                in_str = False
            continue
        if c in ('"', "'"):
            in_str = True
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_feed(html: str) -> list:
    """Parse release_notes_feed items out of the raw page HTML."""
    idx = html.find(FEED_MARKER)
    if idx == -1:
        return []
    obj_str = _extract_object(html, idx)
    if not obj_str:
        return []
    try:
        data = json.loads(obj_str)
    except json.JSONDecodeError:
        return []
    items = data.get("items")
    return items if isinstance(items, list) else []


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


def item_key(item: dict) -> str:
    """Stable per-item identity. The feed has no explicit id, so we combine
    the release date with a slug of the title."""
    return f"{item.get('sort_date', '')}:{slug(item.get('title', ''))}"


def format_items_for_llm(items: list) -> str:
    """Render new items as clean, tagged text for the summariser."""
    lines = []
    for it in items:
        tag = STATUS_TAG.get((it.get("status") or "").lower(), "")
        header = it.get("title", "").strip()
        if it.get("category"):
            header = f"[{it['category']}] {header}"
        if tag:
            header = f"{header} ({tag})"
        body = (it.get("markdown") or it.get("description") or "").strip()
        lines.append(header)
        if body:
            lines.append(body)
        lines.append("")
    text = "\n".join(lines).strip()
    return text[:20000]


def summarize_with_llm(text: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = f"""You are writing a Slack message for SCHMACK (a marketing
agency) summarising Braze release notes.
Return Slack markdown ONLY (no preamble), in EXACTLY this structure and order:

*SCHMACK Need To Knows*
• <1–4 bullets: ONLY items an agency would care about: major new features,
major channel/canvas capabilities, new products, notable GA releases, and
anything Early Access/Beta that could change what SCHMACK can offer clients>

*Marketer Notes:*
• <3–6 bullets: marketer-facing changes (campaigns/canvas, channels,
targeting, reporting, deliverability tools, UX updates)>

*Developer Notes:*
• <3–6 bullets: developer-facing changes (SDKs, APIs, Currents, data schema
changes, integration setup, breaking changes)>

Rules:
- Use the bullet character '•' at the start of every bullet line.
- Keep each bullet short (ideally one line). Lead with the feature name, then
the impact.
- Tag maturity at the end: '(EA)', '(Beta)', or '(GA)' when applicable.
- If something is a breaking change, prefix with 'BREAKING:' and put it in
Developer Notes.
- Exclude minor UI tweaks/bug fixes; prioritise client-facing value and
rollout planning.
- Avoid duplicate bullets across sections.
- Keep total under 1,600 characters.
- Do not include the source URL.

Release notes:
{text}
"""

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )

    summary = next((b.text for b in response.content if b.type == "text"), "").strip()
    if not summary:
        raise RuntimeError("Claude response had no text content")
    return summary


def post_to_slack(title: str, summary: str, source_url: str) -> None:
    webhook = os.environ["SLACK_WEBHOOK_URL"]

    summary = summary.strip()
    if len(summary) > 3500:
        summary = summary[:3500] + "\n…(truncated)"

    # Normalise bullets if the model ever slips into "-" bullets.
    summary = summary.replace("\n- ", "\n• ")

    text = f"*Braze Release Notes: {title}*\n\n{summary}\n\n<{source_url}|View full release notes>"

    payload = {
        "username": "SCHMACK Braze Bot 1.0",
        "icon_emoji": ":robot_face:",
        "text": text,
    }

    r = requests.post(webhook, json=payload, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"Slack webhook error {r.status_code}: {r.text}")
    r.raise_for_status()


def main() -> None:
    state = load_state()
    seen = set(state.get("seen", []))

    html = fetch(BRAZE_HOME)
    items = parse_feed(html)
    if not items:
        raise RuntimeError("Could not detect the release notes feed on the page.")

    all_keys = [item_key(it) for it in items]

    # First run (or migration from the old state format): record everything
    # currently present and post nothing, so we don't flood Slack with the
    # entire back catalogue. Alerts start from the next genuinely new item.
    if not state.get("baselined"):
        save_state({"seen": all_keys, "baselined": True})
        print(f"Baseline established with {len(all_keys)} items. No alert posted.")
        return

    new_items = [it for it, k in zip(items, all_keys) if k not in seen]
    if not new_items:
        print("No new release items detected.")
        return

    # Title the post by the newest release date among the new items.
    newest_month = new_items[0].get("month", "Latest updates")
    title = f"{newest_month} — {len(new_items)} new update(s)"

    text = format_items_for_llm(new_items)
    summary = summarize_with_llm(text)
    post_to_slack(title=title, summary=summary, source_url=BRAZE_HOME)

    seen.update(all_keys)
    save_state({"seen": sorted(seen), "baselined": True})
    print(f"Alert posted for {len(new_items)} new item(s). State now tracks {len(seen)} items.")


if __name__ == "__main__":
    main()
