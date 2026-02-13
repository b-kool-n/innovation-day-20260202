import json
import os
import re
from typing import Optional, Tuple

import requests
from bs4 import BeautifulSoup

from openai import OpenAI

BRAZE_HOME = "https://www.braze.com/docs/releases/home"
STATE_PATH = "state.json"

def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {"last_seen_id": ""}

    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                return {"last_seen_id": ""}
            return json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return {"last_seen_id": ""}

def save_state(state: dict) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        f.write("\n")


def fetch(url: str) -> str:
    r = requests.get(url, timeout=30, headers={"User-Agent": "braze-release-watcher/1.0"})
    r.raise_for_status()
    return r.text


def normalize_text(html_fragment: str) -> str:
    soup = BeautifulSoup(html_fragment, "html.parser")
    # Remove nav/irrelevant elements if present
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_latest_by_details_title(soup: BeautifulSoup) -> Optional[Tuple[str, str]]:
    """
    Returns (latest_id, extracted_text) if the page contains div.details_title entries.
    Assumes newest is first in DOM order; if not, you can sort by parsed date.
    """
    titles = soup.select("div.details_title[id]")
    if not titles:
        return None

    latest = titles[0]
    latest_id = latest.get("id", "").strip()
    if not latest_id:
        return None

    # Find the next <details> block after the title div
    details = latest.find_next("details")
    if not details:
        # fallback: just return section text after the title div
        extracted = normalize_text(str(latest.parent))
        return (latest_id, extracted)

    extracted = normalize_text(str(details))
    return (latest_id, extracted)


def extract_latest_by_release_heading(soup: BeautifulSoup) -> Optional[Tuple[str, str, str]]:
    """
    Fallback: Find the first heading that looks like 'January 8, 2026 release'
    and extract text until the next similar heading.
    Returns (synthetic_id, heading_text, extracted_text).
    """
    # Try common heading tags
    headings = soup.find_all(["h2", "h3"])
    target_idx = None
    for i, h in enumerate(headings):
        t = " ".join(h.get_text(" ", strip=True).split())
        if re.search(r"\brelease\b", t, flags=re.IGNORECASE) and re.search(r"\b20\d{2}\b", t):
            target_idx = i
            heading_text = t
            break

    if target_idx is None:
        return None

    # Create a stable synthetic ID from heading text
    synthetic_id = re.sub(r"[^a-z0-9]+", "-", heading_text.lower()).strip("-")

    start = headings[target_idx]
    end = None
    for j in range(target_idx + 1, len(headings)):
        t2 = " ".join(headings[j].get_text(" ", strip=True).split())
        if re.search(r"\brelease\b", t2, flags=re.IGNORECASE) and re.search(r"\b20\d{2}\b", t2):
            end = headings[j]
            break

    # Collect nodes from start until end
    chunks = []
    node = start
    while node is not None and node != end:
        chunks.append(str(node))
        node = node.find_next_sibling()
        if node is None:
            break

    extracted = normalize_text("\n".join(chunks))
    return (synthetic_id, heading_text, extracted)


def summarize_with_llm(text: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Keep the prompt deterministic and business-oriented.
prompt = (
    "You are writing a Slack message for SCHMACK (a marketing agency) summarising Braze monthly release notes.\n"
    "Return Slack markdown ONLY (no preamble), in EXACTLY this structure and order:\n"
    "\n"
    "*SCHMACK Need To Knows*\n"
    "• <1–4 bullets: ONLY items an agency would care about: major new features, major channel/canvas capabilities, new products, notable GA releases, and anything Early Access/Beta that could change what SCHMACK can offer clients>\n"
    "\n"
    "*Marketer Notes:*\n"
    "• <3–6 bullets: marketer-facing changes (campaigns/canvas, channels, targeting, reporting, deliverability tools, UX updates)>\n"
    "\n"
    "*Developer Notes:*\n"
    "• <3–6 bullets: developer-facing changes (SDKs, APIs, Currents, data schema changes, integration setup, breaking changes)>\n"
    "\n"
    "Rules:\n"
    "- Use the bullet character '•' at the start of every bullet line.\n"
    "- Keep each bullet short (ideally one line). Lead with the feature name, then the impact.\n"
    "- Tag maturity at the end: '(EA)', '(Beta)', or '(GA)' when applicable.\n"
    "- If something is a breaking change, prefix with 'BREAKING:' and put it in Developer Notes.\n"
    "- Avoid duplicate bullets across sections; if it's in Need To Knows, don't repeat it elsewhere unless needed.\n"
    "- Keep total under 1,600 characters.\n"
    "- Do not include the source URL.\n"
    "- Exclude minor UI tweaks/bug fixes; prioritise client-facing value and rollout planning.\n"
    "\n"
    f"Release notes:\n{text}"
)

    resp = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )

    # The SDK returns a structured response; output_text is the simplest accessor.
    return resp.output_text.strip()


def post_to_slack(title: str, summary: str, source_url: str) -> None:
    webhook = os.environ["SLACK_WEBHOOK_URL"]

    summary = summary.strip()
    if len(summary) > 3500:
        summary = summary[:3500] + "\n…(truncated)"

    text = f"*Braze Release Notes: {title}*\n\n{summary}\n\n<{source_url}|View full release notes>"

    payload = {
        "text": text,
    }

    r = requests.post(webhook, json=payload, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"Slack webhook error {r.status_code}: {r.text}")
    r.raise_for_status()


def main() -> None:
    state = load_state()
    last_seen = state.get("last_seen_id", "")

    html = fetch(BRAZE_HOME)
    soup = BeautifulSoup(html, "html.parser")

    latest = extract_latest_by_details_title(soup)
    if latest:
        latest_id, extracted_text = latest
        title = latest_id.replace("-", " ").title()
    else:
        fb = extract_latest_by_release_heading(soup)
        if not fb:
            raise RuntimeError("Could not detect latest release section on the page.")
        latest_id, title, extracted_text = fb

    if latest_id == last_seen:
        print(f"No new release detected (last_seen_id={last_seen}).")
        return

    # Summarize and alert
    # (Optionally truncate extracted_text to control token usage.)
    extracted_text = extracted_text[:20000]

    summary = summarize_with_llm(extracted_text)
    post_to_slack(title=title, summary=summary, source_url=BRAZE_HOME)

    # Update state
    state["last_seen_id"] = latest_id
    save_state(state)
    print(f"Alert posted. Updated last_seen_id to {latest_id}.")


if __name__ == "__main__":
    main()
