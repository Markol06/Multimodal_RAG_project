import aiohttp
import asyncio
import json
import os
import re

from bs4 import BeautifulSoup
from urllib.parse import urljoin

from config import BASE_URL, START_URL

class BatchScraper:
    def __init__(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.data_dir = os.path.join(root_dir, "data", "raw")
        self.images_dir = os.path.join(self.data_dir, "images")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        self.news_path = os.path.join(self.data_dir, "News.json")
        self.all_articles = []

    async def fetch(self, session: aiohttp.ClientSession, url: str) -> str:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.text()

    async def get_last_issues(self, session: aiohttp.ClientSession, count: int = 10) -> list:
        print(f"[INFO] Fetching issues from: {START_URL}")
        html = await self.fetch(session, START_URL)
        soup = BeautifulSoup(html, "html.parser")

        issues = []
        for a in soup.select('a[href^="/the-batch/issue-"]'):
            href = a.get("href")
            if href and href.startswith("/the-batch/issue-"):
                try:
                    num = int(href.rstrip("/").split("-")[-1])
                    issues.append((num, href))
                except ValueError:
                    continue

        issues = sorted(set(issues), key=lambda x: x[0], reverse=True)[:count]
        issue_urls = [(num, urljoin(BASE_URL, href)) for num, href in issues]
        print(f"[INFO] Found latest {count} issues: {[u for _, u in issue_urls]}")
        return issue_urls

    def slugify(self, text: str) -> str:
        return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

    async def scrape_issue(self, session: aiohttp.ClientSession, issue_num: int, issue_url: str):
        html = await self.fetch(session, issue_url)
        soup = BeautifulSoup(html, "html.parser")

        raw_html = str(soup)
        start_idx = raw_html.find('<h1 id="news">')
        if start_idx != -1:
            raw_html = raw_html[start_idx:]
        parts = raw_html.split("<hr")

        for part in parts:
            part_soup = BeautifulSoup(part, "html.parser")

            title_tag = None
            for h1 in part_soup.find_all("h1"):
                h1_text = h1.get_text(strip=True)
                if h1.get("id") == "news":
                    continue
                if h1_text.lower().startswith("issue"):
                    continue
                if len(h1_text) < 3:
                    continue
                title_tag = h1
                break
            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)

            elements = part_soup.find_all(["p", "li"])
            paragraphs = [
                " ".join(elem.stripped_strings)
                for elem in elements
            ]

            content = "\n".join(p for p in paragraphs if p)
            if "✨ New course!" in content or "Register here" in content:
                continue

            img_tag = None
            for img in part_soup.find_all("img", class_="kg-image"):
                alt_text = img.get("alt", "").lower()
                src_url = img.get("src", "").lower()
                if "promo banner" in alt_text or "ads-and-exclusive-banners" in src_url:
                    continue
                img_tag = img
                break

            image_url = None
            image_filename = None

            if img_tag and img_tag.get("src"):
                image_url = img_tag["src"]
                slug = self.slugify(title)
                img_ext = os.path.splitext(image_url.split("?")[0])[1]
                image_filename = f"issue-{issue_num}_{slug}{img_ext}"
                image_path = os.path.join(self.images_dir, image_filename)
                await self.download_image(session, image_url, image_path)

            self.all_articles.append({
                "issue": issue_num,
                "title": title,
                "url": issue_url,
                "content": content,
                "image_url": image_url,
                "image_filename": image_filename
            })

    async def download_image(self, session: aiohttp.ClientSession, img_url: str, filepath: str):
        if not os.path.exists(filepath):
            async with session.get(img_url) as resp:
                if resp.status == 200:
                    with open(filepath, "wb") as f:
                        f.write(await resp.read())

    async def save_all_articles(self):
        with open(self.news_path, "w", encoding="utf-8") as f:
            json.dump(self.all_articles, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved all articles to {self.news_path}")

    async def run(self):
        async with aiohttp.ClientSession() as session:
            issues = await self.get_last_issues(session)
            for issue_num, issue_url in issues:
                print(f"[INFO] Scraping issue {issue_num}: {issue_url}")
                await self.scrape_issue(session, issue_num, issue_url)

            await self.save_all_articles()


if __name__ == "__main__":
    scraper = BatchScraper()
    asyncio.run(scraper.run())
