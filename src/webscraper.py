import requests
import chardet
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, Comment

class WebsiteScraper:
    """
    Diese Klasse kümmert sich ausschließlich um das Sammeln und Extrahieren
    von Texten aus einer Website.
    """

    def __init__(self, start_url="https://www.rue-zahnspange.de", max_pages=50, 
        excluded_keywords = ["impressum", "datenschutz", "agb"]):
        """
        :param start_url: Die Start-URL der Website, z.B. "https://www.example.com"
        :param max_pages: Maximale Anzahl Seiten, die gecrawlt werden.
        """
        self.start_url = start_url
        self.max_pages = max_pages

        # Hier speichern wir {URL: reiner_Text}
        self.scraped_data = {}
        self.excluded_keywords = excluded_keywords

    def scrape_website(self):
        """
        Startet den Crawl-Vorgang, gefolgt von der Extraktion des Textes
        und dem Sammeln interner Links.
        """
        visited = set()
        to_visit = [self.start_url]
        domain = urlparse(self.start_url).netloc

        while to_visit and len(visited) < self.max_pages:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                response = requests.get(url, timeout=10)

                # Rohdaten holen und Encoding per chardet bestimmen
                raw_data = response.content
                detected = chardet.detect(raw_data)
                # Wenn chardet etwas erkennt, nehmen wir das. Sonst Standard "utf-8".
                encoding = "utf-8"
                text_data = raw_data.decode(encoding, errors="replace")

                # Nur weiterverarbeiten, wenn HTML-Content
                if (response.status_code == 200
                    and "text/html" in response.headers.get("Content-Type", "")):
                    soup = BeautifulSoup(text_data, "html.parser")

                    # Text extrahieren
                    text = self._extract_text_from_soup(soup)
                    self.scraped_data[url] = text

                    # Interne Links sammeln
                    for link in soup.find_all("a", href=True):
                        absolute_link = urljoin(url, link["href"])
                        if urlparse(absolute_link).netloc == domain:
                            if (absolute_link not in visited
                                and absolute_link not in to_visit):
                                to_visit.append(absolute_link)

            except requests.RequestException as e:
                print(f"Fehler beim Abrufen von {url}:\n{e}")

    def _extract_text_from_soup(self, soup):
        """
        Extrahiert aus <p>, <h1>, <h2>, <h3>, <li> reinen Text,
        aber NICHT die, die in .faq4_question oder .faq4_answer stecken.
        Außerdem extrahiert er separat die FAQ-Fragen und -Antworten
        (faq4_question / faq4_answer), damit wir beide Zeilenumbrüche
        dort ebenfalls erhalten.
        """

        # 1) Script/Style/Noscript entfernen
        for script_or_style in soup(["script", "style", "noscript"]):
            script_or_style.decompose()

        # 2) Kommentare entfernen
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # 3) Normale Texte (p, h1, h2, h3, li), ABER nicht innerhalb von .faq4_question / .faq4_answer
        texts = []
        all_normal_tags = soup.find_all(["p", "h1", "h2", "h3", "li"])
        for tag in all_normal_tags:
            # Prüfen, ob das Tag einen Vorfahren hat mit Klasse faq4_question oder faq4_answer
            if tag.find_parent(class_="faq4_question") or tag.find_parent(class_="faq4_answer"):
                continue

            # Hier wichtig: separator="\n", strip=False, damit wir Zeilenumbrüche behalten
            txt = tag.get_text(separator="\n", strip=False)
            # Evtl. willst du doppelte Leerzeilen bereinigen. Das kannst du optional tun.
            if txt.strip():
                texts.append(txt.strip("\r\n"))

        # 4) FAQ-Bereiche (Fragen + Antworten)
        questions = soup.select(".faq4_question")
        answers = soup.select(".faq4_answer")

        # 5) Zusammenführen (Frage + Antwort)
        for q, a in zip(questions, answers):
            q_text = q.get_text(separator="\n", strip=False)
            a_text = a.get_text(separator="\n", strip=False)
            q_text = q_text.strip("\r\n")
            a_text = a_text.strip("\r\n")
            if q_text and a_text:
                combined = f"Frage: {q_text}\nAntwort: {a_text}"
                texts.append(combined)

        # 6) Als String zurückgeben. Wir trennen die einzelnen Elemente durch "\n\n"
        #    (kannst du je nach Wunsch anpassen)
        return "\n\n".join(texts)

    def get_scraped_data(self):
        """
        Gibt das Dictionary {URL: Text} zurück.
        Du kannst damit arbeiten, Seiten filtern, etc.
        """
        return self.scraped_data


def prep_text():
    scraper = WebsiteScraper(start_url="https://www.rue-zahnspange.de", max_pages=20)
    scraper.scrape_website()
    scraped_data = scraper.get_scraped_data()

    page_text_list = []
    filtered_urls = []

    # Alle URLs sammeln, die KEINEN der ausgeschlossenen Begriffe enthalten
    for url in scraped_data.keys():
        # Schauen, ob einer der EXCLUDED_KEYWORDS im URL-String (kleingeschrieben) vorkommt
        if any(keyword in url.lower() for keyword in self.excluded_keywords):
            # Falls ja, überspringen wir diese URL
            continue
        # Sonst nehmen wir sie auf
        filtered_urls.append(url)

        # 3. SEO-Analyse starten (für gefilterte Seiten)
    for url in filtered_urls:
        # Die gesamte Seite analysieren
        page_text = scraped_data[url]
        page_text_list.append(page_text)

    return filtered_urls, page_text_list