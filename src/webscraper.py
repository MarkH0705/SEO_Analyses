import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, Comment

class WebsiteScraper:
    """
    Diese Klasse kümmert sich um das Sammeln, Extrahieren und Filtern
    von Texten aus einer Website. Zusätzlich ermöglicht sie eine SEO-Analyse.
    """

    def __init__(self, start_url="https://www.rue-zahnspange.de", max_pages=50, excluded_keywords=None):
        """
        :param start_url: Die Start-URL der Website.
        :param max_pages: Maximale Anzahl Seiten, die gecrawlt werden.
        :param excluded_keywords: Liste von Keywords, die in URLs nicht vorkommen sollen.
        """
        self.start_url = start_url
        self.max_pages = max_pages
        self.excluded_keywords = excluded_keywords if excluded_keywords else ["impressum", "datenschutz", "agb"]

        # Hier speichern wir {URL: reiner_Text}
        self.scraped_data = {}

        # Liste für die gefilterten Texte
        self.filtered_texts = {}

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

                # Rohdaten holen und Encoding setzen
                raw_data = response.content
                encoding = "utf-8"
                text_data = raw_data.decode(encoding, errors="replace")

                # Nur weiterverarbeiten, wenn HTML-Content
                if response.status_code == 200 and "text/html" in response.headers.get("Content-Type", ""):
                    soup = BeautifulSoup(text_data, "html.parser")

                    # Text extrahieren
                    text = self._extract_text_from_soup(soup)
                    self.scraped_data[url] = text

                    # Interne Links sammeln
                    for link in soup.find_all("a", href=True):
                        absolute_link = urljoin(url, link["href"])
                        if urlparse(absolute_link).netloc == domain:
                            if absolute_link not in visited and absolute_link not in to_visit:
                                to_visit.append(absolute_link)

            except requests.RequestException as e:
                print(f"Fehler beim Abrufen von {url}:\n{e}")

    def _extract_text_from_soup(self, soup):
        """
        Extrahiert aus <p>, <h1>, <h2>, <h3>, <li> reinen Text,
        aber NICHT die, die in .faq4_question oder .faq4_answer stecken.
        Außerdem extrahiert er separat die FAQ-Fragen und -Antworten.
        """
        for script_or_style in soup(["script", "style", "noscript"]):
            script_or_style.decompose()

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        texts = []
        all_normal_tags = soup.find_all(["p", "h1", "h2", "h3", "li"])
        for tag in all_normal_tags:
            if tag.find_parent(class_="faq4_question") or tag.find_parent(class_="faq4_answer"):
                continue
            txt = tag.get_text(separator="\n", strip=False)
            if txt.strip():
                texts.append(txt.strip("\r\n"))

        questions = soup.select(".faq4_question")
        answers = soup.select(".faq4_answer")
        for q, a in zip(questions, answers):
            q_text = q.get_text(separator="\n", strip=False).strip("\r\n")
            a_text = a.get_text(separator="\n", strip=False).strip("\r\n")
            if q_text and a_text:
                texts.append(f"Frage: {q_text}\nAntwort: {a_text}")

        return "\n\n".join(texts)

    def get_scraped_data(self):
        """
        Gibt das Dictionary {URL: Text} zurück.
        """
        return self.scraped_data

    def get_filtered_texts(self):
        self.get_scraped_data()
        self.filtered_texts = {
            url: text for url, text in self.scraped_data.items()
            if not any(keyword in url.lower() for keyword in self.excluded_keywords)
            }
        return self.filtered_texts  # Liste der gefilterten Texte zurückgeben
