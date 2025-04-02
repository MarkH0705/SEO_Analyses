import os
import re
import json
import logging
import requests
import chardet
from bs4 import BeautifulSoup
from datetime import datetime

from chatbot import Chatbot

logging.basicConfig(level=logging.INFO)

class SEOPageOptimizer:
    """
    Diese Klasse extrahiert die Texte in Blöcken, lässt ChatGPT:
     - globale Analyse (analysis_original_text)
     - blockweise rewriting (large_text_optimization)
     - globale Beschreibung der Änderungen (describe_improvements)
    und baut eine final HTML + JSON-Report.
    """

    def __init__(self,
                 output_dir="",
                 prompts_file="",
                 google_ads_keywords=""
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.prompts = self.load_prompts(prompts_file)
        self.google_ads_keywords = google_ads_keywords

    def load_prompts(self, prompts_file):
        logging.info(f"Lade Prompts aus Datei: {prompts_file}")
        if not os.path.exists(prompts_file):
            logging.warning(f"Prompt-Datei {prompts_file} nicht gefunden! Rückgabe leeres Dict.")
            return {}
        with open(prompts_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def fetch_html(self, url: str) -> str:
        logging.info(f"Lade HTML (raw) von {url}...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        raw_data = response.content
        detected = chardet.detect(raw_data)
        encoding = "utf-8"
        logging.info(f"chardet sagt: {encoding} (Confidence: {detected['confidence']})")

        # Nur eine Dekodierung
        text_data = raw_data.decode(encoding, errors="replace")

        logging.info("HTML erfolgreich geladen und manuell dekodiert.")
        return text_data

    def extract_dom_texts(self, html: str):
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        meta_desc_tag = soup.find("meta", attrs={"name":"description"})
        title_text = title_tag.get_text(strip=True) if title_tag else ""
        meta_desc_text = meta_desc_tag.get("content", "").strip() if meta_desc_tag else ""

        text_blocks = []
        block_id_num = 1

        # Body
        for tag_name in ["p","h1","h2","h3","h4","h5","h6","li"]:
            for elem in soup.find_all(tag_name):
                txt = elem.get_text(separator=" ", strip=True)
                if txt:
                    block_id = f"BLOCK_{block_id_num:03d}"
                    text_blocks.append({
                        "id": block_id,
                        "elem": elem,
                        "source": "body",
                        "original_text": txt,
                        "optimized_text": None
                    })
                    block_id_num += 1

        # Link-Attribute
        for a_tag in soup.find_all("a"):
            if a_tag.has_attr("title"):
                txt = a_tag["title"].strip()
                if txt:
                    block_id = f"BLOCK_{block_id_num:03d}"
                    text_blocks.append({
                        "id": block_id,
                        "elem": a_tag,
                        "source": "attribute",
                        "attr_name": "title",
                        "original_text": txt,
                        "optimized_text": None
                    })
                    block_id_num += 1

        # Accordion-Attribute
        for div_tag in soup.find_all(attrs={"data-accordion-title": True}):
            txt = div_tag["data-accordion-title"].strip()
            if txt:
                block_id = f"BLOCK_{block_id_num:03d}"
                text_blocks.append({
                    "id": block_id,
                    "elem": div_tag,
                    "source": "attribute",
                    "attr_name": "data-accordion-title",
                    "original_text": txt,
                    "optimized_text": None
                })
                block_id_num += 1

        logging.info(f"Blocks gesamt: {len(text_blocks)}")
        return text_blocks, title_text, meta_desc_text, soup

    def build_big_string(self, text_blocks):
        """
        Baut 1 großen String => Original-Blöcke
        """
        lines = []
        for b in text_blocks:
            lines.append(f"---{b['id']}_START---")
            lines.append(b["original_text"])
            lines.append(f"---{b['id']}_END---")
        return "\n".join(lines)

    def build_big_string_optimized(self, text_blocks):
        """
        Dasselbe, aber mit optimized_text
        """
        lines = []
        for b in text_blocks:
            text = b["optimized_text"] or "(keine Optimierung)"
            lines.append(f"---{b['id']}_START---")
            lines.append(text)
            lines.append(f"---{b['id']}_END---")
        return "\n".join(lines)

    def generate_llm_text(self, agent_key, **kwargs):
        if agent_key not in self.prompts:
            logging.warning(f"Agent-Key '{agent_key}' nicht in JSON. Return ''")
            return ""
        system_prompt = self.prompts[agent_key].get("system_prompt","")
        user_template = self.prompts[agent_key].get("user_prompt_template","")
        if not user_template:
            logging.warning(f"Kein 'user_prompt_template' für agent_key '{agent_key}'. Return leer.")
            return ""

        user_prompt = user_template.format(**kwargs)
        cb = Chatbot(systemprompt=system_prompt, userprompt=user_prompt)
        response = cb.chat()
        return response.strip()

    def parse_llm_response(self, llm_text):
        pattern = re.compile(r'---(BLOCK_\d+)_START---\s*(.*?)\s*---\1_END---', re.DOTALL)
        found = pattern.findall(llm_text)
        return found

    # ================== 1) Globale Analyse ==================
    def do_pre_analysis(self, text_blocks):
        full_text = self.build_big_string(text_blocks)
        analysis = self.generate_llm_text("analysis_original_text", full_text=full_text, google_ads_keywords=self.google_ads_keywords)
        return analysis

    # ================== 2) SEO-Optimierung ==================
    def do_large_text_optimization(self, text_blocks):
        full_text = self.build_big_string(text_blocks)
        llm_out = self.generate_llm_text(
            "large_text_optimization",
            full_text=full_text,
            google_ads_keywords=self.google_ads_keywords
        )
        results = self.parse_llm_response(llm_out)
        block_map = { b["id"]: b for b in text_blocks }
        for b_id, new_txt in results:
            block_map[b_id]["optimized_text"] = new_txt.strip()
        return text_blocks

    # ================== 3) Globale Beschreibung =============
    def describe_improvements_global(self, text_blocks):
        """
        Original => build_big_string
        Neu => build_big_string_optimized
        => 'describe_improvements' => 1 Gesamter Bericht
        """
        original_str = self.build_big_string(text_blocks)
        optimized_str = self.build_big_string_optimized(text_blocks)

        desc = self.generate_llm_text(
            "describe_improvements",
            original_text=original_str,
            optimized_text=optimized_str
        )
        return desc

    # ================== 4) Re-Inject in HTML ================
    def inject_content(self, soup, text_blocks, new_title=None, new_desc=None):
        for b in text_blocks:
            opt_txt = b["optimized_text"]
            if not opt_txt:
                continue
            opt_txt_html = opt_txt.replace('\n', '<br/>')
            elem = b["elem"]
            if b["source"] == "body":
                elem.clear()
                elem.append(opt_txt)
            elif b["source"] == "attribute":
                attr_name = b["attr_name"]
                elem[attr_name] = opt_txt

        # Falls Title / Desc
        if new_title:
            title_tag = soup.find("title")
            if not title_tag:
                head = soup.find("head")
                if head:
                    title_tag = soup.new_tag("title")
                    head.append(title_tag)
            if title_tag:
                title_tag.string = new_title

        if new_desc:
            desc_tag = soup.find("meta", attrs={"name":"description"})
            if not desc_tag:
                head = soup.find("head")
                if head:
                    new_meta = soup.new_tag("meta", attrs={"name":"description", "content": new_desc})
                    head.append(new_meta)
            else:
                desc_tag["content"] = new_desc

        # charset
        head = soup.find("head")
        if head and not head.find("meta", attrs={"charset": True}):
            meta_charset = soup.new_tag("meta", charset="utf-8")
            head.insert(0, meta_charset)

        return soup

    def save_html(self, soup, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(str(soup))
        logging.info(f"✅ Datei gespeichert: {filepath}")

    # ================== 5) Hauptworkflow ====================
    def optimize_page(self, url: str, outfile: str):
        logging.info(f"SEO-Optimierung startet => {url} => {outfile}")

        # 1) HTML + extrahieren
        html = self.fetch_html(url)
        blocks, old_title, old_desc, soup = self.extract_dom_texts(html)

        # 2) Analyse Original
        analysis_text = self.do_pre_analysis(blocks)

        # 3) SEO => block rewriting
        blocks = self.do_large_text_optimization(blocks)

        # 4) Globale Beschreibung
        improvement_desc = self.describe_improvements_global(blocks)

        # 5) Title & Desc => optional
        new_title = old_title
        new_desc  = old_desc

        # 6) Re-Inject
        new_soup = self.inject_content(soup, blocks, new_title, new_desc)
        self.save_html(new_soup, outfile)

        # 7) combined report
        combined_data = {
            "analysis_of_original_text": analysis_text,
            "improvement_description": improvement_desc,
            "blocks": []
        }
        for b in blocks:
            combined_data["blocks"].append({
                "id": b["id"],
                "original_text": b["original_text"],
                "optimized_text": b["optimized_text"]
            })

        # Hier: Aus url einen kurzen Dateinamen bauen:
        # z.B. alle Slashes entfernen, oder du kannst parse_url
        sanitized_url = url.replace("/", "")
        report_filename = f"report_{sanitized_url}.json"

        json_report = os.path.join(self.output_dir, report_filename)
        with open(json_report, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)

        logging.info(f"Fertig! => {outfile}\nReport => {json_report}")

