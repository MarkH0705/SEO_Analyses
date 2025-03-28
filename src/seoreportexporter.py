import os
import json
import asyncio
import pypandoc
import nest_asyncio
from jinja2 import Template
from playwright.async_api import async_playwright


class SEOReportExporter:
    def __init__(self, seo_json, output_path, intro_json_path, image_paths):
        self.seo_json = seo_json
        self.output_path = output_path
        self.html_path = os.path.join(self.output_path, "final", "preview.html")
        self.pdf_path = os.path.join(self.output_path, "final", "output.pdf")
        self.docx_path = os.path.join(self.output_path, "final", "output.docx")
        self.intro_json_path = intro_json_path
        self.image_paths = image_paths

        os.makedirs(os.path.join(self.output_path, "final"), exist_ok=True)

        self.sections_intro = self.load_intro_texts()

    def load_intro_texts(self):
        with open(self.intro_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_html(self):
        html_template = """
        <!DOCTYPE html>
        <html lang=\"de\">
        <head>
            <meta charset=\"UTF-8\">
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
            <title>Website Analyse</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; color: #333; }
                h1, h2 { text-align: center; color: #2c3e50; }
                .section { margin-bottom: 30px; }
                .img-block { text-align: center; margin: 20px 0; }
                .img-block img { max-width: 90%; height: auto; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                .url { font-size: 1.2em; font-weight: bold; color: #007BFF; margin-bottom: 10px; }
                .header { font-size: 1.1em; font-weight: bold; color: #555; margin-bottom: 10px; }
                .content { white-space: normal; margin-bottom: 20px; }
                .column { border: 1px solid #ccc; padding: 10px; box-sizing: border-box; background-color: #f9f9f9; margin-top: 20px; }
                .page-break { page-break-after: always; }
            </style>
        </head>
        <body>
            <h1>SEO Analyse & Optimierungsreport</h1>

            <div class=\"section\">
                <h2>1. Semantische Beziehungen in Vektor-Räumen</h2>
                <p class=\"content\">{{ intro.embedding_text | safe }}</p>
                <div class=\"img-block\">
                    <img src=\"{{ images.embedding_demo }}\" alt=\"BERT Embedding Visualisierung\">
                </div>
            </div>

            <div class=\"section\">
                <h2>2. Keyword-Recherche mit Google Ads</h2>
                <p class=\"content\">{{ intro.keyword_text | safe }}</p>
                <div class=\"img-block\">
                    <img src=\"{{ images.google_ads_heatmap }}\" alt=\"Keyword Heatmap aus Google Ads\">
                </div>
            </div>

            <div class=\"section\">
                <h2>3. Cosine Similarity erklärt</h2>
                <p class=\"content\">{{ intro.similarity_text | safe }}</p>
                <div class=\"img-block\">
                    <img src=\"{{ images.cosine_similarity_demo }}\" alt=\"Cosine Similarity Plot\">
                </div>
            </div>

            <div class=\"section\">
                <h2>4. Keyword-Abdeckung & Analyse</h2>
                <p class=\"content\">{{ intro.keyword_analysis_text | safe }}</p>
                <div class=\"img-block\">
                    <img src=\"{{ images.wordclouds }}\" alt=\"Wordclouds\">
                </div>
                <div class=\"img-block\">
                    <img src=\"{{ images.similarity_bars }}\" alt=\"Keyword Similarity Balken\">
                </div>
            </div>

            <div class=\"page-break\"></div>

            {% for url, sections in seo_json.items() %}
            <div class=\"section\">
                <p class=\"url\">🌐 Website: {{ url }}</p>
                <p class=\"header\">🔍 Analyse</p>
                <p class=\"content\">{{ sections.Analyse | replace('\\n','<br>') | safe }}</p>
                <p class=\"header\">🧠 Erklärung</p>
                <p class=\"content\">{{ sections.Erklärung | replace('\\n','<br>') | safe }}</p>
                <div class=\"column\">
                    <p class=\"header\">🚀 SEO-Text</p>
                    <p class=\"content\">{{ sections.SEO | replace('\\n','<br>') | safe }}</p>
                </div>
            </div>
            <div class=\"page-break\"></div>
            {% endfor %}
        </body>
        </html>
        """
        template = Template(html_template)
        html_output = template.render(seo_json=self.seo_json, intro=self.sections_intro, images=self.image_paths)

        with open(self.html_path, "w", encoding="utf-8") as f:
            f.write(html_output)

        print("✅ HTML exportiert:", self.html_path)

    async def export_pdf(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            url = "file://" + self.html_path
            await page.goto(url, wait_until="load")
            await page.pdf(
                path=self.pdf_path,
                format="A4",
                margin={"top": "1cm", "right": "1cm", "bottom": "1cm", "left": "1cm"}
            )
            await browser.close()
        print("✅ PDF mit Playwright erstellt:", self.pdf_path)

    def export_docx(self):
        pypandoc.convert_file(
            source_file=self.html_path,
            to="docx",
            outputfile=self.docx_path,
            extra_args=["--standalone"]
        )
        print("✅ Konvertierung nach DOCX abgeschlossen:", self.docx_path)

    def run_all_exports(self):
        self.generate_html()
        nest_asyncio.apply()
        asyncio.run(self.export_pdf())
        self.export_docx()


