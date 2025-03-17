import os
import subprocess

class GitHubManager:
    def __init__(self, pat, mail, user, repo, project):  # Jetzt ohne Default-Werte!
        """Holt GitHub-Zugangsdaten aus Google Colab Secrets."""
        self.github_pat = pat
        self.github_email = mail
        self.github_username = user
        self.github_repo = repo

        if not all([self.github_pat, self.github_email, self.github_username, self.github_repo]):
            raise ValueError("⚠️ Fehlende GitHub-Secrets! Bitte Secrets in Google Colab setzen.")

        self.project_path = project # Dein SEO-Projekt pfad

    def run_command(self, command, cwd=None):
        """Führt einen Shell-Befehl aus und gibt das Ergebnis zurück."""
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Fehler: {command}\n{result.stderr}")
        return result.stdout

    def clone_repo(self):
        """Klonen des GitHub-Repos in Google Colab."""
        print("📥 Klonen des GitHub-Repositories...")

        if os.path.exists("cloned-repo"):
            self.run_command("rm -rf cloned-repo")

        repo_url = f'https://{self.github_pat}@github.com/{self.github_username}/{self.github_repo}.git'
        self.run_command(f"git clone {repo_url} cloned-repo")

    def sync_project(self):
        """Kopiert das gesamte Projekt in das GitHub-Repository und pusht es."""
        if not os.path.exists("cloned-repo"):
            print("⚠️ Repository wurde nicht geklont! Starte zuerst `clone_repo()`.")
            return

        os.chdir("cloned-repo")  # Wechsel ins geklonte Repo

        print("📂 Kopiere das Projekt ins Repository...")
        self.run_command(f"cp -r {self.project_path}/* ./")

        print("🔧 Git-Konfiguration wird gesetzt...")
        self.run_command(f'git config user.email "{self.github_email}"')
        self.run_command(f'git config user.name "{self.github_username}"')

        # ordner und dateitypen festlegen, die nach github gepusht werden sollen
        print("➕ Änderungen hinzufügen...")
        self.run_command("git add notebooks/*.ipynb src/*.py output/*.html output/*.pdf data/keywords/*.xlsx data/prompts/*.json data/faiss_db READ_ME.md TO_DO.md tests")

        print("📌 Änderungen committen...")
        commit_message = "🚀 Automatische Aktualisierung des SEO-Projekts"
        self.run_command(f'git commit -m "{commit_message}"')

        print("⬆️ Änderungen werden auf GitHub gepusht...")
        self.run_command("git push origin main")

        os.chdir("..")  # Zurück ins Hauptverzeichnis
        self.run_command("rm -rf cloned-repo")  # Aufräumen

        print("✅ Repository erfolgreich synchronisiert!")