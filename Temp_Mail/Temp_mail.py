import random
import string
import webbrowser
import pyperclip

class GmailAliasGenerator:
    def __init__(self, base_email=None):
        if base_email and "@gmail.com" in base_email:
            self.base_email = base_email
            self.username = base_email.split('@')[0]
        else:
            self.base_email = None
            self.username = None

    def set_base_email(self, email):
        if "@gmail.com" not in email:
            print("Erreur: Veuillez entrer une adresse Gmail valide (se terminant par @gmail.com)")
            return False
        self.base_email = email
        self.username = email.split('@')[0]
        print(f"Adresse de base définie: {self.base_email}")
        return True

    def generate_dot_alias(self):
        if not self.username:
            print("Veuillez d'abord définir une adresse Gmail de base!")
            return None
        if len(self.username) <= 2:
            print("Nom d'utilisateur trop court pour la méthode des points.")
            return self.base_email
        num_dots = random.randint(1, len(self.username) - 1)
        positions = sorted(random.sample(range(1, len(self.username)), num_dots))
        new_username = ""
        last_pos = 0
        for pos in positions:
            new_username += self.username[last_pos:pos] + "."
            last_pos = pos
        new_username += self.username[last_pos:]
        alias = f"{new_username}@gmail.com"
        print(f"Alias généré (méthode des points): {alias}")
        return alias

    def generate_plus_alias(self, suffix=None):
        if not self.username:
            print("Veuillez d'abord définir une adresse Gmail de base!")
            return None
        if not suffix:
            length = random.randint(4, 10)
            chars = string.ascii_lowercase + string.digits
            suffix = ''.join(random.choice(chars) for _ in range(length))
        alias = f"{self.username}+{suffix}@gmail.com"
        print(f"Alias généré (méthode du suffixe +): {alias}")
        return alias

    def generate_random_alias(self):
        if not self.username:
            print("Veuillez d'abord définir une adresse Gmail de base!")
            return None
        method = random.choice(["dot", "plus"])
        if method == "dot":
            return self.generate_dot_alias()
        else:
            return self.generate_plus_alias()

    def open_gmail(self):
        url = "https://mail.google.com"
        print(f"Ouverture de Gmail dans le navigateur: {url}")
        webbrowser.open(url)

def explain_gmail_aliases():
    explanation = """
=== Comment fonctionnent les alias Gmail ===

Gmail offre deux méthodes principales pour créer des alias sans avoir à créer de nouveaux comptes:

1. Méthode des points (.) : 
   - Pour Gmail, "nom.utilisateur@gmail.com" et "nomutilisateur@gmail.com" sont identiques
   - Vous pouvez placer des points n'importe où dans votre nom d'utilisateur
   - Exemples: j.ohn.doe@gmail.com, jo.hn.doe@gmail.com, john.d.oe@gmail.com

2. Méthode du suffixe (+) :
   - Vous pouvez ajouter "+quelquechose" après votre nom d'utilisateur
   - Exemples: johndoe+shopping@gmail.com, johndoe+travail@gmail.com, johndoe+newsletter@gmail.com

Tous les emails envoyés à ces variantes arriveront dans votre boîte de réception principale.
Ces méthodes sont utiles pour:
- Filtrer automatiquement les emails avec des règles Gmail
- Identifier qui partage votre adresse email
- Suivre quelle entreprise vend vos données

IMPORTANT: Vous devez utiliser une adresse Gmail existante dont vous êtes propriétaire.
Ce script ne crée PAS de nouveaux comptes Gmail.
"""
    print(explanation)
    return explanation

if __name__ == "__main__":
    print("=== Générateur d'Alias Gmail Temporaires ===")
    explain_gmail_aliases()

    generator = GmailAliasGenerator()

    while True:
        print("\nOptions:")
        print("1. Définir l'adresse Gmail de base")
        print("2. Générer un alias avec la méthode des points")
        print("3. Générer un alias avec la méthode du suffixe +")
        print("4. Générer un alias aléatoire")
        print("5. Ouvrir Gmail dans le navigateur")
        print("6. Explication des alias Gmail")
        print("0. Quitter")

        choice = input("Entrez votre choix: ")

        if choice == "1":
            email = input("Entrez votre adresse Gmail: ")
            generator.set_base_email(email)

        elif choice == "2":
            alias = generator.generate_dot_alias()
            if alias:
                try:
                    pyperclip.copy(alias)
                    print("Adresse copiée dans le presse-papier!")
                except:
                    print("Impossible de copier dans le presse-papier. Installez pyperclip (pip install pyperclip)")

        elif choice == "3":
            custom = input("Entrez un suffixe personnalisé (ou appuyez sur Entrée pour un suffixe aléatoire): ")
            alias = generator.generate_plus_alias(custom if custom else None)
            if alias:
                try:
                    pyperclip.copy(alias)
                    print("Adresse copiée dans le presse-papier!")
                except:
                    print("Impossible de copier dans le presse-papier. Installez pyperclip (pip install pyperclip)")

        elif choice == "4":
            alias = generator.generate_random_alias()
            if alias:
                try:
                    pyperclip.copy(alias)
                    print("Adresse copiée dans le presse-papier!")
                except:
                    print("Impossible de copier dans le presse-papier. Installez pyperclip (pip install pyperclip)")

        elif choice == "5":
            generator.open_gmail()

        elif choice == "6":
            explain_gmail_aliases()

        elif choice == "0":
            print("Au revoir!")
            break

        else:
            print("Choix invalide. Veuillez réessayer.")
