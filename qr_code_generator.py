# ============================================================
# QR SVG GENERATOR
# ============================================================
# Autor: Kacper 
#
# Funkcje:
# - generuje wektorowe kody QR w formacie SVG
# - wczytuje linki z pliku TXT
# - zapisuje pliki jako:
#       nazwa_domeny_QR.svg
# - QR są statyczne i działają tak długo jak link
#
# Wymagania:
# pip install qrcode[pil]
#
# Jak używać:
# 1. Utwórz plik links.txt
# 2. Wklej po jednym linku w każdej linii
# 3. Uruchom:
#       python qr_generator.py
#
# Gotowe pliki pojawią się w folderze:
#       output_qr
# ============================================================

import os
from urllib.parse import urlparse

import qrcode
import qrcode.image.svg


# =========================
# KONFIGURACJA
# =========================

INPUT_FILE = "links.txt"
OUTPUT_DIR = "output_qr"


# =========================
# FUNKCJE
# =========================

def sanitize_filename(name):
    """
    Usuwa niedozwolone znaki z nazwy pliku.
    """
    forbidden = '<>:"/\\|?*'
    for char in forbidden:
        name = name.replace(char, "_")

    name = name.replace("www.", "")
    return name


def extract_domain(url):
    """
    Pobiera domenę z URL.
    """
    parsed = urlparse(url)

    domain = parsed.netloc

    if not domain:
        domain = parsed.path

    return sanitize_filename(domain)


def generate_qr_svg(url, output_path):
    """
    Generuje QR jako SVG.
    """

    factory = qrcode.image.svg.SvgImage

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )

    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(image_factory=factory)

    with open(output_path, "wb") as f:
        img.save(f)


# =========================
# MAIN
# =========================

def main():

    if not os.path.exists(INPUT_FILE):
        print(f"\n[ERROR] Nie znaleziono pliku: {INPUT_FILE}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        links = [line.strip() for line in file if line.strip()]

    if not links:
        print("\n[ERROR] Plik links.txt jest pusty")
        return

    print("\n==============================")
    print(" GENEROWANIE QR SVG")
    print("==============================\n")

    generated = 0

    for url in links:

        try:
            domain = extract_domain(url)

            filename = f"{domain}_QR.svg"

            output_path = os.path.join(OUTPUT_DIR, filename)

            generate_qr_svg(url, output_path)

            print(f"[OK] {filename}")

            generated += 1

        except Exception as e:
            print(f"[ERROR] {url}")
            print(f"        {e}")

    print("\n==============================")
    print(f"Wygenerowano: {generated} kodów QR")
    print(f"Folder: {OUTPUT_DIR}")
    print("==============================\n")


if __name__ == "__main__":
    main()