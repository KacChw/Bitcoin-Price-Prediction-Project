import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Tworzymy słownik z Twoimi surowymi danymi
data = {
    'Kraj': ['Austria', 'Belgia', 'Bułgaria', 'Chorwacja', 'Cypr', 'Czechy', 'Dania', 'Estonia', 'Finlandia', 
             'Francja', 'Grecja', 'Hiszpania', 'Holandia', 'Irlandia', 'Litwa', 'Luksemburg', 'Łotwa', 'Malta', 
             'Niemcy', 'Polska', 'Portugalia', 'Rumunia', 'Słowacja', 'Słowenia', 'Szwecja', 'Węgry', 'Włochy'],
    'x1_PKB': [123.0, 118.0, 62.0, 76.0, 94.0, 91.0, 133.0, 82.0, 108.0, 101.0, 67.0, 86.0, 130.0, 235.0, 89.0, 240.0, 71.0, 105.0, 115.0, 80.0, 79.0, 77.0, 73.0, 91.0, 115.0, 76.0, 97.0],
    'x2_Gini': [28.1, 24.2, 38.4, 29.7, 29.4, 23.7, 28.2, 31.8, 26.6, 29.8, 31.4, 32.0, 26.4, 26.9, 35.3, 30.5, 34.2, 31.1, 29.4, 26.3, 33.7, 32.0, 21.2, 23.0, 27.7, 29.0, 32.7],
    'x3_Ubowstwo_en': [8.8, 5.1, 22.5, 6.1, 20.9, 6.9, 4.2, 15.5, 2.6, 10.2, 19.2, 17.7, 5.2, 8.9, 17.5, 3.5, 15.4, 7.8, 8.2, 11.5, 20.8, 19.8, 7.1, 1.9, 3.6, 10.4, 9.5],
    'x4_Cena_pradu': [0.28, 0.43, 0.11, 0.14, 0.35, 0.32, 0.38, 0.22, 0.23, 0.26, 0.24, 0.23, 0.47, 0.40, 0.21, 0.20, 0.25, 0.13, 0.41, 0.21, 0.23, 0.42, 0.19, 0.20, 0.21, 0.12, 0.33],
    'x5_Bezrobocie': [5.1, 5.6, 4.3, 6.3, 6.4, 2.6, 5.1, 6.4, 7.2, 7.3, 11.1, 12.1, 3.6, 4.3, 6.8, 5.2, 6.5, 2.9, 3.1, 2.8, 6.5, 5.6, 5.8, 3.7, 7.7, 4.1, 7.8],
    'x6_RD': [3.3, 3.4, 0.8, 1.4, 0.7, 2.0, 3.0, 1.8, 3.2, 2.2, 1.5, 1.4, 2.3, 1.0, 1.0, 1.0, 0.9, 0.5, 3.1, 1.4, 1.7, 0.5, 1.0, 2.1, 3.6, 1.4, 1.3],
    'x7_Inflacja': [3.1, 2.2, 2.8, 4.6, 1.5, 1.5, 1.0, 3.5, 2.5, 2.0, 3.4, 3.4, 2.6, 3.6, 4.4, 3.8, 3.4, 2.3, 2.8, 3.2, 2.7, 9.0, 3.7, 2.4, 1.5, 2.1, 1.6],
    'x8_PPS': [37906, 35965, 12056, 15994, 21463, 19523, 38732, 17410, 32237, 29334, 16009, 25160, 39685, 39334, 19348, 46197, 14931, 21276, 37906, 19523, 17389, 13526, 14823, 21276, 30338, 13460, 24605],
    'x9_HDI': [0.93, 0.94, 0.80, 0.87, 0.91, 0.90, 0.95, 0.90, 0.94, 0.91, 0.89, 0.91, 0.95, 0.95, 0.88, 0.93, 0.88, 0.92, 0.95, 0.88, 0.87, 0.83, 0.86, 0.93, 0.95, 0.85, 0.91],
    'x10_Dlug': [76.5, 107.1, 28.4, 61.2, 69.1, 44.1, 29.7, 22.9, 81.3, 117.7, 149.7, 103.2, 44.5, 41.2, 38.3, 27.9, 43.6, 49.5, 62.1, 54.2, 95.4, 52.4, 56.0, 67.1, 31.5, 73.5, 137.8],
    'x11_OZE': [36.4, 14.7, 21.6, 30.2, 19.4, 19.3, 45.1, 42.2, 47.9, 22.2, 22.7, 23.3, 17.1, 16.1, 31.2, 11.6, 43.3, 10.1, 23.8, 19.0, 34.8, 25.1, 17.5, 25.0, 66.0, 16.5, 19.8],
    'x12_CO2': [6.4, 7.1, 6.2, 4.5, 6.8, 8.3, 4.5, 8.3, 5.8, 4.1, 5.0, 4.5, 6.9, 6.1, 4.6, 10.5, 3.5, 3.4, 7.0, 7.8, 3.4, 3.7, 5.9, 5.5, 3.4, 4.6, 5.2],
    'x13_Recykling': [62.8, 55.8, 16.7, 36.0, 15.3, 43.5, 46.6, 37.9, 44.8, 40.3, 17.4, 41.4, 58.4, 40.8, 49.3, 56.5, 50.8, 11.5, 68.7, 27.6, 30.6, 12.4, 48.6, 62.4, 39.4, 33.4, 50.8],
    'x14_Auta_EV': [18.5, 15.2, 1.8, 2.5, 4.1, 2.1, 32.4, 4.6, 24.1, 20.8, 1.9, 6.5, 35.1, 15.6, 3.2, 45.2, 3.8, 6.2, 25.4, 2.4, 8.1, 2.2, 1.9, 4.8, 55.4, 4.1, 4.2],
    'x15_Eko_Uprawy': [27.0, 7.43, 4.0, 9.0, 8.76, 16.0, 11.5, 22.58, 13.5, 9.47, 16.0, 11.91, 5.06, 4.97, 8.77, 7.19, 15.59, 0.77, 11.11, 4.88, 21.23, 6.16, 14.79, 11.94, 16.66, 6.1, 19.49],
    'x17_Podatki_Srod': [4.46, 5.17, 9.71, 8.74, 5.69, 4.14, 4.63, 7.40, 5.15, 4.28, 9.40, 4.37, 7.66, 4.50, 5.40, 2.86, 7.17, 4.69, 4.43, 8.17, 5.75, 8.65, 5.35, 7.36, 3.87, 5.58, 6.48],
    'x18_Odpady': [585, 759, 445, 478, 633, 570, 845, 373, 609, 561, 507, 467, 551, 586, 517, 793, 461, 651, 620, 364, 535, 301, 496, 495, 449, 416, 491],
    'x19_Eco_Score': [136, 125, 58, 72, 88, 98, 152, 114, 148, 121, 91, 115, 142, 128, 103, 151, 82, 94, 132, 91, 104, 64, 85, 112, 158, 78, 118],
    'x20_Wydatki_Eko': [274, 84, 6, 24, 2, 22, 365, 72, 73, 176, 6, 55, 81, 47, 18, 259, 6, 2, 181, 8, 5, 2, 7, 27, 248, 3, 62]
}

df = pd.DataFrame(data)

# 2. Oddzielamy nazwy krajów od cech numerycznych
kraje = df['Kraj']
zmienne_numeryczne = df.drop(columns=['Kraj'])

# 3. STANDARYZACJA DANYCH
scaler = StandardScaler()
dane_ustandaryzowane = scaler.fit_transform(zmienne_numeryczne)
df_scaled = pd.DataFrame(dane_ustandaryzowane, columns=zmienne_numeryczne.columns)

# 4. MACIERZ KORELACJI
macierz_korelacji = df_scaled.corr()

# 5. RYSOWANIE HEATMAPY (Wykresu korelacji) - poprawione 'coolwarm'
plt.figure(figsize=(14, 10))
sns.heatmap(macierz_korelacji, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
plt.title('Macierz korelacji zmiennych ekonomicznych i środowiskowych w UE', fontsize=16)
plt.tight_layout()
plt.show()


import itertools

# 1. Definiujemy zmienną objaśnianą (Y) oraz kandydatów na zmienne objaśniające (X)
target_Y = 'x20_Wydatki_Eko'
features_X = [col for col in df_scaled.columns if col != 'Kraj' and col != target_Y]

# Pobieramy macierz korelacji (potrzebna do wzoru Hellwiga)
corr_matrix = df_scaled.drop(columns=['Kraj']).corr()

best_H = 0
best_combination = []

# 2. Pętla sprawdzająca kombinacje (od 1 do 4 zmiennych)
for r in range(1, 5):
    for comb in itertools.combinations(features_X, r):
        H_m = 0
        for x_j in comb:
            # Korelacja x_j z Y (podniesiona do kwadratu)
            r_0j_sq = corr_matrix.loc[x_j, target_Y] ** 2
            
            # Suma korelacji x_j z innymi zmiennymi w tej kombinacji
            sum_r_ij = sum(abs(corr_matrix.loc[x_i, x_j]) for x_i in comb if x_i != x_j)
            
            # Pojemność indywidualna h_mj
            h_mj = r_0j_sq / (1 + sum_r_ij)
            H_m += h_mj
            
        # Szukamy najwyższej wartości H_m
        if H_m > best_H:
            best_H = H_m
            best_combination = comb

# 3. Wyświetlenie wyników
print("=== WYNIKI METODY HELLWIGA ===")
print(f"Najwyższa pojemność integralna (H_m): {best_H:.4f}")
print("Optymalny zestaw zmiennych wybrany do dalszych analiz:")
for i, var in enumerate(best_combination, 1):
    print(f"{i}. {var}")