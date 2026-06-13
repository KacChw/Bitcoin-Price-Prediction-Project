
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# ==========================================
# KROK 1: PRZYGOTOWANIE DANYCH I KORELACJA
# ==========================================

# 1. Surowe dane
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
kraje = df['Kraj']
zmienne_numeryczne = df.drop(columns=['Kraj'])

# 2. Standaryzacja
scaler = StandardScaler()
dane_ustandaryzowane = scaler.fit_transform(zmienne_numeryczne)
df_scaled = pd.DataFrame(dane_ustandaryzowane, columns=zmienne_numeryczne.columns)

# 3. Generowanie macierzy korelacji
corr_matrix = df_scaled.corr()

# 4. Rysowanie heatmapy korelacji
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
plt.title('Macierz korelacji zmiennych ekonomicznych i środowiskowych w UE', fontsize=16)
plt.tight_layout()
# plt.show() # Odkomentuj, jeśli chcesz wygenerować wykres ponownie

# 5. NOWY WYKRES: Wizualizacja całej macierzy danych ustandaryzowanych (Z-score)
df_scaled_map = df_scaled.copy()
df_scaled_map.index = kraje

plt.figure(figsize=(16, 12))
sns.heatmap(df_scaled_map, annot=True, cmap='RdYlBu_r', fmt=".2f", center=0, 
            cbar_kws={'label': 'Odchylenie standardowe od średniej (Z-score)'}, linewidths=0.3)
plt.title('Mapa profilowa ustandaryzowanych zmiennych diagnostycznych dla państw UE', fontsize=16, pad=15)
plt.xlabel('Zmienne diagnostyczne', fontsize=12)
plt.ylabel('Kraje UE', fontsize=12)
plt.tight_layout()
plt.show()

# ==========================================
# WIZUALIZACJA SUROWYCH DANYCH (SZACHOWNICA)
# ==========================================
# Przygotowanie danych (zaokrąglamy tylko wartości numeryczne)
df_surowe_tabela = df.copy()
for col in zmienne_numeryczne.columns:
    df_surowe_tabela[col] = df_surowe_tabela[col].apply(lambda x: f"{x:.2f}" if x % 1 != 0 else f"{int(x)}")

fig, ax = plt.subplots(figsize=(20, 11))
ax.axis('off')
ax.axis('tight')

# Generowanie kolorów dla szachownicy (naprzemienne wiersze biały / blady zielony)
kolory_wierszy = []
for i in range(len(df_surowe_tabela)):
    if i % 2 == 0:
        kolory_wierszy.append(['#ffffff'] * len(df_surowe_tabela.columns))
    else:
        kolory_wierszy.append(['#e8f5e9'] * len(df_surowe_tabela.columns)) # Blada, pastelowa zieleń

# Rysowanie tabeli graficznej
tabela_graficzna = ax.table(
    cellText=df_surowe_tabela.values,
    colLabels=df_surowe_tabela.columns,
    cellLoc='center',
    loc='center',
    cellColours=kolory_wierszy
)

# Stylizacja nagłówków tabeli (ciemnozielony z białym tekstem)
for j in range(len(df_surowe_tabela.columns)):
    komorka_naglowka = tabela_graficzna[0, j]
    komorka_naglowka.set_facecolor('#2e7d32')
    komorka_naglowka.get_text().set_color('white')
    komorka_naglowka.get_text().set_weight('bold')
    komorka_naglowka.get_text().set_fontsize(10)

# Dopasowanie rozmiarów
tabela_graficzna.auto_set_font_size(False)
tabela_graficzna.set_fontsize(9)
tabela_graficzna.scale(1, 1.6) # Rozciągnięcie komórek w pionie dla oddechu

plt.title('Macierz wejściowa surowych danych diagnostycznych (Państwa UE)', fontsize=15, pad=25, weight='bold', color='#1b5e20')
plt.tight_layout()
plt.show()

# ==========================================
# KROK 2: METODA HELLWIGA (SELEKCJA ZMIENNYCH)
# ==========================================

target_Y = 'x20_Wydatki_Eko'
features_X = [col for col in df_scaled.columns if col != target_Y]

best_H = 0
best_combination = []

for r in range(1, 5):
    for comb in itertools.combinations(features_X, r):
        H_m = 0
        for x_j in comb:
            r_0j_sq = corr_matrix.loc[x_j, target_Y] ** 2
            sum_r_ij = sum(abs(corr_matrix.loc[x_i, x_j]) for x_i in comb if x_i != x_j)
            
            h_mj = r_0j_sq / (1 + sum_r_ij)
            H_m += h_mj
            
        if H_m > best_H:
            best_H = H_m
            best_combination = comb

print("\n" + "="*30)
print("=== WYNIKI METODY HELLWIGA ===")
print("="*30)
print(f"Najwyższa pojemność integralna (H_m): {best_H:.4f}")
print("Optymalny zestaw zmiennych wybrany do opisu wydatków na żywność EKO:")
for i, var in enumerate(best_combination, 1):
    print(f"{i}. {var}")
print("="*30)


# ==========================================
# KROK 3: ANALIZA SKŁADOWYCH GŁÓWNYCH (PCA)
# ==========================================

pca = PCA(n_components=2)
pca_results = pca.fit_transform(df_scaled)

df_pca = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2'])
df_pca.insert(0, 'Kraj', kraje)

wyjasniona_wariancja = pca.explained_variance_ratio_
print("\n" + "="*30)
print("=== WYNIKI ANALIZY PCA ===")
print("="*30)
print(f"Składowa PC1 wyjaśnia: {wyjasniona_wariancja[0]*100:.2f}% zmienności zbioru.")
print(f"Składowa PC2 wyjaśnia: {wyjasniona_wariancja[1]*100:.2f}% zmienności zbioru.")
print(f"Łącznie dwie pierwsze składowe wyjaśniają: {(wyjasniona_wariancja[0]+wyjasniona_wariancja[1])*100:.2f}% informacji.")

ladunki = pd.DataFrame(
    pca.components_.T, 
    columns=['PC1_Ladunek', 'PC2_Ladunek'], 
    index=zmienne_numeryczne.columns
)
print("\nŁadunki czynnikowe (Korelacja zmiennych z PC1 i PC2):")
print(ladunki.round(3))
print("="*30)

plt.figure(figsize=(13, 9))
plt.scatter(df_pca['PC1'], df_pca['PC2'], color='teal', edgecolors='black', s=120, alpha=0.8)

for i, txt in enumerate(df_pca['Kraj']):
    plt.annotate(txt, (df_pca['PC1'].iloc[i] + 0.08, df_pca['PC2'].iloc[i] + 0.08), fontsize=10, weight='bold')

plt.axhline(0, color='grey', linestyle='--', linewidth=1)
plt.axvline(0, color='grey', linestyle='--', linewidth=1)

plt.xlabel(f'PC1 ({wyjasniona_wariancja[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'PC2 ({wyjasniona_wariancja[1]*100:.1f}%)', fontsize=12)
plt.title('Kraje UE w przestrzeni dwuwymiarowej składowych głównych', fontsize=15, pad=15)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()


# ==========================================
# KROK 4: TMR HELLWIGA (RANKING LINOWY)
# ==========================================

stymulanty = ['x1_PKB', 'x6_RD', 'x8_PPS', 'x9_HDI', 'x11_OZE', 'x13_Recykling', 'x14_Auta_EV', 'x15_Eko_Uprawy', 'x19_Eco_Score', 'x20_Wydatki_Eko']
destymulanty = ['x2_Gini', 'x3_Ubowstwo_en', 'x4_Cena_pradu', 'x5_Bezrobocie', 'x7_Inflacja', 'x10_Dlug', 'x12_CO2', 'x18_Odpady']

zmienne_ranking = stymulanty + destymulanty
df_ranking_input = df_scaled[zmienne_ranking]

wzorzec = {}
for col in zmienne_ranking:
    if col in stymulanty:
        wzorzec[col] = df_ranking_input[col].max()
    else:
        wzorzec[col] = df_ranking_input[col].min()

wzorzec_series = pd.Series(wzorzec)
odleglosci = np.sqrt(((df_ranking_input - wzorzec_series) ** 2).sum(axis=1))

d_bar = odleglosci.mean()
s_d = odleglosci.std()
d_0 = d_bar + 2 * s_d

miernik_hellwiga = 1 - (odleglosci / d_0)

ranking_hellwiga = pd.DataFrame({
    'Kraj': kraje,
    'Miernik_Hellwiga': miernik_hellwiga
}).sort_values(by='Miernik_Hellwiga', ascending=False).reset_index(drop=True)

print("\n" + "="*30)
print("=== RANKING TMR HELLWIGA ===")
print("="*30)
for idx, row in ranking_hellwiga.iterrows():
    print(f"{idx+1:2d}. {row['Kraj']:12s} | Wynik: {row['Miernik_Hellwiga']:.4f}")
print("="*30)


# ==========================================
# KROK 5: METODA TOPSIS (RANKING LINOWY #2)
# ==========================================

df_min_max = zmienne_numeryczne[zmienne_ranking].copy()
for col in zmienne_ranking:
    min_val = df_min_max[col].min()
    max_val = df_min_max[col].max()
    if max_val != min_val:
        df_min_max[col] = (df_min_max[col] - min_val) / (max_val - min_val)
    else:
        df_min_max[col] = 0.0

wzorzec_topsis = {}
antywzorzec_topsis = {}

for col in zmienne_ranking:
    if col in stymulanty:
        wzorzec_topsis[col] = 1.0
        antywzorzec_topsis[col] = 0.0
    else:
        wzorzec_topsis[col] = 0.0
        antywzorzec_topsis[col] = 1.0

w_series = pd.Series(wzorzec_topsis)
a_series = pd.Series(antywzorzec_topsis)

d_plus = np.sqrt(((df_min_max - w_series) ** 2).sum(axis=1))
d_minus = np.sqrt(((df_min_max - a_series) ** 2).sum(axis=1))

wspolczynnik_topsis = d_minus / (d_plus + d_minus)

ranking_topsis = pd.DataFrame({
    'Kraj': kraje,
    'Miernik_TOPSIS': wspolczynnik_topsis
}).sort_values(by='Miernik_TOPSIS', ascending=False).reset_index(drop=True)

print("\n" + "="*30)
print("=== RANKING METODĄ TOPSIS ===")
print("="*30)
for idx, row in ranking_topsis.iterrows():
    print(f"{idx+1:2d}. {row['Kraj']:12s} | Wynik: {row['Miernik_TOPSIS']:.4f}")
print("="*30)


# ==========================================
# KROK 6: METODA COPRAS (RANKING LINOWY #3)
# ==========================================

df_copras_norm = zmienne_numeryczne[zmienne_ranking].copy()
df_copras_norm = df_copras_norm / df_copras_norm.sum()

waga = 1.0 / len(zmienne_ranking)
df_copras_weighted = df_copras_norm * waga

S_plus = df_copras_weighted[stymulanty].sum(axis=1)
S_minus = df_copras_weighted[destymulanty].sum(axis=1)

sum_S_minus = S_minus.sum()
inv_S_minus_sum = (1 / S_minus).sum()

Q_i = S_plus + (sum_S_minus / (S_minus * inv_S_minus_sum))
N_i = (Q_i / Q_i.max()) * 100

ranking_copras = pd.DataFrame({
    'Kraj': kraje,
    'Miernik_COPRAS': Q_i,
    'Uzytecznosc_%': N_i
}).sort_values(by='Miernik_COPRAS', ascending=False).reset_index(drop=True)

print("\n" + "="*45)
print("=== RANKING METODĄ COPRAS ===")
print("="*45)
for idx, row in ranking_copras.iterrows():
    print(f"{idx+1:2d}. {row['Kraj']:12s} | Wynik Q_i: {row['Miernik_COPRAS']:.4f} | Użyteczność: {row['Uzytecznosc_%']:.2f}%")
print("="*45)


# ====================================================================
# WIZUALIZACJA: ZBIORCZA TABELA RANKINGÓW (GRADIENT I WĘŻSZE KOLUMNY)
# ====================================================================
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Obliczenia czystych wyników numerycznych (bez wcześniejszego formatowania na string)
df_h_in = df_scaled[zmienne_ranking]
wzorzec_h = {col: (df_h_in[col].max() if col in stymulanty else df_h_in[col].min()) for col in zmienne_ranking}
odleglosci_h = np.sqrt(((df_h_in - pd.Series(wzorzec_h)) ** 2).sum(axis=1))
miernik_hellwiga = 1 - (odleglosci_h / (odleglosci_h.mean() + 2 * odleglosci_h.std()))
r_h_tab = pd.DataFrame({'Kraj (Hellwig)': kraje, 'Wynik_H_raw': miernik_hellwiga}).sort_values(by='Wynik_H_raw', ascending=False).reset_index(drop=True)
r_h_tab['Miejsce'] = r_h_tab.index + 1

df_t_in = zmienne_numeryczne[zmienne_ranking].copy()
for col in zmienne_ranking: df_t_in[col] = (df_t_in[col] - df_t_in[col].min()) / (df_t_in[col].max() - df_t_in[col].min())
w_t = pd.Series({col: (1.0 if col in stymulanty else 0.0) for col in zmienne_ranking})
a_t = pd.Series({col: (0.0 if col in stymulanty else 1.0) for col in zmienne_ranking})
miernik_topsis = np.sqrt(((df_t_in - a_t) ** 2).sum(axis=1)) / (np.sqrt(((df_t_in - w_t) ** 2).sum(axis=1)) + np.sqrt(((df_t_in - a_t) ** 2).sum(axis=1)))
r_t_tab = pd.DataFrame({'Kraj (TOPSIS)': kraje, 'Wynik_T_raw': miernik_topsis}).sort_values(by='Wynik_T_raw', ascending=False).reset_index(drop=True)
r_t_tab['Miejsce'] = r_t_tab.index + 1

df_c_in = zmienne_numeryczne[zmienne_ranking] / zmienne_numeryczne[zmienne_ranking].sum()
S_plus = df_c_in[stymulanty].sum(axis=1)
S_minus = df_c_in[destymulanty].sum(axis=1)
uzytecznosci = ((S_plus + (S_minus.sum() / (S_minus * (1 / S_minus).sum()))) / (S_plus + (S_minus.sum() / (S_minus * (1 / S_minus).sum()))).max()) * 100
r_c_tab = pd.DataFrame({'Kraj (COPRAS)': kraje, 'Wynik_C_raw': uzytecznosci}).sort_values(by='Wynik_C_raw', ascending=False).reset_index(drop=True)
r_c_tab['Miejsce'] = r_c_tab.index + 1

# Budowa zunifikowanej tabeli bazowej
df_plot_ranking = pd.DataFrame({'Miejsce': range(1, 28)})
df_plot_ranking = df_plot_ranking.merge(r_h_tab[['Miejsce', 'Kraj (Hellwig)', 'Wynik_H_raw']], on='Miejsce')
df_plot_ranking = df_plot_ranking.merge(r_t_tab[['Miejsce', 'Kraj (TOPSIS)', 'Wynik_T_raw']], on='Miejsce')
df_plot_ranking = df_plot_ranking.merge(r_c_tab[['Miejsce', 'Kraj (COPRAS)', 'Wynik_C_raw']], on='Miejsce')

# Przygotowanie wspólnego gradientu (Mapa kolorów YlGn - od żółtego do ciemnej zieleni)
cmap = plt.cm.get_cmap('YlGn')

# Wyznaczamy ekstrema dla każdej metody osobno, aby poprawnie wyskalować odcienie (0 = najjaśniejszy, 1 = najciemniejszy)
min_h, max_h = df_plot_ranking['Wynik_H_raw'].min(), df_plot_ranking['Wynik_H_raw'].max()
min_t, max_t = df_plot_ranking['Wynik_T_raw'].min(), df_plot_ranking['Wynik_T_raw'].max()
min_c, max_c = df_plot_ranking['Wynik_C_raw'].min(), df_plot_ranking['Wynik_C_raw'].max()

cell_colours = []
cell_text = []

for i in range(len(df_plot_ranking)):
    row = df_plot_ranking.iloc[i]
    
    # Normalizacja wyników do przedziału [0.05, 0.6] -> ucięte od góry, żeby czarny tekst był idealnie czytelny na zielonym tle
    norm_h = 0.05 + 0.55 * ((row['Wynik_H_raw'] - min_h) / (max_h - min_h))
    norm_t = 0.05 + 0.55 * ((row['Wynik_T_raw'] - min_t) / (max_t - min_t))
    norm_c = 0.05 + 0.55 * ((row['Wynik_C_raw'] - min_c) / (max_c - min_c))
    
    # Pobranie dokładnie tych samych kodów HEX z jednego gradientu
    color_h = mcolors.to_hex(cmap(norm_h))
    color_t = mcolors.to_hex(cmap(norm_t))
    color_c = mcolors.to_hex(cmap(norm_c))
    
    # Tło dla kolumn tekstowych (klasyczna, bardzo delikatna szachownica dla kontrastu)
    bg_text = '#ffffff' if i % 2 == 0 else '#f9f9f9'
    
    # Mapowanie kolorów na kolumny: Miejsce, Kraj(H), Wynik(H), Kraj(T), Wynik(T), Kraj(C), Wynik(C)
    row_colors = [bg_text, bg_text, color_h, bg_text, color_t, bg_text, color_c]
    cell_colours.append(row_colors)
    
    # Jednoczesne bezpieczne formatowanie tekstu do wyświetlenia
    row_text = [
        str(int(row['Miejsce'])),
        str(row['Kraj (Hellwig)']),
        f"{row['Wynik_H_raw']:.4f}",
        str(row['Kraj (TOPSIS)']),
        f"{row['Wynik_T_raw']:.4f}",
        str(row['Kraj (COPRAS)']),
        f"{row['Wynik_C_raw']:.2f}%"
    ]
    cell_text.append(row_text)

# Rysowanie tabeli graficznej
fig, ax = plt.subplots(figsize=(13, 12)) # Zmniejszona szerokość figury z 16 na 13
ax.axis('off')
ax.axis('tight')

# SZTYWNE, WĘŻSZE SZEROKOŚCI KOLUMN (Suma = 0.82 szerokości wykresu - idealne proporcje dla oka)
col_widths = [0.05, 0.17, 0.09, 0.17, 0.09, 0.17, 0.09]
naglowki = ['Miejsce', 'Kraj (Hellwig)', 'Wynik', 'Kraj (TOPSIS)', 'Wynik', 'Kraj (COPRAS)', 'Użyteczność']

tabela_wynikowa = ax.table(
    cellText=cell_text,
    colLabels=naglowki,
    cellLoc='center',
    loc='center',
    cellColours=cell_colours,
    colWidths=col_widths
)

# Stylizacja ciemnozielonego nagłówka
for j in range(len(naglowki)):
    cell = tabela_wynikowa[0, j]
    cell.set_facecolor('#2e7d32')
    cell.get_text().set_color('white')
    cell.get_text().set_weight('bold')
    cell.get_text().set_fontsize(11)

tabela_wynikowa.auto_set_font_size(False)
tabela_wynikowa.set_fontsize(10)
tabela_wynikowa.scale(1, 1.6) # Rozciągnięcie wierszy w pionie dla eleganckiego "oddechu"

plt.title('Porównanie wyników i stabilności porządkowania liniowego państw UE', fontsize=14, pad=25, weight='bold', color='#1b5e20')
plt.tight_layout()
plt.show()

# ==========================================
# KROK 7: ANALIZA SKUPIEŃ (METODA WARDA)
# ==========================================

X_clustering = df_scaled[zmienne_ranking]
linked = linkage(X_clustering, method='ward')

plt.figure(figsize=(13, 8))
dendrogram(
    linked,
    labels=kraje.values,
    orientation='top',
    distance_sort='descending',
    show_leaf_counts=True,
    leaf_font_size=11
)

plt.title('Dendrogram podziału państw UE metodą Warda', fontsize=16, pad=15)
plt.xlabel('Kraje UE', fontsize=12)
plt.ylabel('Odległość wiązania (Wariancja)', fontsize=12)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# ==========================================
# KROK 8: METODA K-ŚREDNICH (K-MEANS)
# ==========================================

LICZBA_SKUPIEN = 3
kmeans = KMeans(n_clusters=LICZBA_SKUPIEN, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_clustering)

df['Grupa_KMeans'] = clusters

print("\n" + "="*45)
print(f"=== PODZIAŁ UE NA {LICZBA_SKUPIEN} GRUP METODĄ K-ŚREDNICH ===")
print("="*45)

for g in range(LICZBA_SKUPIEN):
    kraje_w_grupie = df[df['Grupa_KMeans'] == g]['Kraj'].tolist()
    print(f"\nGRUPA {g+1} (Liczba państw: {len(kraje_w_grupie)}):")
    print(", ".join(kraje_w_grupie))

print("\n" + "="*45)

profilowanie = df.groupby('Grupa_KMeans')[['x1_PKB', 'x3_Ubowstwo_en', 'x14_Auta_EV', 'x20_Wydatki_Eko']].mean()
profilowanie.index = [f"Grupa {i+1}" for i in range(LICZBA_SKUPIEN)]

print("PROFILOWANIE GRUP (Średnie wartości wybranych cech):")
print(profilowanie.round(2).to_string())
print("="*45)

# ====================================================================
# WIZUALIZACJA: WYKRES PROFILI GRUP (ODCHYLENIA OD ŚREDNIEJ UE)
# # ====================================================================
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np

# if 'kmeans' in locals():
#     df['Skupienie'] = kmeans.labels_
#     df_scaled['Skupienie'] = kmeans.labels_
# else:
#     # Alternatywnie, jeśli używasz fit_predict w locie:
#     etykiety_skupien = kmeans.fit_predict(df_scaled[zmienne_ranking])
#     df['Skupienie'] = etykiety_skupien
#     df_scaled['Skupienie'] = etykiety_skupien

# # Wybieramy kluczowe zmienne reprezentatywne dla profilu
# wybrane_zmienne = ['x1_PKB', 'x3_Ubowstwo_en', 'x14_Auta_EV', 'x20_Wydatki_Eko']

# # Obliczamy średnie wartości ustandaryzowane dla każdego skupienia (współrzędne centroidów)
# profile_grup = df_scaled.groupby('Skupienie')[wybrane_zmienne].mean().reset_index()

# # Przekształcamy ramkę danych do formatu "long" na potrzeby biblioteki seaborn
# profile_long = profile_grup.melt(id_vars='Skupienie', var_name='Zmienna', value_name='Odchylenie_od_sredniej')

# # Mapujemy surowe kody zmiennych na pełne, czytelne nazwy do wykresu
# mapa_nazw = {
#     'x1_PKB': 'PKB per capita\n(x1)',
#     'x3_Ubowstwo_en': 'Ubóstwo energetyczne\n(x3)',
#     'x14_Auta_EV': 'Samochody elektryczne\n(x14)',
#     'x20_Wydatki_Eko': 'Wydatki na żywność EKO\n(x20)'
# }
# profile_long['Zmienna'] = profile_long['Zmienna'].map(mapa_nazw)

# # Mapujemy numery skupień na oficjalne nazwy makroregionów do legendy
# # [UWAGA: Jeśli w Twojej konsoli numery grup przypisały się inaczej (np. Polska ma numer 0), podmień cyfry w słowniku poniżej]
# mapa_skupien = {
#     0: 'Grupa 1: Zielona Elita Zachodu',
#     1: 'Grupa 2: Goniący Środek (w tym Polska)',
#     2: 'Grupa 3: Obszar Barier Strukturalnych'
# }
# profile_long['Skupienie'] = profile_long['Skupienie'].map(mapa_skupien)

# # Definiujemy spójną, sztywną paletę kolorów (Zielony = Sukces, Niebieski = Środek, Czerwony = Bariery)
# paleta_kolorow = {
#     'Grupa 1: Zielona Elita Zachodu': '#2ecc71',
#     'Grupa 2: Goniący Środek (w tym Polska)': '#3498db',
#     'Grupa 3: Obszar Barier Strukturalnych': '#e74c3c'
# }

# # Inicjalizacja płótna wykresu
# plt.figure(figsize=(13, 8))
# sns.set_theme(style="whitegrid")

# # Rysowanie głównego wykresu słupkowego
# ax = sns.barplot(
#     data=profile_long,
#     x='Zmienna',
#     y='Odchylenie_od_sredniej',
#     hue='Skupienie',
#     palette=paleta_kolorow,
#     edgecolor='black',
#     linewidth=0.8
# )

# # Pogrubiona, czarna linia pozioma na poziomie Y=0 (Średnia unijna)
# plt.axhline(0, color='#000000', linestyle='-', linewidth=1.8, alpha=0.9)
# plt.text(-0.43, 0.05, 'Średnia UE = 0', color='#000000', weight='bold', fontsize=10)

# # Stylizacja i formatowanie osi oraz tytułów
# plt.title('Profil ekonomiczno-środowiskowy wyodrębnionych grup państw UE\n(Wartości ustandaryzowane - odchylenia centroidów od średniej unijnej)', 
#           fontsize=14, pad=22, weight='bold', color='#1b5e20')
# plt.xlabel('Zmienne diagnostyczne', fontsize=12, labelpad=12, weight='bold', color='#263238')
# plt.ylabel('Odchylenie od średniej (w jednostkach odchylenia standardowego)', fontsize=12, labelpad=12, weight='bold', color='#263238')

# plt.xticks(fontsize=11, weight='bold')
# plt.yticks(fontsize=11)
# plt.grid(axis='y', linestyle=':', alpha=0.6, color='gray')

# # Konfiguracja estetyczna legendy
# plt.legend(title='Wyodrębnione makroregiony UE', title_fontsize=11, fontsize=10, loc='upper right', frameon=True, shadow=True)

# # Wyświetlenie gotowej grafiki
# plt.tight_layout()
# plt.show()

# ==========================================
# KROK 9: ANALIZA CZYNNIKOWA (FACTOR ANALYSIS)
# ==========================================

fa = FactorAnalysis(n_components=2, random_state=42)
fa_results = fa.fit_transform(df_scaled)

macierz_czynnikowa = pd.DataFrame(
    fa.components_.T, 
    columns=['Czynnik_1', 'Czynnik_2'], 
    index=zmienne_numeryczne.columns
)

print("\n" + "="*45)
print("=== WYNIKI ANALIZY CZYNNIKOWEJ ===")
print("="*45)
print("Ładunki czynnikowe dla ukrytych uwarunkowań:")
print(macierz_czynnikowa.round(3))
print("="*45)

print("\nWSKAZÓWKA DO INTERPRETACJI:")
print("- Czynnik o wysokich ładunkach przy PKB, PPS i Wydatkach Eko nazwij: 'Eko-zamożność konsumencka'.")
print("- Czynnik o wysokich ładunkach przy OZE i Eko uprawach nazwij: 'Strukturalny potencjał środowiskowy'.")
print("="*45)


# ==========================================
# DODATEK DO KROKU 3: KRYTERIUM KAISERA I WYKRES OSYPISKA
# ==========================================

wartosci_wlasne = pca.explained_variance_

full_pca = PCA()
full_pca.fit(df_scaled)
pelne_wartosci_wlasne = full_pca.explained_variance_

print("\n" + "="*45)
print("=== WARTOŚCI WŁASNE DLA KRYTERIUM KAISERA ===")
print("="*45)
for i, val in enumerate(pelne_wartosci_wlasne, 1):
    status = "AKCEPTUJEMY ( > 1 )" if val > 1 else "ODRZUCAMY ( < 1 )"
    print(f"PC{i:2d} | Wartość własna: {val:.3f} | Decyzja: {status}")
print("="*45)

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(pelne_wartosci_wlasne) + 1), pelne_wartosci_wlasne, marker='o', linestyle='-', color='b', label='Wartości własne PC')
plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Granica Kaisera (y = 1.0)')

plt.title('Wykres osypiska (Scree Plot) z granicą kryterium Kaisera', fontsize=14, pad=12)
plt.xlabel('Numer składowej głównej (PC)', fontsize=11)
plt.ylabel('Wartość własna (Eigenvalue)', fontsize=11)
plt.xticks(range(1, len(pelne_wartosci_wlasne) + 1))
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

