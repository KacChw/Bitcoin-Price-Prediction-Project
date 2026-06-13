# ====================================================================
# WIZUALIZACJA: GRAFICZNA MACIERZ RANKINGÓW (GRADIENT OD WIERSZA)
# ====================================================================
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Dokładne przeliczenie surowych wyników dla 3 metod
df_h_in = df_scaled[zmienne_ranking]
wzorzec_h = {col: (df_h_in[col].max() if col in stymulanty else df_h_in[col].min()) for col in zmienne_ranking}
odleglosci_h = np.sqrt(((df_h_in - pd.Series(wzorzec_h)) ** 2).sum(axis=1))
miernik_hellwiga = 1 - (odleglosci_h / (odleglosci_h.mean() + 2 * odleglosci_h.std()))
r_h_tab = pd.DataFrame({'Kraj_H': kraje, 'Score_H': miernik_hellwiga}).sort_values(by='Score_H', ascending=False).reset_index(drop=True)
r_h_tab['Miejsce'] = r_h_tab.index + 1

df_t_in = zmienne_numeryczne[zmienne_ranking].copy()
for col in zmienne_ranking: df_t_in[col] = (df_t_in[col] - df_t_in[col].min()) / (df_t_in[col].max() - df_t_in[col].min())
w_t = pd.Series({col: (1.0 if col in stymulanty else 0.0) for col in zmienne_ranking})
a_t = pd.Series({col: (0.0 if col in stymulanty else 1.0) for col in zmienne_ranking})
miernik_topsis = np.sqrt(((df_t_in - a_t) ** 2).sum(axis=1)) / (np.sqrt(((df_t_in - w_t) ** 2).sum(axis=1)) + np.sqrt(((df_t_in - a_t) ** 2).sum(axis=1)))
r_t_tab = pd.DataFrame({'Kraj_T': kraje, 'Score_T': miernik_topsis}).sort_values(by='Score_T', ascending=False).reset_index(drop=True)
r_t_tab['Miejsce'] = r_t_tab.index + 1

df_c_in = zmienne_numeryczne[zmienne_ranking] / zmienne_numeryczne[zmienne_ranking].sum()
S_plus = df_c_in[stymulanty].sum(axis=1)
S_minus = df_c_in[destymulanty].sum(axis=1)
uzytecznosci = ((S_plus + (S_minus.sum() / (S_minus * (1 / S_minus).sum()))) / (S_plus + (S_minus.sum() / (S_minus * (1 / S_minus).sum()))).max()) * 100
r_c_tab = pd.DataFrame({'Kraj_C': kraje, 'Score_C': uzytecznosci}).sort_values(by='Score_C', ascending=False).reset_index(drop=True)
r_c_tab['Miejsce'] = r_c_tab.index + 1

# Połączenie struktur w jedną macierz uporządkowaną wg wierszy (od 1 do 27 miejsca)
df_matrix = pd.DataFrame({'Miejsce': range(1, 28)})
df_matrix = df_matrix.merge(r_h_tab[['Miejsce', 'Kraj_H', 'Score_H']], on='Miejsce')
df_matrix = df_matrix.merge(r_t_tab[['Miejsce', 'Kraj_T', 'Score_T']], on='Miejsce')
df_matrix = df_matrix.merge(r_c_tab[['Miejsce', 'Kraj_C', 'Score_C']], on='Miejsce')

# 2. Projektowanie układu graficznego (sztywne, wąskie kolumny wynikowe)
fig, ax = plt.subplots(figsize=(12, 14))
ax.set_xlim(0, 8.8)
ax.set_ylim(-28, 1.5)
ax.axis('off')

col_specs = [
    {'name': 'Miejsce',        'width': 0.6, 'x': 0.0, 'is_score': False},
    {'name': 'Kraj (Hellwig)', 'width': 1.8, 'x': 0.6, 'is_score': False},
    {'name': 'Wynik',          'width': 1.0, 'x': 2.4, 'is_score': True},
    {'name': 'Kraj (TOPSIS)',  'width': 1.8, 'x': 3.4, 'is_score': False},
    {'name': 'Wynik',          'width': 1.0, 'x': 5.2, 'is_score': True},
    {'name': 'Kraj (COPRAS)',  'width': 1.8, 'x': 6.2, 'is_score': False},
    {'name': 'Użyteczność',    'width': 1.0, 'x': 8.0, 'is_score': True}
]

# Pobranie palety gradientu z nowej, zalecanej lokalizacji w Matplotlib
cmap = plt.colormaps['YlGn']

# --- RYSOWANIE NAGŁÓWKA ---
for col in col_specs:
    rect = patches.Rectangle((col['x'], 0.2), col['width'], 0.8, facecolor='#1b5e20', edgecolor='#ffffff', linewidth=1)
    ax.add_patch(rect)
    ax.text(col['x'] + col['width']/2, 0.6, col['name'], color='white', weight='bold', fontsize=11, ha='center', va='center')

# --- RYSOWANIE WIERSZY (DANYCH) ---
for i in range(len(df_matrix)):
    row = df_matrix.iloc[i]
    y_pos = -i - 0.8
    
    # KLUCZOWA ZMIANA: Siła koloru zależy WYŁĄCZNIE od indeksu wiersza i (od 0 do 26)
    # Pierwsza kolumna tworzy idealny gradient, a kolejne kolumny wynikowe biorą dokładnie ten sam kolor w danym wierszu
    norm_wiersza = 0.1 + 0.7 * ((26 - i) / 26)
    color_gradientu = mcolors.to_hex(cmap(norm_wiersza))
    
    # Dla kolumn tekstowych stosujemy klasyczną, bardzo delikatną, neutralną szachownicę
    bg_text = '#ffffff' if i % 2 == 0 else '#f4fbf7'
    
    for col in col_specs:
        cell_width = col['width']
        x_pos = col['x']
        
        if not col['is_score']:
            val_color = bg_text
            text_color = '#000000'
            font_weight = 'normal'
            
            if col['name'] == 'Miejsce': text_val = str(int(row['Miejsce']))
            elif 'Hellwig' in col['name']: text_val = str(row['Kraj_H'])
            elif 'TOPSIS' in col['name']: text_val = str(row['Kraj_T'])
            elif 'COPRAS' in col['name']: text_val = str(row['Kraj_C'])
        else:
            # Wszystkie kolumny z wynikami w tym wierszu dostają identyczny kolor z gradientu
            val_color = color_gradientu
            text_color = '#000000' if norm_wiersza < 0.55 else '#ffffff'
            font_weight = 'bold'
            
            if 'Hellwig' in col['name'] or col['x'] == 2.4: text_val = f"{row['Score_H']:.4f}"
            elif 'TOPSIS' in col['name'] or col['x'] == 5.2: text_val = f"{row['Score_T']:.4f}"
            elif 'COPRAS' in col['name'] or col['x'] == 8.0: text_val = f"{row['Score_C']:.2f}%"
            
        # Rysowanie komórki
        rect = patches.Rectangle((x_pos, y_pos), cell_width, 0.95, facecolor=val_color, edgecolor='#e0e0e0', linewidth=0.5)
        ax.add_patch(rect)
        
        # Centrowanie tekstu
        ax.text(x_pos + cell_width/2, y_pos + 0.475, text_val, color=text_color, weight=font_weight, fontsize=10, ha='center', va='center')

plt.title('Macierz komparatywna wyników porządkowania liniowego państw UE', fontsize=14, pad=20, weight='bold', color='#1b5e20')
plt.tight_layout()
plt.show()