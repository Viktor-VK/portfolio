# -*- coding: utf-8 -*-
"""Рисует схему разбора причин дефицита (в нотации BPMN) в схема_разбора.png. Запуск: python схема_разбора.py"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

ERR_DIST, ERR_BUY, NO_ERR = '#2a9d8f', '#f4a261', '#9e9e9e'

fig, ax = plt.subplots(figsize=(12, 11.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 11.5)
ax.axis('off')

X, XE = 3.2, 9.0          # колонка проверок и колонка итогов
YS = [10.6, 8.9, 7.0, 5.1, 3.2, 1.2]


def arrow(x0, y0, x1, y1, text=None, dx=0.12, dy=0.12):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=1.4, color='#333'))
    if text:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, text, fontsize=10, color='#333', fontweight='bold')


def gateway(y, text):
    w, h = 1.25, 0.8
    ax.add_patch(Polygon([(X, y + h), (X + w, y), (X, y - h), (X - w, y)], closed=True, fc='#fff8e1', ec='#333', lw=1.4))
    ax.text(X, y, '×', ha='center', va='center', fontsize=20, color='#333')
    ax.text(X - w - 0.2, y, text, ha='right', va='center', fontsize=10.5, wrap=True)


def end(y, title, text, color):
    ax.add_patch(Circle((XE - 1.1, y), 0.28, fc='white', ec=color, lw=4))
    ax.add_patch(FancyBboxPatch((XE - 0.6, y - 0.55), 3.4, 1.1, boxstyle='round,pad=0.05', fc=color, ec='none', alpha=0.18))
    ax.text(XE - 0.45, y + 0.2, title, ha='left', va='center', fontsize=10.5, fontweight='bold', color='#222')
    ax.text(XE - 0.45, y - 0.22, text, ha='left', va='center', fontsize=9.5, color='#333')


# старт
ax.add_patch(Circle((X, YS[0]), 0.3, fc='white', ec='#333', lw=1.6))
ax.text(X - 0.5, YS[0], 'Дефицит в магазине:\nвесь день без товара\nили закончился за день', ha='right', va='center', fontsize=10.5)
arrow(X, YS[0] - 0.3, X, YS[1] + 0.8)

gateway(YS[1], 'Остатка склада накануне\nвечером хватало, чтобы\nдовезти этому магазину?')
gateway(YS[2], 'В других магазинах этого склада\nзапас больше недели продаж?')
gateway(YS[3], 'Поставщик привёз весь\nпоследний заказ\n(или заказа не было)?')
gateway(YS[4], 'В других регионах запас\n(склад + магазины) больше\n2 недель продаж региона?')

for i in (1, 2, 3):
    arrow(X, YS[i] - 0.8, X, YS[i + 1] + 0.8, 'нет')
arrow(X, YS[4] - 0.8, X, YS[5] + 0.62, 'нет')

end(YS[1], 'Ошибка распределения', 'склад не довёз товар в магазин', ERR_DIST)
end(YS[2], 'Перекос распределения', 'склад развёз товар неравномерно', ERR_DIST)
end(YS[3], 'Ошибка закупа', 'заказали недостаточно, хотя\nпоставщик мог дать больше', ERR_BUY)
end(YS[4], 'Перекос закупа', 'дефицитный товар неравномерно\nраспределён между складами', ERR_BUY)
for i in (1, 2, 3, 4):
    arrow(X + 1.25, YS[i], XE - 1.38, YS[i], 'да', dx=-0.2, dy=0.12)

# чистый дефицит - не ошибка
ax.add_patch(Circle((X, YS[5]), 0.28, fc='white', ec=NO_ERR, lw=4))
ax.add_patch(FancyBboxPatch((X + 0.5, YS[5] - 0.55), 3.9, 1.1, boxstyle='round,pad=0.05', fc=NO_ERR, ec='none', alpha=0.18))
ax.text(X + 0.65, YS[5] + 0.2, 'Дефицит у поставщика', ha='left', va='center', fontsize=10.5, fontweight='bold')
ax.text(X + 0.65, YS[5] - 0.22, 'товара не хватило на всех - не ошибка', ha='left', va='center', fontsize=9.5)

ax.text(0.1, 11.2, 'Разбор причины дефицита в магазине', fontsize=14, fontweight='bold')

out = Path(__file__).with_suffix('.png')
fig.savefig(out, dpi=110, bbox_inches='tight', facecolor='white')
print(out)
