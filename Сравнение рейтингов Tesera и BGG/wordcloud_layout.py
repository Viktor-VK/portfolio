"""Раскладка облака слов без внешних библиотек: по спирали Архимеда от центра, крупные слова первыми.

Общая шкала шрифта (max_count) позволяет строить несколько облаков в одном масштабе:
одинаковая частота = одинаковый размер слова. Координаты - в пикселях холста width x height,
y растёт вниз (как в SVG).
"""
import math

WIDTH, HEIGHT = 900, 520


def font_size(count, max_count, min_px=11, max_px=60):
    return min_px + (max_px - min_px) * (count / max_count) ** 0.8


def layout(counts, max_count, width=WIDTH, height=HEIGHT, char_w=0.66):
    """counts: {слово: частота}. Возвращает список (слово, частота, x, y, размер_шрифта).
    Слова, которые не поместились, пропускаются."""
    placed, boxes = [], []
    for word, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        fs = font_size(c, max_count)
        bw, bh = char_w * fs * len(word) + 10, 1.25 * fs + 4
        t = 0.0
        while t <= 400:
            x = width / 2 + 4.2 * t * math.cos(t)
            y = height / 2 + 2.6 * t * math.sin(t)
            box = (x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2)
            inside = box[0] > 0 and box[1] > 0 and box[2] < width and box[3] < height
            if inside and all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in boxes):
                boxes.append(box)
                placed.append((word, c, round(x, 1), round(y, 1), round(fs, 1)))
                break
            t += 0.12
    return placed


def lean(t_count, b_count, threshold=3):
    """Куда тяготеет тема: 'tesera', 'bgg' или 'even' (разница меньше threshold игр)."""
    if t_count - b_count >= threshold:
        return "tesera"
    if b_count - t_count >= threshold:
        return "bgg"
    return "even"
