"""Метрика реконструкції ланцюга.

Три незалежні числа, кожне відповідає на своє питання:

* `step_coverage`  — скільки кроків еталона взагалі трапились у ланцюгу
                     (повнота, порядок ігнорується);
* `ordered_coverage` — скільки з них лягли У ПРАВИЛЬНОМУ ПОРЯДКУ
                     (найдовша зростаюча підпослідовність позицій);
* `precision_against_gold` — яка частка вузлів ланцюга взагалі має стосунок
                     до еталона (скільки в ньому «води»).

Свідомо НЕ зводимо їх в один інтегральний бал. Зведення ховає компроміс:
довгий ланцюг легко набирає покриття й провалює точність, короткий — навпаки.
Три числа поруч чесніші за одне.

Обмеження, які варто памʼятати, читаючи результат:
* збіг — за токенами, тож синоніми («IIS 6.0» проти «Microsoft IIS») метрика
  не бачить і карає, як за помилку;
* простий шлях не виражає кон'юнкції передумов: якщо еталон вимагає A ТА B,
  ланцюг через саме лише A зарахується як покритий.
"""


def _matched_positions(chain, gold) -> list[int]:
    """Для кожного кроку еталона — позиція ПЕРШОГО вузла ланцюга, що його
    покриває, або -1. Порядок кроків зберігається."""
    positions = []
    for tokens in gold:
        hit = -1
        for i, node in enumerate(chain):
            if node in tokens:
                hit = i
                break
        positions.append(hit)
    return positions


def step_coverage(chain, gold) -> float:
    """Частка кроків еталона, покритих ланцюгом. Порядок не враховується."""
    if not gold:
        return 0.0
    matched = sum(1 for p in _matched_positions(chain, gold) if p >= 0)
    return matched / len(gold)


def ordered_coverage(chain, gold) -> float:
    """Частка кроків еталона, покритих У ПРАВИЛЬНІЙ ПОСЛІДОВНОСТІ.

    Беремо позиції покритих кроків у ланцюгу й шукаємо найдовшу неспадну
    підпослідовність. Завжди <= `step_coverage`: ті самі збіги, але з
    додатковою вимогою до порядку."""
    if not gold:
        return 0.0
    positions = [p for p in _matched_positions(chain, gold) if p >= 0]
    if not positions:
        return 0.0

    best = [0] * len(positions)
    for i, value in enumerate(positions):
        best[i] = 1
        for j in range(i):
            if positions[j] <= value and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
    return max(best) / len(gold)


def precision_against_gold(chain, gold) -> float:
    """Частка вузлів ланцюга, що зустрічаються хоч у якомусь кроці еталона.

    Низьке значення означає, що обхід притягнув багато сторонніх понять —
    ланцюг «правильний, але розбавлений»."""
    if not chain:
        return 0.0
    gold_tokens: set = set()
    for tokens in gold:
        gold_tokens |= set(tokens)
    if not gold_tokens:
        return 0.0
    hits = sum(1 for node in chain if node in gold_tokens)
    return hits / len(chain)
