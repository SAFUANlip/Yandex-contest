from typing import List


def get_min_changes(flights: List[List[int]], source: int, destination: int) -> int:
    '''
    Каждый аэропорт иожно считать вершиной, а аэропорты, входящие с ним в одни маршрут будут смежными вершинами.

    Решение будет заключатся в использовании обхода в ширину, для определния необходимого числа самолетов (маршрутов)
    Один уровень обхода в ширину - + 1 используемый маршрут.
    Т.к. вершина будет хранить смежные вершины со всех маршрутов, то будут рассмотрены все возможные способы начать
    полет (т.е. разные начальные маршруты)
    '''
    result = -1
    if source == destination:
        return 0

    # сохранение графа в виде массива смежности
    graph = {}
    for route in flights:
        for i in range(len(route)):
            if route[i] not in graph:
                graph[route[i]] = set()
            graph[route[i]].update(route)
            graph[route[i]].remove(route[i])

    queue = [(source, 0)]
    visited = set([source])

    # BFS
    while queue:
        current_airport, flights_count = queue.pop(0)
        if current_airport == destination:
            return flights_count

        for next_airport in graph.get(current_airport, []):
            if next_airport not in visited:
                visited.add(next_airport)
                queue.append((next_airport, flights_count + 1))

    return result


with open('input.txt', 'r') as f:
    r, s, t = f.readlines()
    flights = eval(r)
    source = eval(s)
    destination = eval(t)

result = get_min_changes(flights, source, destination)
with open('output.txt', 'w') as output:
    output.write(f'{result}\n')

