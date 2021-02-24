from typing import List, Set, Dict


class TabuList:
    def __init__(self, capacity: int = 100, retain=3) -> None:
        self.capacity = capacity
        self.retain = retain

        self.__list: List[str] = []
        self.__index: Set[str] = set()
        self.__remaining: Dict[str, int] = {}

    def add_sets(self, item_sets: List[Set[str]]):
        for item_set in item_sets:
            self.add_all(list(item_set))

    def add_all(self, items: List[str]):
        if len(items) > self.capacity:
            items = items[len(items) - self.capacity :]

        for item in items:
            self.add(item)

    def add(self, item: str):
        if item in self.__index:
            return

        while len(self.__index) >= self.capacity:
            self._pop()

        self.__index.add(item)
        self.__list.insert(0, item)
        self.__remaining[item] = self.retain

    def _pop(self):
        item = self.__list.pop()
        self.__index.remove(item)
        self.__remaining.pop(item)

    def contains(self, item: str) -> bool:
        return item in self.__index

    def intersects(self, item_set: Set[str]) -> bool:
        return len(self.__index.intersection(item_set)) > 0

    def step(self):
        for k in self.__remaining.keys():
            self.__remaining[k] -= 1

        new_remaining = {}
        for k, v in self.__remaining.items():
            if v <= 0:
                self.__index.remove(k)
                self.__list.remove(k)
            else:
                new_remaining[k] = v

        self.__remaining = new_remaining

    def empty(self):
        self.__list = []
        self.__index = set()
        self.__remaining = {}
