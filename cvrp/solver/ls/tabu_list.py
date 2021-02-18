from typing import List, Set


class TabuList:
    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self.__list: List[str] = []
        self.__index: Set[str] = set()

    def add_all(self, items: List[str]):
        for item in items:
            self.add(item)

    def add(self, item: str):
        if item in self.__index:
            return

        while len(self.__index) >= self.capacity:
            self._pop()

        self.__index.add(item)
        self.__list.insert(0, item)

    def _pop(self):
        item = self.__list.pop()
        self.__index.remove(item)

    def contains(self, item: str) -> bool:
        return item in self.__index
