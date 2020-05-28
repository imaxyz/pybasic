from abc import ABCMeta, abstractmethod
from enum import Enum, auto

class Animal(metaclass=ABCMeta):

    # クラス変数
    type_name: str = '<>'

    # クラス変数
    items = []

    # 抽象メソッドを定義
    @abstractmethod
    def cry(self):
        pass


class Dog(Animal):
    def __init__(self, name: str = 'デフォルト名称'):
        self.name = name
        self.__weight = 10.0     # プライベート変数はダブルアンダースコアで定義

    def __del__(self):
        print('Dogのデストラクタ')

    def __repr__(self):
        return 'Dog object'

    def cry(self):
        print(f'{self.name} 「わん」')
        self.items.append('わん')


class Cat(Animal):
    def __del__(self):
        print('Catのデストラクタ')

    def __repr__(self):
        return 'Cat object'

    def cry(self):
        print(f'{self.type_name} 「にゃー」')
        self.items.append('にゃー')


class AnimalFactory:

    @staticmethod
    def create_animal() -> Animal:
        """Animalを実装する、何かのオブジェクトを生成する"""
        import random

        if random.random() > 0.5:
            return Dog()
        else:
            return Cat()

class Week(Enum):
    monday = auto()
    tuesday = auto()
    wednesday = auto()
    thursday = auto()
    friday = auto()
    saturday = auto()
    sunday = auto()

def main():
    factory = AnimalFactory()
    some_animal = factory.create_animal()
    some_animal2 = factory.create_animal()
    some_animal3 = factory.create_animal()
    print(some_animal, 'was created.')

    some_animal.cry()
    some_animal2.cry()
    some_animal3.cry()

    print('items:', some_animal.items)

    print(Week.thursday.name, Week.thursday.value)


if __name__ == '__main__':
    main()
