from abc import ABCMeta, abstractmethod


class Animal(metaclass=ABCMeta):

    # 抽象メソッドを定義
    @abstractmethod
    def cry(self):
        pass


class Dog(Animal):
    def __init__(self, name: str):
        self.name = name
        self.__weight = 10.0     # プライベート変数はダブルアンダースコアで定義

    def cry(self):
        print('わん')


class Cat(Animal):
    def cry(self):
        print('にゃー')


class AnimalFactory:
    @staticmethod
    def create_animal() -> Animal:
        import random

        if random.random() > 0.5:
            return Dog()
        else:
            return Cat()


factory = AnimalFactory()
some_animal = factory.create_animal()
some_animal.cry()
