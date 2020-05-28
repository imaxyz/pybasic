from abc import ABCMeta, abstractmethod


class Animal(metaclass=ABCMeta):

    # 抽象メソッドを定義
    @abstractmethod
    def cry(self):
        pass


class Dog(Animal):
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
