class Man:

    def __init__(self, name):
        self.name = name
        print('Initialized!!')

    def hello(self):
        print('Hello, ' + self.name + '!')

    def goodby(self):
        print('Good-bye, ' + self.name + '!')


man = Man('sample-man')
man.hello()
man.goodby()
