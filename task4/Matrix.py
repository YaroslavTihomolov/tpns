import random


class Matrix:

    def __init__(self, height: int, width: int, value: [float]):
        if value:
            self.__value = value
        else:
            self.__value: [float] = [[random.randint(-5, 5) for _ in range(height)] for _ in range(width)]
        self.__height = height
        self.__width = width
        self.__size = height * width

    def convolution(self, other: 'Matrix'):
        sum: float = 0
        for i in range(self.__size):
            sum += self.__value[0] * other.__value[0]
        return sum