import os
import yaml


def test_config(config_path):
    with open(os.path.join(config_path)) as file:
        config = yaml.safe_load(file)

    def add(a, b):
        return a + b

    # print(config)
    c = add(**config["beta"])
    print(c)


class Test:
    def __init__(self):
        self.a = 1
        print(self.__getattribute__("b"))


class A:
    def __init__(self, x):
        self.x = x
        print("A")


class B:
    def __init__(self, x):
        self.x = x
        print("B")


if __name__ == "__main__":
    # test_config("../config/config.yaml")
    test = Test()
    # print(test.a)
    # print(getattr(test, "b", None))
    # Class = A
    # class_ = Class(1)
    # print(class_.x)
