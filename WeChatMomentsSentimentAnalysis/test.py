if __name__ == "__main__":
    d = {'a':[1, 2], "b": [1, 2]}
    print(sum([len(value) for value in d.values()]))