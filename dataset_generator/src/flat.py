import random, csv
from . import config

random.seed(0)


def generate():
  '''
  Generate flat data, of shape: (200, 4).
  '''

  X = []
  Y = []
  for y in range(0, 100, 10):
    for example in range(20):
      x = []
      x.append(random.randrange(y, y + 4))
      x.append(random.randrange(0, 7))
      x.append(random.randrange(7, 9))
      x.append(random.randrange(y, y + 7))
      X.append(x)
      Y.append(y)

  return X, Y


def main():
  X, Y = generate()
  with open(config.DUMPS_DIR + "flat.csv", "w") as f:
    writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONE)
    writer.writerow(["small_relevant", "small_random",
                    "big_random", "big_relevant", "label"])
    for i in range(len(X)):
        writer.writerow(X[i] + [Y[i]])


if __name__ == "__main__":
  main()
