import random, csv
from . import config


def generate():
  '''
  Generate flat data.
  '''

  X = []
  Y = []
  for i in range(0, 10):
    for _ in range(20):
      x = []
      x.append(random.randrange(i, i + 7))
      x.append(random.randrange(0, 7))
      x.append(random.randrange(7, 9))
      x.append(random.randrange(i, i + 70))
      X.append(x)
      Y.append(i)

  return X, Y


def main():
  X, Y = generate()
  with open(config.DUMPS_DIR + "flat.csv", "w") as f:
    writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_NON)
    writer.writerow("small_relevant", "small_random",
                    "big_random", "big_relevant")
    for i in range(len(X)):
        writer.writerow(X[i] + [Y[i]])


if __name__ == "__main__":
  main()
