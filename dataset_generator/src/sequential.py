import random, csv
from . import config

random.seed(0)


def generate():
  '''
  Generate single sequential data,
  of shape: (200, 7-12, 4).
  CSV as (200 * 7-12, 5).
  '''

  X = []
  Y = []
  for y in range(0, 100, 10):
    for example in range(20):
      seq_id = str(y) + ":" + str(example)
      for step in range(random.randrange(7, 12)):
        x = [seq_id]
        x.append(random.randrange(y + step, y + step + 4))
        x.append(random.randrange(0, 7))
        x.append(random.randrange(7, 9))
        x.append(random.randrange(y + step, y + step + 7))
        X.append(x)
        Y.append(y)

  return X, Y


def main():
  X, Y = generate()
  with open(config.DUMPS_DIR + "sequential.csv", "w") as f:
    writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONE)
    writer.writerow(["seq_number", "small_relevant", "small_random",
                    "big_random", "big_relevant", "label"])
    for i in range(len(X)):
        writer.writerow(X[i] + [Y[i]])


if __name__ == "__main__":
  main()
