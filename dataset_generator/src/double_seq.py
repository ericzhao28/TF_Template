import random, csv
from . import config


def generate():
  '''
  Generate double sequence data
  '''

  X = []
  Y = []
  for i, j in enumerate(range(0, 10)):
    for _ in range(20):
      for ix, jx in enumerate(range(j, j + 400, 50)):
        for k in range(jx, 20 + jx):
          step = [i, ix]
          step.append(random.randrange(k, k + 7))
          step.append(random.randrange(0, 7))
          step.append(random.randrange(7, 9))
          step.append(random.randrange(k, k + 70))
          X.append(step)
          Y.append(j)

  return X, Y


def main():
  X, Y = generate()
  with open(config.DUMPS_DIR + "double_seq.csv", "w") as f:
    writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONE)
    writer.writerow("seq_number", "step_number", "small_relevant",
                    "small_random", "big_random", "big_relevant")
    for i in range(len(X)):
        writer.writerow(X[i] + [Y[i]])


if __name__ == "__main__":
  main()