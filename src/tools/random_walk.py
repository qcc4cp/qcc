# python3
"""Example: Simple classic random walk."""

import random
from absl import app

# This little hacktastic tool simulates a classical
# walk. While it is fun to play around with the parameters
# it essentially produces a result that mirrors the
# distribution of the random number generator, which
# is random.gauss at this point.


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  n_steps = 50
  n_collections = 10000
  final_pos = [0] * 2 * n_steps
  print('Simulate random walk, collect distribution...')
  for i in range(n_collections):
    for step in range(n_steps):
      direction = random.gauss(0, 10)
      step += int(direction)
    if step > 0 and step < 2 * n_steps:
      final_pos[step] += 1

  max_elem = 0
  for i in range(len(final_pos)):
    if final_pos[i] > max_elem:
      max_elem = final_pos[i]

  index = 0
  for index in range(n_steps * 2):
    if index == n_steps-1:
      continue
    print('{} {:.6f}'.format(index, 1.0 / max_elem * final_pos[index]))
    index = index + 1


if __name__ == '__main__':
  app.run(main)
