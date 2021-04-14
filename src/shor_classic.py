# python3
"""Example: Shor's algorithm for factorization - using classic order finding."""

import math
import random

from absl import app


def is_prime(num):
  """Check to see whether num can be factored at all."""

  for i in range(3, num // 2, 2):
    if num % i == 0:
      return False
  return True


def is_coprime(num, larger_num):
  """Determine if num is a coprime to larger_num."""

  return math.gcd(num, larger_num) == 1


def get_odd_non_prime(fr, to):
  """Get a non-prime number in the range."""

  while True:
    n = random.randint(fr, to)
    if n % 2 == 0:
      continue
    if not is_prime(n):
      return n


def get_coprime(larger_num):
  """Find a numnber < larger_num which is coprime to it."""

  while True:
    val = random.randint(3, larger_num - 1)
    if is_coprime(val, larger_num):
      return val


def classic_order(num, modulus):
  """Find the order classically via simple iteration."""

  order = 1
  while True:
    newval = (num ** order) % modulus
    if newval == 1:
      return order
    order += 1


def run_experiment(fr, to):
  """Run the classical part of Shor's algorithm."""

  n = get_odd_non_prime(fr, to)
  a = get_coprime(n)
  order = classic_order(a, n)

  factor1 = math.gcd(a ** (order // 2) + 1, n)
  factor2 = math.gcd(a ** (order // 2) - 1, n)
  if factor1 == 1 or factor2 == 1:
    return None

  print('Found Factors: N = {:4d} = {:4d} * {:4d} (r={:4})'.
        format(factor1 * factor2, factor1, factor2, order))
  if factor1 * factor2 != n:
    raise AssertionError('Invalid factoring')

  return (factor1, factor2)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Classic Part of Shors Algorithm.')
  for i in range(25):
    run_experiment(21, 9999)


if __name__ == '__main__':
  app.run(main)
