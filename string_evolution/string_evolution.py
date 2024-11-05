#!/bin/python3

from random import randint
import string

# Constants related to referential (target) string
REF_STRING = "".join([
    "  I           oooo           I  ",
    "  I      ooood II boooo      I  ",
    "     ooood     II     boooo     ",
    "    d        IIIIII        b    ",
    "    q        IIIIII        p    ",
    "     ooooq     II     poooo     ",
    "  I      ooooq II poooo      I  ",
    "  I           oooo           I  ",
    "                                ",
])
STRING_WIDTH = 32
#REF_STRING = "".join([
#    " a  ",
#    "  b ",
#    "   c",
#    "    ",
#])
#STRING_WIDTH = 4
STRING_HEIGHT = len(REF_STRING) // STRING_WIDTH

# Constants related to genome
CHARS = string.ascii_letters + " " + string.digits + string.printable
#CHARS = "abc "
while (len(CHARS) < len(REF_STRING) * 2):
    CHARS += CHARS
NUMBER_LIMIT = len(CHARS)
REFERENCE_LEVELS = 5
# Content of encoded gene:
#   [ <number of used characters>]
# + <all characters list>
# <number of reference levels> * (
#   [ <number of used indexes on level X> ]
# + <all level X indexes list>
# )
ENCODED_GENE_LENGTH = (1 + NUMBER_LIMIT) * (1 + REFERENCE_LEVELS)
MUTATIONS_NUM_MIN = 2
MUTATIONS_NUM_MAX = 16
MUTATION_SHIFT_MAX = 20 # Either + or -

# Constants related to evolution
POPULATION_SIZE = 100
SELECTION_SIZE = 5
POPULATION_PER_SELECTED = POPULATION_SIZE // SELECTION_SIZE
GENERATIONS_NUM = 10000

class Genome:
    def __init__(self, parent=None):
        self.gene = [0 for i in range(ENCODED_GENE_LENGTH)]
        if (parent == None):
            self.generate_randomly()
        else:
            self.generate_from_parent(parent)

    def generate_randomly(self):
        self.gene = [randint(0, NUMBER_LIMIT-1) for _ in range(ENCODED_GENE_LENGTH)]

    def generate_from_parent(self, parent):
        # Clone parent
        self.gene = parent.gene.copy()

        # Mutate parent
        mutations_num = randint(MUTATIONS_NUM_MIN, MUTATIONS_NUM_MAX)
        for i in range(mutations_num):
            shift = randint(-MUTATION_SHIFT_MAX, MUTATION_SHIFT_MAX)
            mutation_index = randint(0, ENCODED_GENE_LENGTH-1)
            self.gene[mutation_index] = (self.gene[mutation_index] + shift) % NUMBER_LIMIT

    def get_speciment(self):
        return generate_string(self.get_characters(), [self.get_indexes(i) for i in range(REFERENCE_LEVELS)])

    def get_characters(self):
        characters = self.gene[1 : 1 + self.gene[0]]
        characters = [to_char(c) for c in characters]
        return characters

    def get_indexes(self, level):
        level_start = (1 + NUMBER_LIMIT) * (level + 1)
        level_length = NUMBER_LIMIT
        if (level == 0):
            level_length = self.gene[level_start]
        return self.gene[level_start + 1 : level_start + 1 + level_length]

def to_num(char):
    return CHARS.index(char)

def to_char(num):
    return CHARS[num]

def generate_string(characters, index_levels):
    """
    Generate an output string by using an input string of characters
    and an array of arrays of indexes, where each array is indexing
    the next array and the last of them is indexing the characters.
    """
    if (len(characters) == 0):
        return ""

    output_string = ""
    indexes = index_levels[0]
    for i in indexes:
        index = i
        for ii in index_levels[1:]:
            index = ii[index % len(ii)]
        output_string += characters[index % len(characters)]
    return output_string

def evaluate_string(s):
    l_min = min(len(s), len(REF_STRING))

    points = 0

    for i in range(l_min):
        points += abs(to_num(s[i]) - to_num(REF_STRING[i]))

    for i in range(l_min, len(s)):
        points += to_num(s[i]) + 1

    for i in range(l_min, len(REF_STRING)):
        points += to_num(REF_STRING[i]) + 1

    return points

def print_string(s):
    out = ""
    for i in range(len(s)):
        out += s[i]
        if ((i+1) % STRING_WIDTH == 0):
            out += "\n"
    out += "\n"
    print(out)

def generate_population(parent, population_size):
    return [Genome(parent) for i in range(population_size)]

def evaluate_speciment(genome):
    return evaluate_string(genome.get_speciment())

def select_best_from_population(population, best_number):
    evaluated_population = []
    for p in population:
        evaluated_population.append((p, evaluate_speciment(p)))
    evaluated_population.sort(key=lambda x : x[1])
    evaluated_population = evaluated_population[:best_number]
    return [x[0] for x in evaluated_population]

def print_population(population):
    print("Population:")
    for w in population:
        c = "".join(w.get_characters())
        print(f"{len(c)} characters: |{c}|")
        for e in range(REFERENCE_LEVELS):
            i = [(f"{x:3}, ") for x in w.get_indexes(e)]
            print(f"{len(i)} indexes {e}:")
            print_string(i)
        s = w.get_speciment()
        print_string(s)
        p = evaluate_string(s)
        print(f"points: {p}")
        print("--------")

def generate():
    population = generate_population(Genome(), POPULATION_SIZE)
    for i in range(GENERATIONS_NUM):
        #if (i % 20 == 0):
        #    print_population(select_best_from_population(population, 5))
        selected = select_best_from_population(population, SELECTION_SIZE)
        if (evaluate_speciment(selected[0]) == 0):
            population = [selected[0]]
            break
        population = []
        for s in selected:
            population += generate_population(s, POPULATION_PER_SELECTED)

    winner = select_best_from_population(population, 1)
    print_population(winner)
    print(f"Generations used: {i+1}")
    return i+1

if (__name__ == "__main__"):
    for i in range(10):
        generate()