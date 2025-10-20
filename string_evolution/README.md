# String Evolution Demo

This project uses evolution algorithm to progress from initially random set of genotypes generating a random population of strings to a target string.
A direct approach would be to use a string directly as the gene which is being mutated and evaluated for fitness.
However, in real multicellular organisms genes only influence the organism's fitness through many layers of interaction.
The gene itself performs generation of chemicals which form complex structures or are fed to existing structures to produce more chemicals and more complex structures on so on.
This project simulates this indirect influence between genes and fitness to demonstrate how evolution can achieve a complex target even through many layers of indirection.

The used gene is structured as a set of source characters and multiple levels of indexes.
Each lavel contains indexes addressing the next level until the last level which contains indexes to characters.
This leads to some parts of the gene having massive influence on the resulting string while others may not have any influence at all.
