import random
import time
import math
import numpy as np
from BaseAI import BaseAI

directions = [0,1,2,3]

class IntelligentAgent(BaseAI):
    def getMove(self, grid):
        time_limit = 0.2
        start_time = time.process_time()
        alpha = float('-inf')
        beta = float('inf')
        depth = 2
        best_move = random.choice(directions)
        initial_moves = grid.getAvailableMoves()
        for move in initial_moves:
            """ if 4 < len(initial_moves) <= 6:
                depth = 3
            elif len(initial_moves) <= 4:
                depth = 2 """
            new_grid = move[1]
            utility = self.expectiminimax_alphabeta(new_grid, depth, alpha, beta, False)
            if utility > alpha:
                alpha = utility
                best_move = move[0]
            if time.process_time() - start_time > time_limit:
                return best_move
        return best_move
    
    def heuristic(self, grid):
        EMPTY_WEIGHT = 1.0
        MONOTONICITY_WEIGHT = 1.1
        SMOOTHNESS_WEIGHT = 1.1

        empty = (16 - len(grid.getAvailableCells()))**2
        monotonicity = self.calculateMonotonicity(grid)
        smoothness = self.calculateSmoothness(grid)

        return (MONOTONICITY_WEIGHT * monotonicity +
                SMOOTHNESS_WEIGHT * smoothness-
                EMPTY_WEIGHT * empty)

    def calculateMonotonicity(self, grid):
    # Check for monotonic properties in rows and columns
        map = grid.map
        column_map = np.transpose(map)
        l = r = u = d = 0
        for i in range(4):
            row = map[i]
            column = column_map[i]
            l_c, r_c = self.checkMonotonicity(row)
            u_c, d_c = self.checkMonotonicity(column)
            l += l_c
            r += r_c
            u += u_c
            d += d_c
        return max(l, r) + max(u, d)
    
    def checkMonotonicity(self, array):
        # Check if array is monotonic
        l = r = 0
        for i in range(3):
            mono_l = mono_r = 0
            if array[i] !=0 and array[i] <= array[i + 1]:
                mono_l += 1
                l += 4*(mono_l**2)
            else:
                l -= abs(array[i] - array[i + 1]) * 1.5
                mono_l = 0
            if array[i] >= array[i + 1]:
                mono_r += 1
                r += 4*(mono_r**2)
            else:
                r -= abs(array[i] - array[i + 1]) * 1.5
                mono_r = 0
        return l, r

    def smoothness(self, m, n):
            #return abs(math.log2(m) - math.log2(n))
            return abs(m - n)

    
    def calculateSmoothness(self, grid):
        # get rid of nested for loops to speed up
        n1, n2, n3, n4 = grid.map[0]
        n5, n6, n7, n8 = grid.map[1]
        n9, n10, n11, n12 = grid.map[2]
        n13, n14, n15, n16 = grid.map[3]
        for n in [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16]:
            if n == 0:
                n = 2
        smoothness = 0
        smoothness = smoothness - min(self.smoothness(n1, n2), self.smoothness(n1, n5)) - min(self.smoothness(n2, n1), self.smoothness(n2, n6), self.smoothness(n2, n3)) - min(self.smoothness(n3, n2), self.smoothness(n3, n7), self.smoothness(n3, n4)) - min(self.smoothness(n4, n3), self.smoothness(n4, n8))
        smoothness = smoothness - min(self.smoothness(n5, n1), self.smoothness(n5, n6), self.smoothness(n5, n9)) - min(self.smoothness(n6, n2), self.smoothness(n6, n5), self.smoothness(n6, n7), self.smoothness(n6, n10)) - min(self.smoothness(n7, n3), self.smoothness(n7, n6), self.smoothness(n7, n8), self.smoothness(n7, n11)) - min(self.smoothness(n8, n4), self.smoothness(n8, n7), self.smoothness(n8, n12))
        smoothness = smoothness - min(self.smoothness(n9, n5), self.smoothness(n9, n10), self.smoothness(n9, n13)) - min(self.smoothness(n10, n6), self.smoothness(n10, n9), self.smoothness(n10, n11), self.smoothness(n10, n14)) - min(self.smoothness(n11, n7), self.smoothness(n11, n10), self.smoothness(n11, n12), self.smoothness(n11, n15)) - min(self.smoothness(n12, n8), self.smoothness(n12, n11), self.smoothness(n12, n16))
        smoothness = smoothness - min(self.smoothness(n13, n14), self.smoothness(n13, n9)) - min(self.smoothness(n14, n13), self.smoothness(n14, n10), self.smoothness(n14, n15)) - min(self.smoothness(n15, n14), self.smoothness(n15, n11), self.smoothness(n15, n16)) - min(self.smoothness(n16, n15), self.smoothness(n16, n12))

        return smoothness
    
    def expectiminimax_alphabeta(self, grid, level, alpha, beta, maximizing_player=True):
        # Early termination if the level is 0 or no more moves are available
        if level == 0 or not grid.getAvailableCells():
            return self.heuristic(grid)

        if maximizing_player:
            # Maximization logic
            max_utility = float('-inf')
            for move in grid.getAvailableMoves():
                new_grid = move[1]
                utility = self.expectiminimax_alphabeta(new_grid, level - 1, alpha, beta, False)
                max_utility = max(max_utility, utility)
                alpha = max(alpha, utility)
                if beta <= alpha:
                    break
            return max_utility
        else:
            # Chance node logic
            min_utility = float('inf')
            for cell in grid.getAvailableCells():
                expected_utility = 0
                for probability, tile_value in ((0.1, 4), (0.9, 2)):
                    grid.setCellValue(cell, tile_value)
                    utility = self.expectiminimax_alphabeta(grid, level - 1, alpha, beta, True)
                    expected_utility += probability * utility
                    grid.setCellValue(cell, 0) # Undo move
                expected_utility /= 2.0  # Average utility of both possible new tiles
                min_utility = min(min_utility, expected_utility)
                beta = min(beta, expected_utility)
                if beta <= alpha:
                    break
            return min_utility