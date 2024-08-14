import numpy as np
from os import getcwd
from math import floor

COUNTER = 0
    
##############################################
def read_puzzle(txt_file, puzzle_id):
    
    f = open(txt_file, "r")
    txt = f.read()
    
    start_idx = int(98*(puzzle_id-1)+8)
    end_idx = start_idx + 89
    puzzle = txt[start_idx:end_idx]
    
    puzzle_arr = np.zeros((9, 9), dtype=int)
    for row_start in range(0, 81, 10):
        row = puzzle[row_start:row_start+9]
        for j in range(9):
            puzzle_arr[floor(row_start/10), j] = row[j]
    
    print("========== Successfully read puzzle from txt file ==========")
    
    return puzzle_arr


##############################################
def is_valid(puzzle:np.ndarray):
    # check rows
    puz = puzzle
    for row in puz:
        non_zero_vals = row[row>0]
        u = np.unique(row)
        if len(non_zero_vals) != len(u)-1:
            return False
    # check columns
    puz = puzzle
    puz.transpose()
    for col in puz:
        non_zero_vals = col[col>0]
        u = np.unique(col)
        if len(non_zero_vals) != len(u)-1:
            return False
    # check 3x3 boxes
    puz = puzzle
    for row_start in range(0, 8, 3):
        for col_start in range(0, 8, 3):
            box = puz[row_start:row_start+3, col_start:col_start+3]
            box = box.flatten()
            non_zero_vals = box[box>0]
            u = np.unique(box)
            if len(non_zero_vals) != len(u)-1:
                return False
    
    # all checks passed, puzzle state is valid
    return True
    

##############################################
def is_complete(puzzle:np.ndarray):
    full_row_col = np.linspace(1, 9, 9, dtype=int)
    # check if all entries are filled
    if np.any(puzzle==0):
        return False
    # all entries are filled, now check whether each row is correct
    # check rows
    puz = puzzle.copy()
    for row in puz:
        row.sort()
        if np.sum(row==full_row_col) < 9:
            return False
    # check columns
    puz = puzzle.copy()
    puz = puz.transpose()
    for col in puz:
        col.sort()
        if np.sum(col==full_row_col) < 9:
            return False
    # check 3x3 boxes
    puz = puzzle.copy()
    for row_start in range(0, 8, 3):
        for col_start in range(0, 8, 3):
            box = puz[row_start:row_start+3, col_start:col_start+3]
            box = box.flatten()
            box.sort()
            if np.sum(box==full_row_col) < 9:
                return False
            
    # all checks passed, puzzle completed
    return True


##############################################
def is_valid_move(puzzle:np.ndarray, pos, n):
    # check row
    row = puzzle[pos[0]]
    if n in row:
        return False
    # check column
    col = puzzle[:, pos[1]]
    if n in col:
        return False
    # check 3x3 box
    row_start = floor(pos[0]/3)*3
    col_start = floor(pos[1]/3)*3
    box = puzzle[row_start:row_start+3, col_start:col_start+3]
    if n in box:
        return False
    # all checks passed
    return True


##############################################
def get_empty_pos(puzzle:np.ndarray):
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] == 0:
                return (i, j)
        
        
##############################################
def solve(puzzle:np.ndarray):
    
    global COUNTER
    
    # return True if puzzle is solved
    if is_complete(puzzle):
        return puzzle, True
    
    # loop through all moves at the next empty position
    pos = get_empty_pos(puzzle)
    # print("empty pos = (%d, %d)" % (pos[0], pos[1]))
    for num in range(1, 10):
        if is_valid_move(puzzle, pos, num):
            COUNTER += 1
            # print("Entering %d at position (%d, %d)" % (num, pos[0], pos[1]))
            new_puzzle = puzzle.copy()
            new_puzzle[pos[0], pos[1]] = num
            new_puzzle, solved = solve(new_puzzle)
            if solved:
                return new_puzzle, True
            
    return puzzle, False
    


##############################################
if __name__ == "__main__":
    
    puzzles_file_path = getcwd() + "\\sudoku\\puzzles.txt"
    puzzle_id = 8
    puzzle = read_puzzle(puzzles_file_path, puzzle_id)
    print(puzzle)
    
    res, solved = solve(puzzle)
    print("="*50)
    print("Finished solving Sudoku!")
    print(res)
    print(COUNTER)