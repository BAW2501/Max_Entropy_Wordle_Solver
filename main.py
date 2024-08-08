import numpy as np
import pygame
import random
import sys
from Solver import WordleSolver

pygame.init()
pygame.display.set_caption("Wordle Clone")
WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()
FPS = 30

# Colors
GREY, DARK_GREY, WHITE, RED = (100, 100, 100), (20, 20, 20), (255, 255, 255), (255, 108, 108)
COLOR_INCORRECT, COLOR_MISPLACED, COLOR_CORRECT = (50, 50, 50), (255, 193, 53), (0, 185, 6)

# Game settings
NUM_ROWS, NUM_COLS = 6, 5
RECT_WIDTH, RECT_HEIGHT = 50, 50
DX, DY = 10, 10
X_PADDING, Y_PADDING = 5, 5

# Calculate base offsets
BASE_OFFSET_X = (WIDTH - (NUM_COLS * (RECT_WIDTH + DX) - DX)) // 2
BASE_OFFSET_Y = (HEIGHT - (NUM_ROWS * (RECT_HEIGHT + DY) - DY)) // 2

def draw_grid():
    rects = []
    for y in range(NUM_ROWS):
        row = []
        for x in range(NUM_COLS):
            x_pos = BASE_OFFSET_X + x * (RECT_WIDTH + DX)
            y_pos = BASE_OFFSET_Y + y * (RECT_HEIGHT + DY)
            pygame.draw.rect(SCREEN, GREY, (x_pos, y_pos, RECT_WIDTH, RECT_HEIGHT), 2)
            row.append((x_pos, y_pos))
        rects.append(row)
    return rects

def draw_title():
    font = pygame.font.Font(None, 65)
    title_surface = font.render("WORDLE", True, WHITE)
    SCREEN.blit(title_surface, (BASE_OFFSET_X + RECT_WIDTH, BASE_OFFSET_Y - (RECT_HEIGHT * 2)))
    pygame.draw.line(SCREEN, WHITE, (BASE_OFFSET_X - RECT_WIDTH, BASE_OFFSET_Y - RECT_HEIGHT), 
                     (BASE_OFFSET_X + (RECT_WIDTH * (NUM_COLS + 1)) + (DX * (NUM_COLS - 1)), BASE_OFFSET_Y - RECT_HEIGHT))

def draw_text(text, color, y_offset):
    font = pygame.font.Font(None, 40)
    text_surface = font.render(text, True, color)
    x_pos = BASE_OFFSET_X + (RECT_WIDTH * (NUM_COLS / 5))
    y_pos = BASE_OFFSET_Y + y_offset
    SCREEN.blit(text_surface, (x_pos, y_pos))

def main():
    letter_font = pygame.font.Font(None, 65)
    used_words, curr_word = [], ""
    word_count, curr_letter = 0, 0
    flag_win, flag_lose, flag_invalid_word, flag_not_enough_letters = False, False, False, False
    timer_flag_1, timer_flag_2 = 0, 0

    wordlist = np.loadtxt("data/english-hidden.txt", dtype=str)
    allowedlist = np.loadtxt("data/english-all.txt", dtype=str)
    guess_word = random.choice(wordlist)
    print(f'The word to guess is {guess_word}')

    solver = WordleSolver()
    hyp = solver.guess() if len(used_words) else "soare"
    #print(f'Try : {hyp}')
    solver.update(hyp, solver.evaluate(hyp, guess_word))

    while True:
        SCREEN.fill(DARK_GREY)
        rects = draw_grid()  # Draw the grid every frame
        draw_title()
        draw_text("Best Guess :" + hyp.upper(), WHITE, -(DY * 4))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if flag_win or flag_lose:
                    if event.key == pygame.K_r:
                        main()
                elif event.key == pygame.K_BACKSPACE and curr_word:
                    curr_word = curr_word[:-1]
                    curr_letter -= 1
                elif event.key == pygame.K_RETURN:
                    solver.update(hyp, solver.evaluate(hyp, guess_word))
                    hyp = solver.guess()
                    #print(f'Try : {hyp}')
                    if len(curr_word) == 5:
                        if curr_word.lower() in allowedlist:
                            used_words.append(curr_word)
                            word_count += 1
                            curr_word, curr_letter = "", 0
                        else:
                            flag_invalid_word, timer_flag_1 = True, 0
                    else:
                        flag_not_enough_letters, timer_flag_2 = True, 0
                elif len(curr_word) < NUM_COLS and event.unicode.isalpha():
                    curr_word += event.unicode.upper()
                    curr_letter += 1

        if flag_invalid_word:
            draw_text("Not in word list", RED, -(DY * 4))
            timer_flag_1 += 1
        if flag_not_enough_letters:
            draw_text("Not enough letters", RED, -(DY * 4))
            timer_flag_2 += 1
        if timer_flag_1 == 2 * FPS:
            flag_invalid_word, timer_flag_1 = False, 0
        if timer_flag_2 == 2 * FPS:
            flag_not_enough_letters, timer_flag_2 = False, 0

        if flag_win:
            draw_text("Correct! Press 'R' to play again", WHITE, (DY * 7) + (RECT_HEIGHT * NUM_ROWS))
        if flag_lose:
            draw_text("Try again! Press 'R' to play again", WHITE, (DY * 7) + (RECT_HEIGHT * NUM_ROWS))

        for i, letter in enumerate(curr_word):
            word_surface = letter_font.render(letter, True, WHITE)
            SCREEN.blit(word_surface, (rects[word_count][i][0] + X_PADDING, rects[word_count][i][1] + Y_PADDING))

        for word_index, word in enumerate(used_words):
            remaining_letters = list(guess_word)
            num_correct = 0
            same_indices = [i for i, (x, y) in enumerate(zip(guess_word, word.lower())) if x == y]
            
            for index in same_indices:
                num_correct += 1
                remaining_letters[index] = ""
                pygame.draw.rect(SCREEN, COLOR_CORRECT, (*rects[word_index][index], RECT_WIDTH, RECT_HEIGHT))
                letter_surface = letter_font.render(word[index], True, WHITE)
                SCREEN.blit(letter_surface, (rects[word_index][index][0] + X_PADDING, rects[word_index][index][1] + Y_PADDING))

            for letter_index, letter in enumerate(word):
                if letter_index not in same_indices:
                    rect = (*rects[word_index][letter_index], RECT_WIDTH, RECT_HEIGHT)
                    letter_lower = letter.lower()
                    color = COLOR_INCORRECT if letter_lower not in remaining_letters else COLOR_MISPLACED
                    pygame.draw.rect(SCREEN, color, rect)
                    if color == COLOR_MISPLACED:
                        remaining_letters[remaining_letters.index(letter_lower)] = ""
                    letter_surface = letter_font.render(letter, True, WHITE)
                    SCREEN.blit(letter_surface, (rects[word_index][letter_index][0] + X_PADDING, rects[word_index][letter_index][1] + Y_PADDING))

            if num_correct == 5:
                flag_win = True
            elif len(used_words) == NUM_ROWS:
                flag_lose = True

        pygame.display.update()
        CLOCK.tick(FPS)

if __name__ == "__main__":
    main()