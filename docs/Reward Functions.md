Reward Functions
1. End of game award score for win and -score for loss
    1. Allows for really long runs and higher scores. The longer runs are concerning because they can reach 10K+ for a simple board (4,4,7). It tells me it's chasing the wrong thing like score.
 2. End of game award score, deduct -1 for each step
    1. Critic function is regular and converges quickly. It also tries to decrease number of moves but the score is not affected.
 3. End of game award score
    1. Critic function converges even quicker than #2. Score doesn't jump up much
 4. End of game award 1000 for win and -1000 for loss
    1. Allows for really long runs like #1. Unlike #1, the episode length are relatively stable. Furthermore, the average return is indicative that the runs end up in wins more often than losses.
 5. Award score delta at each step
    1. Critic learning is very easy. It doesn't necessarily translate to higher win or score increase.
 6. Award score delta at each step and add +/- fixed reward at game end
    1. Critic learning is very easy. It does better than #5.
 7. Award score delta at each step, add +/- fixed reward at game end, add penalty of -10 for each non-move
    1. Critic learning is very easy. Does slightly worse than #6 but takes fewer moves.
 8. End of game award +/- 1 for win/loss. Add penalty of .01 for non-moves
    1. Unlike #4, we don't see long runs and the performance isn't as good for win ratio