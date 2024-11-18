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
  9. End of game +/-(max_merge_reward), add non move penalty of -1, add delta score for each step. Use gamma of .8 to limit future rewards contribution to 10 past ones.
    1. Game moves was stable because of the penalty unlike other rewards with +/- value. Critic function was smooth. Success rate was high for long runs, ~78% total and ~83% for the last 1000 episodes. 
  10. Same as #9 but add reward for new max tile
    1. Roughly the same performance as #9.
  11. Same as #10 but increase gamma to .9 (range of 20 moves)
    1. Slightly worse performance. Seems to take longer to converge.
  12. Same as #10 but decrease gamma to .65 (range of 5 moves)
    1. About same performance.
  13. Same as #9 but remove max tile reward and add empty space differene (new - old) reward
    1. Slightly worse performance. 