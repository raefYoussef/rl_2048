Reward Functions
1. Reward for new tile, and gamma of .9, num of updates is 100
   1. 13.54% win rate
2. Same as #1, increase gamma to .99, change num of updates to 20
   1. 19.32% win rate
3. Same as #2, increase num updates to 50
   1. Better performance, 29.09%. Shows sign of better performance with more epochs
4. Same as #3, gamma = .99, increase num updates to 100
   1. Squeezed better performance, 30.36%. 
5. Same as #4, gamma = .999
   1. worse performance, 28.25%.
6. Same as #5, lr 1e-4 -> 1e-3
   1. Oscillates and did not converge
7. Same as #5, lr 1e-4 -> 1e-5, updates 100 -> 1000
   1. Too slow and did not converge