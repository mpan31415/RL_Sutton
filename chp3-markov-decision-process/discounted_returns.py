
rewards = [-1, 2, 6, 3, 2]
T = 5
gamma = 0.5


def compute_return(time_step) -> list:
    if time_step == T:
        return [0]
    else:
        lst = compute_return(time_step+1)
        last_return = lst[0]
        this_return = rewards[time_step] + gamma * last_return
        return [this_return] + lst

# for i in range(5):
returns = compute_return(0)
print(returns)
