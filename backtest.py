import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# Configuration for the simulation
ORDER_SIZE = 5000  # The number of shares we want to buy
TAKER_FEE = 0.003  # Fee for taking liquidity (market order)
MAKER_REBATE = 0.001  # Rebate for providing liquidity (limit order)

# Load the first snapshot (top-of-book info) from market data
# This will be used to make allocation decisions
data_iter = pd.read_csv('l1_day.csv', chunksize=1)
snapshot = next(data_iter)
ask_prices = []
ask_sizes = []

# Extract top 10 levels of ask prices and sizes from the snapshot
for lvl in range(10):
    price = snapshot[f'ask_px_{lvl:02d}'].values[0]
    size = snapshot[f'ask_sz_{lvl:02d}'].values[0]
    if np.isnan(price) or np.isnan(size):
        break
    ask_prices.append(price)
    ask_sizes.append(int(size))

# If there's not enough liquidity in the top 10 levels, add a virtual level
# This ensures we always have enough to fill the order
total_liquidity = sum(ask_sizes)
if total_liquidity < ORDER_SIZE:
    ask_prices.append(ask_prices[-1] + 0.01)
    ask_sizes.append(ORDER_SIZE - total_liquidity)

# Compute the cost of a given allocation
# Includes taker fee, maker rebate, underfill/overfill, and queue penalty
def compute_cost(split, prices, sizes, lam_over, lam_under, theta):
    executed = 0
    cost = 0.0
    for i, alloc in enumerate(split):
        price = prices[i]
        avail = sizes[i]
        execute_i = min(alloc, avail)
        executed += execute_i
        cost += execute_i * (price + TAKER_FEE)
        posted = alloc - execute_i
        if posted > 0:
            cost -= posted * MAKER_REBATE
    underfill = max(0, ORDER_SIZE - executed)
    overfill = max(0, executed - ORDER_SIZE)
    cost += theta * (underfill + overfill)
    cost += lam_under * underfill + lam_over * overfill
    return cost

# Search for the best allocation across venues using brute-force
# Recursively explore all combinations of how to allocate shares
def allocate(order_size, prices, sizes, lam_over, lam_under, theta):
    N = len(prices)
    step = 100
    best_cost = float('inf')
    best_split = None
    def recurse(idx, remaining, current_split):
        nonlocal best_cost, best_split
        if idx == N:
            if remaining > 0:
                return
            cost = compute_cost(current_split, prices, sizes, lam_over, lam_under, theta)
            if cost < best_cost:
                best_cost = cost
                best_split = current_split.copy()
            return
        alloc_max = remaining if idx == N - 1 else min(remaining, sizes[idx])
        for q in range(0, alloc_max + 1, step):
            current_split.append(q)
            recurse(idx + 1, remaining - q, current_split)
            current_split.pop()
    recurse(0, order_size, [])
    if best_split is None:
        best_split = [0] * N
        best_cost = compute_cost(best_split, prices, sizes, lam_over, lam_under, theta)
    return best_split, best_cost

# Grid search over penalty parameters
param_grid = [(lam_o, lam_u, theta) for lam_o in [0.0, 0.5, 1.0] for lam_u in [0.0, 0.5, 1.0] for theta in [0.0, 0.1, 0.5]]
best_params = None
best_realized_cost = float('inf')
best_split = None

# Load trade events (used to simulate market fills)
trade_events = []
for chunk in pd.read_csv('l1_day.csv', chunksize=100000):
    trades = chunk[(chunk['action'] == 'T') & (chunk['side'] == 'A')]
    for _, row in trades.iterrows():
        trade_events.append((row['ts_event'], str(row['price']), float(row['size'])))
trade_events.sort(key=lambda x: x[0])

# Simulate filling the order based on actual trade events
# Tracks how much is filled over time and the cost
def simulate_execution(split, prices, sizes):
    filled = 0
    total_cost = 0.0
    timeline = []
    posted_price = None
    posted_remaining = 0
    start_ts = trade_events[0][0]
    timeline.append((start_ts, total_cost))
    for i, alloc in enumerate(split):
        if alloc <= 0:
            continue
        price = prices[i]
        avail = sizes[i]
        exec_qty = alloc if alloc <= avail else avail
        if exec_qty > 0:
            total_cost += exec_qty * price
            total_cost += exec_qty * TAKER_FEE
            filled += exec_qty
        if alloc > avail:
            posted_price = price
            posted_remaining = alloc - avail
    if posted_remaining > 0:
        for t, trade_price, trade_size in trade_events:
            trade_price = float(trade_price)
            if trade_price <= posted_price:
                fill_qty = min(trade_size, posted_remaining)
                total_cost += posted_price * fill_qty
                total_cost -= fill_qty * MAKER_REBATE
                filled += fill_qty
                posted_remaining -= fill_qty
                timeline.append((t, total_cost))
            if posted_remaining <= 0:
                break
    df = pd.DataFrame(timeline, columns=["timestamp", "cumulative_cost"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df = df.dropna()
    df.to_csv("results.csv", index=False)
    return total_cost, filled

# Run the grid search and simulate each allocation
for lam_o, lam_u, theta in param_grid:
    split, exp_cost = allocate(ORDER_SIZE, ask_prices, ask_sizes, lam_o, lam_u, theta)
    realized_cost, filled_shares = simulate_execution(split, ask_prices, ask_sizes)
    if filled_shares < ORDER_SIZE:
        continue
    if realized_cost < best_realized_cost:
        best_realized_cost = realized_cost
        best_params = (lam_o, lam_u, theta)
        best_split = split

# Compute the final cost of the best allocation
lam_o, lam_u, theta = best_params
best_cost = best_realized_cost
best_avg_price = best_cost / ORDER_SIZE

# Compute Best Ask baseline cost (buy everything immediately)
remain = ORDER_SIZE
base_cost = 0.0
level = 0
while remain > 0 and level < len(ask_prices):
    take_qty = min(remain, ask_sizes[level])
    base_cost += take_qty * ask_prices[level]
    remain -= take_qty
    level += 1
if remain > 0:
    base_cost += remain * ask_prices[-1]
    remain = 0
base_cost += ORDER_SIZE * TAKER_FEE
base_avg_price = base_cost / ORDER_SIZE

# Load and plot cumulative cost over time
# Blue line = smart strategy, orange line = baseline
df = pd.read_csv("results.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
df = df.dropna()

plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["cumulative_cost"], label="Tuned Strategy", color="blue")
plt.axhline(y=base_cost, color='orange', linestyle='--', label="Best Ask Baseline")
plt.xlabel("Time")
plt.ylabel("Cumulative Cost")
plt.title("Cumulative Execution Cost Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results.png")

# Final JSON summary printed to stdout
output = {
    "best_parameters": {
        "lambda_over": lam_o,
        "lambda_under": lam_u,
        "theta_queue": theta
    },
    "tuned": {
        "cash_spent": round(best_cost, 2),
        "avg_price": round(best_avg_price, 6)
    },
    "best_ask": {
        "cash_spent": round(base_cost, 2),
        "avg_price": round(base_avg_price, 6)
    },
    "savings_vs_best_ask_bps": round((base_cost - best_cost) / base_cost * 10000, 2)
}
print(json.dumps(output, indent=2))
