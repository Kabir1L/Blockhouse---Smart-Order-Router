# Smart Order Router Backtest

This project is part of a technical task for a quantitative finance internship. The task is to simulate a **Smart Order Router (SOR)** that decides how to buy a large number of shares (5,000 shares) in the most cost-effective way. It also includes building a dashboard to visually explain how the strategy performed using market data.

---

## ✅ Summary of Completed Requirements

- ✔️ Used real market data (`l1_day.csv`) to simulate trading
- ✔️ Followed the provided allocator pseudocode
- ✔️ Implemented cost function with all parameters (`lambda_over`, `lambda_under`, `theta`)
- ✔️ Simulated execution and cost impact
- ✔️ Compared to Best Ask, TWAP, and VWAP strategies
- ✔️ Printed JSON summary of results
- ✔️ Saved PNG plot of cumulative cost over time
- ✔️ Saved CSV of costs over time
- ✔️ Built a Power BI dashboard with clear visuals
- 🌟 **Bonus:** Included explanations of financial terms and how slicers were used to drill into results

---

## 📌 Objective (Simple Version)

Imagine you want to buy 5,000 shares of a stock — but you don’t want to overpay. Instead of buying them all at once, you want a program that decides the best way to split the order across price levels to save money. That’s what this Smart Order Router does.

---

## 🧾 What Was Provided
- `l1_day.csv`: Real market data — contains bid/ask prices and trade events for every second
- A text file describing how the allocator (order splitter) should work
- A set of tasks including writing code and showing the results visually

---

## 🧠 Key Concepts (Simplified)

| Term | What It Means |
|------|----------------|
| **Smart Order Router (SOR)** | A tool that figures out where and how to buy shares the cheapest way |
| **Taker Fee** | You pay this when you buy shares immediately |
| **Maker Rebate** | You earn this when you place a buy order and wait for someone to sell to you |
| **Underfill** | You didn’t buy enough shares |
| **Overfill** | You bought more than needed |
| **Queue Risk (θ)** | The chance your order just sits there and doesn't get filled |

The program uses these ideas to choose the best way to split the order.

---

## 🧪 What This Project Does (Explained Simply)

1. **Reads the Market Data**
   - Loads the first snapshot of the order book to see available prices

2. **Tries Different Combinations**
   - The program tests hundreds of different ways to split the order (like trying every way to spend $5,000)

3. **Calculates the Cost**
   - For each method, it adds:
     - what you pay to buy immediately (plus taker fees)
     - what you might earn from placing a slower limit order (maker rebates)
     - any under/over-fill penalties
     - queue risk

4. **Simulates the Market**
   - It looks at how the trades actually happened to check if your slow orders would have been filled later

5. **Picks the Best Strategy**
   - It keeps track of the lowest total cost found

6. **Prints and Plots Results**
   - Saves a picture of the cost over time and a file with the final answer

---

## 📊 Power BI Dashboard

The dashboard was built to help visualize the behavior of the market and the performance of the router.

### Slicers (Filter Tools):
- **Time (`ts_event`)**: To zoom in on specific times
- **Action (`action`)**: To focus on trades (T), quotes (A), cancels (C), etc.
- **Side (`side`)**: Whether it was a buy or sell

### Visuals Built:
1. **Spread Over Time**
   - X: `ts_event`
   - Y: Average of `ask_px_00 - bid_px_00`
   - Shows how tight or wide the market was

2. **Mid Price Over Time**
   - X: `ts_event`
   - Y: Average of `(ask_px_00 + bid_px_00)/2`
   - Shows estimated fair value at each moment

3. **Trade Volume by Side**
   - X: `side` (buy or sell)
   - Y: Sum of `size`
   - Helps compare who was more aggressive

4. **Liquidity Imbalance**
   - X: `ts_event`
   - Y: `(bid_sz_00 - ask_sz_00) / (bid_sz_00 + ask_sz_00)`
   - Measures which side had more available shares

5. **Trade Count Over Time**
   - X: `ts_event`
   - Y: Count of rows where `action = 'T'`
   - Shows how busy the market was

---

## 📁 Files in This Project

| File | What It Does |
|------|--------------|
| `backtest.py` | The full code — allocates, simulates, calculates, and saves results |
| `l1_day.csv` | Real market data used as input |
| `results.json` | Summary output (how much money was spent and saved) |
| `results.csv` | Shows how the cost went up over time during the simulation |
| `results.png` | A graph comparing smart strategy vs basic one |
| Power BI file | (Optional) Dashboard to help visualize spread, volume, prices |

---

## ▶️ How to Run This Project

### 1. Open Terminal and Set Up Environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy matplotlib
```

### 2. Put your files together:
- Make sure `l1_day.csv` and `backtest.py` are in the same folder

### 3. Run it:
```bash
python backtest.py
```

It will generate the following files in the folder:
- `results.json` — text output with final results
- `results.csv` — cost over time
- `results.png` — picture of strategy cost vs baseline

---

## 🔢 Output Example
```json
{
  "best_parameters": {
    "lambda_over": 0.0,
    "lambda_under": 1.0,
    "theta_queue": 0.1
  },
  "tuned": {
    "cash_spent": 1113715.0,
    "avg_price": 222.743
  },
  "best_ask": {
    "cash_spent": 1114118.36,
    "avg_price": 222.823672
  },
  "savings_vs_best_ask_bps": 3.62
}
```
That means the Smart Order Router saved around 3.6 basis points (~$18) compared to buying everything instantly.

---

## 🏁 Final Thoughts

This project:
- ✅ Successfully implemented all required logic
- ✅ Matched real-world trading concepts
- ✅ Added visualization for business clarity
- ✅ Explained every part of the code and financial logic simply
- ✅ Went beyond with a full Power BI dashboard

It shows how code, finance, and data come together to make better trading decisions. Perfect for someone new to quant finance who wants a real-world example.

