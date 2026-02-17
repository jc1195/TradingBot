"""Run parameter sweep to find optimal trading settings."""
import sys
sys.path.insert(0, ".")

from src.trading_bot.optimizer import run_parameter_sweep, print_sweep_report

# Test on a DIVERSE set: best current picks + historically good performers
# NIO (scanner top pick), RIOT (high vol, historically 50% WR),
# PLTR (historically 50% WR), NVDA (historically 60% WR), GME (volatile)
SYMBOLS = ["NIO", "RIOT", "PLTR", "NVDA", "GME"]
KINDS = ["stock"]

print("=" * 60)
print("PARAMETER OPTIMIZATION")
print(f"Assets: {SYMBOLS}")
print("Testing different TP/SL/gate/confidence combos...")
print("This will take ~30-60 minutes with your 3070")
print("=" * 60)
print()

results = run_parameter_sweep(
    symbols=SYMBOLS,
    kinds=KINDS,
    period="1y",
    interval="1d",
    starting_capital=500.0,
    tp_range=[5.0, 8.0, 12.0, 15.0],
    sl_range=[2.0, 3.0, 5.0],
    gate_range=[10, 15, 20],
    conf_range=[0.60, 0.65, 0.70],
    max_combos=24,
)

print_sweep_report(results)
