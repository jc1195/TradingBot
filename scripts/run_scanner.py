"""Run the asset scanner to find the best trading opportunities."""
import sys
sys.path.insert(0, ".")

from src.trading_bot.scanner import AssetScanner, print_scan_report

print("Scanning all 20 watchlist assets...")
print("(Fetching data + news + computing indicators for each)")
print()

scanner = AssetScanner(period="6mo", interval="1d")
result = scanner.scan(top_n=8)
print_scan_report(result)
