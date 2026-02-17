"""Download all historical data."""
import sys
sys.path.insert(0, ".")

from src.trading_bot.data_cache import bulk_download, get_cache_stats

print("Downloading 2 years of daily data for all 20 assets...")
results = bulk_download(period="2y", interval="1d")
stats = get_cache_stats()
print(f"Done! {stats['files']} files, {stats['total_candles']} total candles, {stats['total_size_mb']} MB")
for sym, count in sorted(results.items()):
    print(f"  {sym:6s}: {count:4d} candles")
