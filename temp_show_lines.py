from pathlib import Path

path = Path("src/ohlcv/clean_ohlcv.py")
start = 70
end = 200
with path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if i < start:
            continue
        if i > end:
            break
        print(f"{i:04d}: {line.rstrip()}")
