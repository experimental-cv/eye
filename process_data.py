#!/usr/bin/env python3

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-s', '--savgol-width', default=510, type=int)
    args = parser.parse_args()

    import pandas as pd
    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.filename, index_col=0)
    df["Timestamp"] = pd.to_datetime(df['Timestamp'], errors='coerce', format='ISO8601')
    x = df["Timestamp"]
    x = (x - x[0]).dt.total_seconds()
    y = df["Pupil/Iris"]
    yhat = savgol_filter(y, args.savgol_width, 3)
    plt.plot(x, y)
    plt.plot(x, yhat)
    plt.xlabel("Seconds")
    plt.ylabel("Pupil/Iris")
    plt.title("Pupil dilation vs time")
    plt.show()
