with open("configurations.sh", "w") as f:
    for n in range(1, 11):
      for s in range(30):
        print(f"-n {n} -d 10 50 100 500 1000 -o results/n{n}_s{s}.json -s {s} -e 0 10 25", file=f)
