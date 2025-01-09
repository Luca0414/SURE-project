from math import pi

with open("configurations.txt", "w") as f:
    for n in range(1, 11):
        for s in range(30):
            print(f"-n {n} -d 10 50 100 500 1000 -o results/n{n}_s{s}.json -s {s} -e 0 10 25", file=f)


# Covasim
beta = "x0"
avg_rel_sus = "x1"
avg_contacts_s = "x2"
avg_contacts_w = "x3"
avg_contacts_h = "x4"
target = f"y = 1921505.45 - 13973680.90*{beta} - 1033749.64*log({avg_rel_sus}) - 354743.44*I(log({avg_rel_sus}) * log({beta})) + 1548401.81*log({avg_contacts_s}) + 414910.43*I(log({avg_contacts_s}) * log({beta})) - 1107218.31*log({avg_contacts_w}) - 225605.30*I(log({avg_contacts_w}) * log({beta})) - 422024.01*log({avg_contacts_h}) - 137848.89*I(log({avg_contacts_h}) * log({beta}))"
