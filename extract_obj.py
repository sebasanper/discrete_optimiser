with open("coords3x3_full_cython_random.dat", "r") as inp:
    with open("obj_functions.dat", "w") as out:
        for line in inp:
            cols = line.split()
            out.write("{} {}\n".format(float(cols[11]), float(cols[12])))
